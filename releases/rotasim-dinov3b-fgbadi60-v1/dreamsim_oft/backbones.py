"""
[Begin Work Zone]
The three vision towers, loaded and sliced down to image-only.

Design notes that came out of probing the real checkpoints (scripts/probe_*.py):

  * NONE of these has DreamSim's fused `qkv` Linear. They all use separate
    q_proj/k_proj/v_proj, so the OFT target list had to be rewritten per backbone.
  * The full Siglip2/MetaClip2 models demand input_ids on forward, so we slice off
    the text tower and keep image-only. We never train text, so this is pure win.
  * Each backbone wants DIFFERENT normalization. DreamSim did the same thing
    (a shared resize+ToTensor, then a per-model Normalize), so the dataset hands us
    [0,1] tensors at 224x224 and each Branch normalizes for itself.
  * Which feature to take is a real modelling choice, not a detail:
      - DINOv3 : CLS token (index 0). Seq is 1 CLS + 4 registers + 196 patches = 201.
      - SigLIP2: the MAP (attention-pooled) head output -- this is the text-aligned
                 embedding, and it is why SigLIP is here at all.
      - MetaCLIP2: CLS through `visual_projection` (768 -> 512), i.e. the joint
                 image-text space. This mirrors what DreamSim took from CLIP.

The [Perception Encoder](https://arxiv.org/abs/2504.13181) result -- that the best
embeddings are often not at the network output -- is why `layer` is a parameter here
rather than a hardcoded final layer.

usage:
    from dreamsim_oft.backbones import build_branches, DEFAULT_ENSEMBLE
    branches = build_branches(DEFAULT_ENSEMBLE, device="cuda")
    emb = branches[0].embed(x01)          # x01 in [0,1], (B,3,224,224)
[End Work Zone]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os

import torch
import torch.nn as nn

# The Qwen3.8-27B vision tower's own loader + patchify live in a sibling module: the
# weights come out of a llama.cpp mmproj GGUF, which AutoModel cannot read. The import is
# cheap (qwen_vision imports transformers/gguf lazily, inside the loader).
from .qwen_vision import (POOL_DIMS, QWEN38_MMPROJ_LOCAL, load_vision_tower,
                          qwen_vision_embed)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Qwen vision tower: max IMAGES per packed sequence (see Branch.embed -- the packed
# design is quadratic in batch*patches; native-512 full-microbatch forwards OOM a 24 GB
# card). 8 images x 1024 patches = 8k tokens per sequence. Env-overridable.
QWEN_FWD_CHUNK = int(os.environ.get("QWEN_FWD_CHUNK", "8"))

DINOV3_LOCAL = r"I:\Models\dinov3-vitb16-pretrain-lvd1689m"
# DINOv3 ViT-L/16. Same module layout as B (model.layer.<i>.attention.{q,k,v,o}_proj +
# model.layer.<i>.mlp.{up,down}_proj), so the SAME oft_prefix and the SAME six Linear
# suffixes apply -- no new OFT code. Config (verified from config.json): DINOv3ViTModel,
# hidden 1024, 24 layers, 16 heads, inter 4096, patch 16, use_gated_mlp=false, RoPE
# (rope_theta 100) so the resolution is flexible and no position-embedding surgery is needed.
DINOV3_L_LOCAL = r"I:\Models\dinov3-vitl16-pretrain-lvd1689m"
SIGLIP2_LOCAL = r"I:\Models\siglip2-base-patch16-224"
METACLIP2_LOCAL = r"I:\Models\metaclip-2-worldwide-b16"


@dataclass
class BackboneSpec:
    key: str
    repo: str
    embed_dim: int
    mean: tuple
    std: tuple
    kind: str                      # how to pull the embedding out
    oft_prefix: str                # attention block path, RELATIVE TO THE TOWER
    note: str = ""
    # Which layer to take. None = final. Kept as a knob on purpose.
    layer: int | None = None
    # The resolution the pretrained tower was trained at, and its patch grid stride.
    # Used to decide whether a requested image_size needs position-embedding surgery.
    native_size: int = 224
    patch_size: int = 16


def _perfect_grid(n: int) -> int | None:
    """Return g if n == g*g, else None."""
    if n <= 1:
        return None
    g = int(round(n ** 0.5))
    return g if g * g == n else None


def resize_position_embeddings(tower: nn.Module, image_size: int, patch_size: int = 16,
                               verbose: bool = True) -> bool:
    """Resize LEARNED absolute position tables to a new input size, in place.

    WHY THIS IS NEEDED: SigLIP2 and MetaCLIP2 do not merely dislike a non-224 input, they
    HARD-FAIL on it -- the position table has exactly 196 entries (14x14 patches) and the
    patch sequence indexes into it directly. Measured:
        SigLIP2  : RuntimeError: size of tensor a (1156) must match tensor b (196)
        MetaCLIP2: ValueError: Input image size (544*544) doesn't match model (224*224)
    DINOv3 needs none of this: its positions are RoPE, which is resolution-flexible, and it
    accepts 544 unchanged (also measured).

    The resample is bicubic over the 2-D patch grid, the scheme CLIP-family models use to
    accept non-native resolutions. A CLS row, when present, is preserved verbatim and only
    the patch rows are resampled.

    HONEST CAVEAT, and it matters: this makes those towers run at a resolution they were
    NEVER trained at, so their features are off-distribution. The adapters are 1.29M
    rotations on q/k/v and can only nudge them. A 544 result is therefore "what a 544 model
    reaches", NOT evidence that the frozen towers understand 544.

    Returns True if anything changed.
    """
    emb = getattr(tower, "embeddings", None)
    if emb is None or not hasattr(emb, "position_embedding"):
        if verbose:
            print("[pos-resize] no embeddings.position_embedding -> assuming positions are "
                  "relative (RoPE) and the tower is already resolution-flexible")
        return False

    pe = emb.position_embedding
    old = pe.weight.data
    n_old, dim = old.shape
    new_g = image_size // patch_size
    n_new = new_g * new_g

    grid_old = _perfect_grid(n_old)
    # A CLIP-style table is 1 + g*g rows (a CLS row then the patch grid); a SigLIP-style
    # table is exactly g*g patch rows. Decide from the ROW COUNT, never from the grid side
    # -- conflating those is what made an earlier version read SigLIP2's 196 patch rows as
    # a 14-row table with a phantom CLS row and reshape to a nonsense (4,4,dim).
    if grid_old is not None:
        rows, has_cls = n_old, False
    elif _perfect_grid(n_old - 1) is not None:
        rows, has_cls = n_old - 1, True
    else:
        raise RuntimeError(
            f"cannot infer a square patch grid from {n_old} position rows; expected either "
            f"g*g rows or 1+g*g rows")
    g = int(round(rows ** 0.5))

    if n_new == rows:
        return False  # already the requested size

    cls = old[:1] if has_cls else None
    patches = old[1:] if has_cls else old
    x = patches.reshape(g, g, dim).permute(2, 0, 1).unsqueeze(0)      # (1,dim,g,g)
    x = torch.nn.functional.interpolate(x, size=(new_g, new_g), mode="bicubic",
                                        align_corners=False)
    new_patches = x.squeeze(0).permute(1, 2, 0).reshape(n_new, dim)
    new_weight = torch.cat([cls, new_patches], 0) if has_cls else new_patches

    new_pe = nn.Embedding(new_weight.shape[0], dim)
    with torch.no_grad():
        new_pe.weight.copy_(new_weight)
    new_pe.weight.requires_grad_(False)
    emb.position_embedding = new_pe

    # position_ids is a plain buffer indexing that table; if it is left at its old length
    # the forward indexes out of range.
    if getattr(emb, "position_ids", None) is not None and emb.position_ids.numel() != new_weight.shape[0]:
        emb.position_ids = torch.arange(new_weight.shape[0]).unsqueeze(0).to(emb.position_ids.device)

    # SECOND, INDEPENDENT GATE. The embeddings module caches the expected input size and
    # asserts on it, so resizing the table alone is not enough -- MetaCLIP2 still raised
    # "Input image size (544*544) doesn't match model (224*224)" with a correct 1157-row
    # table. These four attributes are what the forward actually reads.
    for attr, val in (("image_size", image_size),
                      ("num_patches", n_new),
                      ("num_positions", new_weight.shape[0])):
        if hasattr(emb, attr):
            setattr(emb, attr, val)
    cfg = getattr(tower, "config", None)
    if cfg is not None and hasattr(cfg, "image_size"):
        cfg.image_size = image_size

    if verbose:
        print(f"[pos-resize] {n_old} -> {new_weight.shape[0]} position rows "
              f"({g}x{g} -> {new_g}x{new_g} patches{', CLS preserved' if has_cls else ''}) "
              f"for {image_size}px input")
    return True


def _resample_pos_weight(old: torch.Tensor, gh: int, gw: int,
                         verbose: bool = False) -> torch.Tensor:
    """Bicubically resample a SigLIP-style (g*g, dim) position table to (gh*gw, dim).

    SigLIP tables have exactly g*g patch rows (NO CLS row) -- the row-count logic in
    resize_position_embeddings is the authority on that. Returns the new weight; the
    caller owns installation. Used both by the one-shot square resize and by the
    lazy per-grid bake cache (baked_pos).
    """
    n_old, dim = old.shape
    g = int(round(n_old ** 0.5))
    if g * g != n_old:
        raise RuntimeError(
            f"_resample_pos_weight expects a square g*g table, got {n_old} rows")
    x = old.reshape(g, g, dim).permute(2, 0, 1).unsqueeze(0)          # (1,dim,g,g)
    x = torch.nn.functional.interpolate(x, size=(gh, gw), mode="bicubic",
                                        align_corners=False)
    return x.squeeze(0).permute(1, 2, 0).reshape(gh * gw, dim)


BACKBONES: dict[str, BackboneSpec] = {
    "dinov3_vitb16": BackboneSpec(
        key="dinov3_vitb16",
        repo=DINOV3_LOCAL,
        embed_dim=768,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        kind="cls",
        oft_prefix="model.layer.",
        note="self-supervised; CLS token, 12 layers, 4 register tokens",
    ),
    # ADDED 2026-10-01 (kerok: "try an arm with dinov3-L"). Registered properly instead of
    # using the frozen gate's in-memory shim, so a checkpoint's own config can rebuild it.
    # Frozen gate on this tower: val 70.19 vs B's 69.71 (+0.48pp, cluster CI [-2.00,+3.09],
    # deff 2.75, eff n 913, MDE ~2.5pp) => the frozen comparison is UNRESOLVED; any trained
    # result must be reported against BOTH that unresolved frozen pair and the DINOv3-B arm.
    "dinov3_vitl16": BackboneSpec(
        key="dinov3_vitl16",
        repo=DINOV3_L_LOCAL,
        embed_dim=1024,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        kind="cls",
        oft_prefix="model.layer.",
        note="self-supervised; CLS token, 24 layers, 4 register tokens, gated_mlp=false",
    ),
    "siglip2_base16": BackboneSpec(
        key="siglip2_base16",
        repo=SIGLIP2_LOCAL,
        embed_dim=768,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
        kind="siglip_map",
        # Relative to the tower: once sliced, `vision_model.` is gone and the tower
        # itself is the `tower` child, so names read `tower.encoder.layers.<i>...`.
        oft_prefix="encoder.layers.",
        note="sigmoid-loss CLIP successor; text-aligned MAP-pooled output",
    ),
    "metaclip2_b16": BackboneSpec(
        key="metaclip2_b16",
        repo=METACLIP2_LOCAL,
        embed_dim=512,
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        kind="metaclip_proj",
        oft_prefix="encoder.layers.",
        note="CLIP-family; CLS projected into the 512-d joint image-text space",
    ),
    # Qwen3.8-27B vision tower + merger, read out of the local llama.cpp mmproj GGUF
    # (888 MiB) -- the 55.6 GB / 18-shard HF repo is NOT needed for the image path.
    # `repo` is a GGUF FILE here, not a directory: Branch routes on kind=="qwen_vl_vis"
    # to qwen_vision.load_vision_tower instead of AutoModel.from_pretrained.
    #   * embed_dim is the DEFAULT pooling's dim (merged-5120); Branch.embed_dim resolves
    #     the per-branch truth from `qwen_pool`, because 1152 is also selectable.
    #   * native_size 768 / patch_size 16 are metadata only: positions are a 48x48
    #     learned table interpolated per grid + 2D RoPE, so 288 is in-distribution-ish
    #     and needs NO resize_position_embeddings / baked_pos / interpolate_pos.
    #   * oft_prefix is relative to the SLICED tower -> "tower.blocks." once wrapped.
    #     That prefix is SURFACE-DEFINING: the merger's linear_fc1/linear_fc2 share two
    #     of the four target suffixes and are excluded only by this prefix (110 vs 108
    #     tensors -- measured 2026-09-26).
    "qwen38_vit27": BackboneSpec(
        key="qwen38_vit27",
        repo=QWEN38_MMPROJ_LOCAL,
        embed_dim=5120,
        mean=SIGLIP_MEAN,
        std=SIGLIP_STD,
        kind="qwen_vl_vis",
        oft_prefix="blocks.",
        note="Qwen3.8-27B ViT+merger (460.7M, 27 layers, hidden 1152) from the local "
             "mmproj GGUF; MERGED-BLOCK patch order; mean-pooled tokens",
        native_size=768,
        patch_size=16,
    ),
}

DEFAULT_ENSEMBLE = ("dinov3_vitb16", "siglip2_base16", "metaclip2_b16")

# The 544-class pair: MetaCLIP2 dropped (weakest solo branch at 83.31% yet 71.1% of
# the concatenation norm -- measured, see MEMORY). DINOv3 = self-supervised RoPE,
# SigLIP2 = language-aligned; one of each kind, no redundancy.
ENSEMBLE_544 = ("dinov3_vitb16", "siglip2_base16")


class Branch(nn.Module):
    """One frozen vision tower + its normalization + its embedding extraction.

    544/non-square additions (audited 2026-09-24, scratch/audit544/):
      * `interpolate_pos` (SigLIP only): pass the tower's built-in
        `interpolate_pos_encoding=True` flag, which bicubically resamples the
        learned 14x14 absolute position table to ANY (H/16, W/16) grid per batch,
        non-square included, at ~0.04 ms. At 224 the flag early-returns and is
        bit-identical to the untouched-table path, so enabling it is a strict no-op
        for the old square runs. When this is on, resize_position_embeddings is
        deliberately NOT called -- the flag supersedes it.
      * `gradient_checkpointing`: trade ~30% compute for activation memory. Two
        consequences verified by probe: (a) HF checkpointing only ENGAGES when the
        tower module is in train() mode, so Branch.train() must stop pinning the
        tower to eval when this is on (dropout surface verified all-zero: DINOv3
        drop_path/dropout None/0.0, SigLIP attention_dropout 0.0); (b) with
        use_reentrant=False, gradients still reach every OFT adapter AND the input
        image through the frozen weights -- the VAE-loss use case survives.
      * `dinov3_rope_augment` (DINOv3 only): in train() mode DINOv3 stochastically
        jitters its RoPE patch coordinates (config.pos_embed_rescale, 2.0 in this
        checkpoint). True = keep it (a positional regularizer; the towers are then
        not bit-deterministic during training -- eval mode is always clean).
        False = force it off for determinism. None = leave the checkpoint as loaded.
    """

    def __init__(self, spec: BackboneSpec, dtype: torch.dtype = torch.float32,
                 image_size: int | None = None,
                 gradient_checkpointing: bool = False,
                 dinov3_rope_augment: bool | None = None,
                 interpolate_pos: bool = False,
                 baked_pos: bool = False,
                 qwen_pool: str = "merged",
                 qwen_tower_path: str | None = None):
        super().__init__()
        # Frozen-tower override for the qwen branch (release portability): an explicit
        # path (config key) beats the env, which beats the built-in gguf default; a
        # DIRECTORY routes to the HF-safetensors loader inside load_vision_tower.
        self._qwen_tower_path = qwen_tower_path
        self.spec = spec
        self.dtype = dtype
        self.image_size = image_size or spec.native_size
        self._qwen_pool = str(qwen_pool)   # only read by kind == "qwen_vl_vis"
        self._qwen_info = None

        if spec.kind == "qwen_vl_vis":
            # The weights are a llama.cpp mmproj GGUF, so there is no repo for AutoModel
            # to read and nothing for _slice() to slice. load_vision_tower is a strict
            # bijection gate: any unmatched / missing / extra / shape-mismatched tensor
            # RAISES instead of warning, because a mis-mapped tower still produces
            # plausible embeddings.
            # KEYWORD ARGS ON PURPOSE: the loader's first positional is vision_cfg, so
            # passing the path positionally would be silently read as the architecture.
            from .qwen_vision import TOKEN_POOL_MODES
            if self._qwen_pool not in POOL_DIMS and self._qwen_pool not in TOKEN_POOL_MODES:
                raise ValueError(f"unknown qwen_pool {self._qwen_pool!r}; "
                                 f"expected one of {sorted(POOL_DIMS)} "
                                 f"+ {sorted(TOKEN_POOL_MODES)}")
            _tp = self._qwen_tower_path or os.environ.get("QWEN_TOWER_PATH") or spec.repo
            tower, self._qwen_info = load_vision_tower(gguf_path=_tp, dtype=dtype)
            tower.eval()
            proj = None
        else:
            from transformers import AutoModel

            full = AutoModel.from_pretrained(spec.repo, dtype=dtype)
            full.eval()
            tower, proj = self._slice(full)
        self.tower = tower
        self.visual_projection = proj  # only used by metaclip_proj
        # Resolve the input resolution the tower will actually be used at. DINOv3 returns
        # False here (RoPE); SigLIP2/MetaCLIP2 get their position table resampled --
        # UNLESS interpolate_pos is on, in which case SigLIP handles any grid natively
        # at forward time and the table is left at its trained 14x14.
        # baked_pos (SigLIP only, supersedes both): the image is fed at its TRUE aspect
        # (no squash, no crop), and the 14x14 position table is bicubically resampled
        # ONCE PER GRID at first encounter into a cache; each forward just installs the
        # matching cached table + gate attributes (O(1), deterministic). The table is an
        # internal tile-coordinate cheat sheet -- the IMAGE pixels never pass through it.
        self._interp_pos = bool(interpolate_pos and spec.kind == "siglip_map")
        self._baked_pos = bool(baked_pos and spec.kind == "siglip_map")
        self._pos_cache: dict[tuple[int, int], dict] = {}
        if self._baked_pos:
            if self._interp_pos:
                raise ValueError("baked_pos and interpolate_pos are mutually exclusive: "
                                 "baked_pos exists to avoid the runtime interpolation flag")
            # Keep the ORIGINAL trained table: every per-grid bake must resample from
            # THIS, never from whatever table was last installed into the tower
            # (re-baking from a baked table compounds the resampling -- the bug the
            # first gate-4 run caught: 196 -> 1204 -> tried to treat 1204 as g*g).
            self._pos_original = self.tower.embeddings.position_embedding.weight.data
            self._pos_original_pe = self.tower.embeddings.position_embedding
        elif (self.image_size != spec.native_size and not self._interp_pos
              and spec.kind != "qwen_vl_vis"):
            # The Qwen tower is resolution-flexible BY CONSTRUCTION: a 48x48 learned
            # table bilinearly interpolated per grid (fast_pos_embed_interpolate) plus 2D
            # RoPE. There is no single position table to resize, and adding baked_pos /
            # interpolate_pos here would be inventing a mechanism the tower does not have.
            # (_baked_pos / _interp_pos are already gated to kind == "siglip_map", so both
            # flags are inert for this kind -- the exclusion here is for the print and to
            # keep the intent explicit.)
            resize_position_embeddings(self.tower, self.image_size,
                                       patch_size=spec.patch_size)
        # DINOv3 train-mode RoPE coordinate augmentation (see class docstring).
        if spec.kind == "cls" and dinov3_rope_augment is not None:
            cfg = getattr(tower, "config", None)
            want = 2.0 if dinov3_rope_augment else None
            if cfg is not None and getattr(cfg, "pos_embed_rescale", None) != want:
                cfg.pos_embed_rescale = want
                print(f"[backbones] {spec.key}: pos_embed_rescale -> {want} "
                      f"(rope_augment={dinov3_rope_augment})")
        self._grad_ckpt = False
        if gradient_checkpointing:
            # use_reentrant=False is what lets gradients flow to the adapters and the
            # input through FROZEN weights without enable_input_require_grads.
            tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self._grad_ckpt = True
        self._freeze()

        self.register_buffer("norm_mean", torch.tensor(spec.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(spec.std).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _slice(full: nn.Module) -> tuple[nn.Module, nn.Module | None]:
        """Drop the text tower. Returns (vision tower, optional projection)."""
        if hasattr(full, "vision_model"):
            proj = getattr(full, "visual_projection", None)
            tower = full.vision_model
            # Detach the slice from the parent so the text tower is actually freed.
            del full
            return tower, proj
        # THIRD BRANCH (2026-09-26): the Qwen-style VLMs keep the tower at
        # `model.visual` (Qwen3_5ForConditionalGeneration -> Qwen3_5Model.visual ->
        # Qwen3_5VisionModel), NOT at `vision_model`. Without this branch the function
        # falls through to `return full, None` and hands back the WHOLE VLM, which is
        # then called as tower(pixel_values=x) -- a silently wrong forward that raises
        # nowhere near the cause. Checked BEFORE the fallthrough so a VLM can never take
        # the "this model IS the tower" path.
        visual = getattr(getattr(full, "model", None), "visual", None)
        if visual is not None:
            del full
            return visual, None
        # DINOv3ViTModel IS the vision tower; its encoder lives at `.model`.
        return full, None

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep the frozen tower in eval -- UNLESS gradient checkpointing is on.

        Default (no checkpointing): pinning the tower to eval is the frozen-tower
        guarantee; train mode would turn on any stochastic depth / dropout and
        change the features underneath our adapters.

        With gradient checkpointing on, HF only checkpoints modules that are in
        train() mode, so the tower MUST be allowed into train mode or the memory
        saving silently does not happen (measured: eval-mode = no checkpointing,
        1214 MiB; train-mode = checkpointed, 750 MiB). Safe because the dropout
        surface was verified all-zero -- the only stochastic element is DINOv3's
        RoPE augmentation, which is an explicit config decision (see __init__).
        """
        if self._grad_ckpt:
            return super().train(mode)
        super().train(mode)
        self.tower.eval()
        return self

    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.norm_mean.to(x01.dtype)) / self.norm_std.to(x01.dtype)

    def _baked_pos_state(self, gh: int, gw: int) -> dict:
        """Position-table state for grid (gh, gw), baked once and cached.

        Everything the SigLIP embeddings forward reads is rebuilt here: the table
        itself, position_ids (its length indexes the table), and the num_patches /
        num_positions / image_size gates. Deterministic per grid: same (gh,gw) ->
        bit-identical table every time, so eval and resume behave identically.
        """
        st = self._pos_cache.get((gh, gw))
        if st is None:
            emb = self.tower.embeddings
            # Work on the CURRENTLY INSTALLED table's device/dtype: _pos_original was
            # captured at __init__ and may predate a .to(device) move (module moves
            # replace parameter tensors; a stale reference would bake on CPU).
            dev = emb.position_embedding.weight.device
            dtp = emb.position_embedding.weight.dtype
            old = self._pos_original.to(device=dev, dtype=dtp)
            if (gh, gw) == (14, 14):
                # Identity grid: keep the ORIGINAL table. Bicubic resize to the same
                # size is not guaranteed bit-exact, and 224 inputs must stay a no-op.
                new_pe = self._pos_original_pe
                pos_ids = torch.arange(196, device=dev).unsqueeze(0)
                st = {"pe": new_pe, "pos_ids": pos_ids, "num_patches": 196}
            else:
                new_w = _resample_pos_weight(old, gh, gw)
                new_pe = nn.Embedding(new_w.shape[0], new_w.shape[1])
                with torch.no_grad():
                    new_pe.weight.copy_(new_w)
                new_pe.weight.requires_grad_(False)
                new_pe.to(device=old.device, dtype=old.dtype)
                pos_ids = torch.arange(new_w.shape[0], device=old.device).unsqueeze(0)
                st = {"pe": new_pe, "pos_ids": pos_ids, "num_patches": gh * gw}
            self._pos_cache[(gh, gw)] = st
            if (gh, gw) != (14, 14):
                print(f"[baked-pos] {self.spec.key}: baked {gh}x{gw} grid table "
                      f"({old.shape[0]} -> {st['num_patches']} rows)")
        return st

    def _install_baked_pos(self, gh: int, gw: int) -> None:
        st = self._baked_pos_state(gh, gw)
        emb = self.tower.embeddings
        emb.position_embedding = st["pe"]
        if getattr(emb, "position_ids", None) is not None:
            emb.position_ids = st["pos_ids"]
        emb.num_patches = st["num_patches"]
        if hasattr(emb, "num_positions"):
            emb.num_positions = st["num_patches"]
        if hasattr(emb, "image_size"):
            emb.image_size = (gh * 16, gw * 16)
        cfg = getattr(self.tower, "config", None)
        if cfg is not None and hasattr(cfg, "image_size"):
            cfg.image_size = gh * 16  # square-only attribute; nominal only

    def embed(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: (B,3,H,W) in [0,1], H/W multiples of 16. Returns (B, embed_dim)."""
        x = self.normalize(x01).to(self.dtype)
        if self.spec.kind == "qwen_vl_vis":
            # NOT `tower(pixel_values=x)`. This tower's forward is
            # forward(hidden_states, grid_thw) with the pixels ALREADY patchified to
            # (num_patches, C*T*16*16) in MERGED-BLOCK order (raster is wrong by 2.0 and
            # does not crash), and it packs the whole batch into ONE sequence -- so the
            # pooled output is split per image inside qwen_vision_embed.
            # Same normalize-mean/std 0.5/0.5 as SigLIP2, applied above.
            # SUB-BATCH CHUNKING (2026-10-03): the packed-sequence design gives QUADRATIC
            # attention in batch*patches -- a 120-image native-512 forward = 123k tokens
            # wedged a 24 GB card at step 1 (100% GPU, zero progress). Split into
            # QWEN_FWD_CHUNK-image sequences. NOT bit-identical to a full-batch packed
            # forward (full-attention layers see the whole sequence, so they leak across
            # images when packed): chunking REMOVES that cross-image leakage. Deterministic
            # and self-consistent -- every path (train/eval/FR) uses the same chunking.
            chunk = QWEN_FWD_CHUNK
            if x.shape[0] <= chunk:
                return qwen_vision_embed(self.tower, x, self._qwen_pool)
            return torch.cat([qwen_vision_embed(self.tower, x[i:i + chunk], self._qwen_pool)
                              for i in range(0, x.shape[0], chunk)], dim=0)
        if self._baked_pos:
            self._install_baked_pos(x.shape[-2] // 16, x.shape[-1] // 16)
            out = self.tower(pixel_values=x)
        elif self._interp_pos:
            # Dynamic non-square grids: bicubically resample the learned position
            # table to this batch's (H/16, W/16) inside the tower. Bit-identical at
            # 224 (early return), ~0.04 ms otherwise.
            out = self.tower(pixel_values=x, interpolate_pos_encoding=True)
        else:
            out = self.tower(pixel_values=x)

        if self.spec.kind == "cls":
            # DINOv3: last_hidden_state is (B, 1+registers+patches, D); CLS is index 0.
            emb = out.last_hidden_state[:, 0]
        elif self.spec.kind == "siglip_map":
            emb = out.pooler_output
            if emb is None:  # pragma: no cover - defensive
                raise RuntimeError("SigLIP vision tower returned no pooler_output (MAP head)")
        elif self.spec.kind == "metaclip_proj":
            pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
            emb = self.visual_projection(pooled) if self.visual_projection is not None else pooled
        else:  # pragma: no cover
            raise ValueError(f"unknown embed kind {self.spec.kind!r}")

        return emb

    def embed_multi(self, x01: torch.Tensor, layers) -> list[torch.Tensor]:
        """CLS embedding at several transformer depths, ONE tower pass.

        Multi-channel readout (2026-10-02): `output_hidden_states=True` gives
        hidden_states[0..L] (index 0 = embedding output, index L = final layer), and
        CLS is token 0 at every depth for DINOv3. Only kind "cls" is supported -- the
        other towers pool differently per layer and would need their own plumbing.
        """
        if self.spec.kind != "cls":
            raise RuntimeError(f"embed_multi supports kind 'cls' towers only, "
                               f"got {self.spec.kind!r}")
        x = self.normalize(x01).to(self.dtype)
        if self._baked_pos:
            self._install_baked_pos(x.shape[-2] // 16, x.shape[-1] // 16)
            out = self.tower(pixel_values=x, output_hidden_states=True)
        elif self._interp_pos:
            out = self.tower(pixel_values=x, interpolate_pos_encoding=True,
                             output_hidden_states=True)
        else:
            out = self.tower(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        depth = len(hs) - 1
        bad = [l for l in layers if not (1 <= l <= depth)]
        if bad:
            raise RuntimeError(f"readout_layers {bad} outside this tower's depth "
                               f"(1..{depth}); hidden_states has {len(hs)} entries")
        return [hs[int(l)][:, 0].float() for l in layers]

    @property
    def embed_dim(self) -> int:
        """Verify the declared dim against reality once, so a config typo cannot pass."""
        if self.spec.kind == "qwen_vl_vis":
            # spec.embed_dim declares the DEFAULT (merged, 5120); the live branch may be
            # the 1152-d pre-merge one, and PerceptualModel.embed_dim must follow reality.
            # The *_tokens modes report their BASE pool's dim (the token axis is N, not D).
            from .qwen_vision import TOKEN_POOL_MODES
            base = TOKEN_POOL_MODES.get(self._qwen_pool, self._qwen_pool)
            return int(POOL_DIMS[base])
        return self.spec.embed_dim

    @property
    def oft_prefix(self) -> str:
        """`include_prefix` to hand to inject_oft: the tower lives at `self.tower`."""
        return f"tower.{self.spec.oft_prefix}"


def build_branches(keys=DEFAULT_ENSEMBLE, dtype: torch.dtype = torch.float32,
                   device: str = "cuda", verbose: bool = True,
                   image_size: int | None = None,
                   gradient_checkpointing: bool = False,
                   dinov3_rope_augment: bool | None = None,
                   interpolate_pos: bool = False,
                   baked_pos: bool = False,
                   qwen_pool: str = "merged",
                   qwen_tower_path: str | None = None) -> nn.ModuleList:
    branches = []
    for k in keys:
        if k not in BACKBONES:
            raise KeyError(f"unknown backbone {k!r}; known: {sorted(BACKBONES)}")
        b = Branch(BACKBONES[k], dtype=dtype, image_size=image_size,
                   gradient_checkpointing=gradient_checkpointing,
                   dinov3_rope_augment=dinov3_rope_augment,
                   interpolate_pos=interpolate_pos,
                   baked_pos=baked_pos,
                   qwen_pool=qwen_pool,
                   qwen_tower_path=qwen_tower_path)
        branches.append(b)
        if verbose:
            n = sum(p.numel() for p in b.parameters())
            # dim comes off the BRANCH, not the registry: the qwen tower's dim depends on
            # the pooling choice and a registry literal would be a silent lie.
            print(f"[backbones] {k:<18s} dim={b.embed_dim:<4d} params={n/1e6:7.2f}M  "
                  f"oft_prefix={BACKBONES[k].oft_prefix!r}  image_size={b.image_size}"
                  + ("  grad-ckpt" if b._grad_ckpt else "")
                  + ("  interp-pos" if b._interp_pos else "")
                  + ("  baked-pos" if b._baked_pos else "")
                  + (f"  pool={b._qwen_pool}" if b.spec.kind == "qwen_vl_vis" else ""))
    return nn.ModuleList(branches)
