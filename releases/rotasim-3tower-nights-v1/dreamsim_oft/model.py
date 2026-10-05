"""
[Begin Work Zone]
The perceptual metric: an ensemble of frozen vision towers + OFTv2 rotations,
concatenated into ONE embedding whose cosine distance is the perceptual distance.

Why an embedding and not just a distance: this is the whole practical advantage over
LPIPS. `embed()` lets you precompute a database and do O(1) nearest-neighbour
retrieval; LPIPS needs both images every time, at patch resolution.

Distance convention (matches DreamSim exactly):
    D(a, b) = 1 - cos(embed(a), embed(b))        higher = more different

Embedding normalization follows DreamSim's normalize_embedding: subtract the
per-sample mean ACROSS FEATURES, then L2-normalize. That mean-centering is not
cosmetic -- it removes the component along the all-ones direction, which partially
cancels the per-branch scale differences you get when concatenating three towers with
different feature statistics. It is exposed as a flag so it can be ablated.

Trainable surface: ONLY the `oft_R.weight` tensors. Everything else -- every tower,
every projection -- is frozen and held in eval mode. `assert_trainable_surface()`
enforces this, because the failure mode (silently training nothing, or silently
training a whole tower) is invisible in the loss curve.

usage:
    from dreamsim_oft.model import PerceptualModel
    m = PerceptualModel(device="cuda")
    d = m(img_a, img_b)          # (B,) distances
    e = m.embed(img_a)           # (B, 2048)
[End Work Zone]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BACKBONES, DEFAULT_ENSEMBLE, build_branches
from .oft import inject_oft
from .tagging import tag_peft_parameters

# Which parameter-name markers identify a TRAINABLE adapter tensor, per adapter family.
# assert_trainable_surface uses this to catch "silently training nothing" or "silently
# training a whole frozen tower" -- so it has to know about every adapter we support.
ADAPTER_MARKERS: dict[str, tuple[str, ...]] = {
    "oft": ("oft_R.weight",),
    # peft names DoRA/LoRA tensors `...lora_A.default.weight` etc. The magnitude vector
    # is `lora_magnitude_vector.default.weight` in peft 2.x -- NOT `dora_scale`, which is
    # the transformers/PEFT-internal alias. Omitting it here would make
    # adapter_state_dict() save an adapter with NO magnitude vectors, i.e. an incomplete
    # file that loads cleanly and quietly behaves like plain LoRA. Keep both names.
    "dora": ("lora_A", "lora_B", "lora_magnitude_vector", "dora_scale"),
    # Plain LoRA (2026-09-26): same peft tensor names minus the magnitude vector.
    "lora": ("lora_A", "lora_B"),
    # DoRA+OFT (2026-10-01, PR #1335): the rotation tensors AND the per-output-channel
    # magnitude. BOTH must be listed: _is_adapter_param keys on this one family, so a
    # missing magnitude here would fail assert_trainable_surface and -- far worse --
    # adapter_state_dict() would silently save a rotation-only artifact that loads
    # cleanly and behaves like plain OFT.
    "doft": ("oft_R.weight", "dora_log_multiplier"),
}


def model_kwargs_from_config(stored: dict) -> dict:
    """Build PerceptualModel kwargs from a training config dict.

    SINGLE SOURCE OF TRUTH, shared by train.py and any eval script. This exists because
    hand-listing the "forward-affecting" keys is how the test eval came to rebuild a
    NEUMANN-series model on top of weights trained with the EXACT Cayley solve and score
    68% instead of ~96%. Every key below changes the forward MATH while leaving parameter
    names and shapes identical, so load_adapter_state_dict(strict=True) cannot detect a
    mismatch -- it compares key sets, not semantics.

    The specific culprit was `use_cayley_neumann`: absent from the old hand-written list,
    so it silently defaulted to True (Neumann) over exact-Cayley weights. Per the OFT
    audit those two branches produce DIFFERENT rotations (the exact branch yields R^T of
    the Neumann one), so the adapter was effectively scrambled.

    If you add a forward-affecting knob to PerceptualModel, add it here AND make sure the
    training config has a matching field.
    """
    if not stored:
        return {}
    import torch as _torch

    kw: dict = {}
    if stored.get("backbones"):
        kw["keys"] = tuple(stored["backbones"])
    if stored.get("block_size") is not None:
        kw["block_size"] = int(stored["block_size"])
    if stored.get("block_share") is not None:
        kw["block_share"] = bool(stored["block_share"])
    if stored.get("oft_scaled") is not None:
        kw["oft_scaled"] = bool(stored["oft_scaled"])
    if stored.get("oft_dropout") is not None:
        kw["dropout_probability"] = float(stored["oft_dropout"])
    if stored.get("oft_weight_dtype"):
        kw["oft_weight_dtype"] = getattr(_torch, str(stored["oft_weight_dtype"]))
    # Adapter surface (2026-09-26, OFT-everywhere arm). FORWARD-AFFECTING: it decides
    # WHICH Linears get wrapped. Parameter names for the tensors both surfaces share are
    # identical, so strict loading cannot catch a mismatch -- a 72-tensor checkpoint
    # rebuilt on the default q,k,v surface would either drop half its rotations (lenient
    # load) or refuse (strict). Checkpoints that predate the key get the keeper surface.
    if stored.get("oft_targets"):
        kw["oft_targets"] = str(stored["oft_targets"])
    # Qwen3.8 vision-tower pooling: "merged" (5120-d merger output) or "premerge"
    # (1152-d last_hidden_state). FORWARD-AFFECTING in the same way image_size is -- it
    # changes the embedding the cosine compares -- while every adapter parameter name and
    # shape stays identical, so load_adapter_state_dict(strict=True) CANNOT detect it.
    # Inert for every other backbone kind. Exactly the trap class this function exists for.
    if stored.get("qwen_pool"):
        kw["qwen_pool"] = str(stored["qwen_pool"])
    if stored.get("distance_mode"):
        kw["distance_mode"] = str(stored["distance_mode"])
    if stored.get("qwen_tower_path"):
        kw["qwen_tower_path"] = str(stored["qwen_tower_path"])
    # ---- the one that bit us ----
    if stored.get("use_cayley_neumann") is not None:
        kw["use_cayley_neumann"] = bool(stored["use_cayley_neumann"])
    if stored.get("normalize_embeds") is not None:
        kw["normalize_embeds"] = bool(stored["normalize_embeds"])
    if stored.get("per_branch_norm") is not None:
        kw["per_branch_norm"] = bool(stored["per_branch_norm"])
    # Input resolution. FORWARD-AFFECTING in the strongest sense: it decides the patch grid,
    # and SigLIP2/MetaCLIP2 carry position tables sized for exactly one grid. A model rebuilt
    # at the wrong size would load adapter weights happily and compute garbage.
    if stored.get("image_size") is not None:
        kw["image_size"] = int(stored["image_size"])
    if stored.get("adapter_type"):
        kw["adapter_type"] = str(stored["adapter_type"])
    if stored.get("dora_r") is not None:
        kw["dora_r"] = int(stored["dora_r"])
    if stored.get("dora_alpha") is not None:
        kw["dora_alpha"] = float(stored["dora_alpha"])
    if stored.get("dora_dropout") is not None:
        kw["dora_dropout"] = float(stored["dora_dropout"])
    if stored.get("dora_targets"):
        kw["dora_targets"] = tuple(t.strip() for t in str(stored["dora_targets"]).split(",") if t.strip())
    # Multi-channel readout (2026-10-02): WHICH transformer layer outputs feed the
    # distance, e.g. [4, 8, 12] = one cosine per readout depth. FORWARD-AFFECTING in the
    # strongest sense (it changes the embedding(s) the cosines compare) while adapter
    # parameter names/shapes stay identical, so strict loading cannot catch a mismatch --
    # exactly the trap class this function exists for. Carried as a tuple.
    if stored.get("readout_layers"):
        kw["readout_layers"] = tuple(int(l) for l in stored["readout_layers"])
    # ---- 544-class knobs ----
    # gradient_checkpointing changes only memory/compute, but dinov3_rope_augment
    # changes the STOCHASTIC forward during training, and interpolate_pos changes
    # how SigLIP handles any non-224 grid -- both are recorded so a rebuilt model
    # matches the one that was trained.
    if stored.get("gradient_checkpointing") is not None:
        kw["gradient_checkpointing"] = bool(stored["gradient_checkpointing"])
    if stored.get("dinov3_rope_augment") is not None:
        kw["dinov3_rope_augment"] = bool(stored["dinov3_rope_augment"])
    if stored.get("interpolate_pos") is not None:
        kw["interpolate_pos"] = bool(stored["interpolate_pos"])
    # baked_pos changes the FORWARD MATH (which position table serves each grid) while
    # leaving parameter names identical -- exactly the trap class above. A model trained
    # with baked tables and rebuilt with the flag off would load cleanly and compute
    # something else. Registered here for that reason.
    if stored.get("baked_pos") is not None:
        kw["baked_pos"] = bool(stored["baked_pos"])
    return kw


class PerceptualModel(nn.Module):
    def __init__(self, keys=DEFAULT_ENSEMBLE, block_size: int = 32, block_share: bool = False,
                 oft_scaled: bool = False, dropout_probability: float = 0.0,
                 oft_targets: str = "q_proj,k_proj,v_proj",
                 oft_weight_dtype: torch.dtype = torch.float32,
                 use_cayley_neumann: bool = True, num_cayley_neumann_terms: int = 5,
                 adapter_type: str = "oft",
                 dora_r: int = 16, dora_alpha: float = 8.0, dora_dropout: float = 0.3,
                 dora_targets=("q_proj", "k_proj", "v_proj"),
                 normalize_embeds: bool = True, per_branch_norm: bool = False,
                 image_size: int = 224,
                 gradient_checkpointing: bool = False,
                 dinov3_rope_augment: bool | None = None,
                 interpolate_pos: bool = False,
                 baked_pos: bool = False,
                 qwen_pool: str = "merged",
                 qwen_tower_path: str | None = None,
                 distance_mode: str = "pooled",
                 readout_layers: tuple | list | None = None,
                 dtype: torch.dtype = torch.float32,
                 device: str = "cuda", verbose: bool = True):
        super().__init__()
        self.keys = tuple(keys)
        self.normalize_embeds = normalize_embeds
        self.per_branch_norm = per_branch_norm
        self.image_size = image_size
        self.gradient_checkpointing = gradient_checkpointing
        self.dinov3_rope_augment = dinov3_rope_augment
        self.interpolate_pos = interpolate_pos
        self.baked_pos = baked_pos
        # Recorded on the MODULE (train.py prints the forward knobs back off `model`, so
        # a constructor/config mismatch cannot hide).
        self.qwen_pool = str(qwen_pool)
        # SPATIAL READOUT (2026-10-04): "pooled" = cosine between mean-pooled
        # embeddings (historical); "token_mean" = mean over the token map of
        # per-token cosines (LPIPS/DISTS-class spatial matching). Requires every
        # qwen branch to use a *_tokens qwen_pool and the pair to share geometry.
        if distance_mode not in ("pooled", "token_mean"):
            raise ValueError(f"unknown distance_mode {distance_mode!r}")
        self.distance_mode = distance_mode
        self.block_size = block_size
        self.block_share = block_share
        self.oft_scaled = oft_scaled
        self.oft_weight_dtype = oft_weight_dtype
        self.use_cayley_neumann = use_cayley_neumann
        self.adapter_type = adapter_type
        if adapter_type not in ADAPTER_MARKERS:
            raise ValueError(f"unknown adapter_type {adapter_type!r}; "
                             f"expected one of {sorted(ADAPTER_MARKERS)}")

        self.branches = build_branches(self.keys, dtype=dtype, device=device, verbose=verbose,
                                       image_size=image_size,
                                       gradient_checkpointing=gradient_checkpointing,
                                       dinov3_rope_augment=dinov3_rope_augment,
                                       interpolate_pos=interpolate_pos,
                                       baked_pos=baked_pos,
                                       qwen_pool=qwen_pool,
                                       qwen_tower_path=qwen_tower_path)

        self.oft_stats = []
        if adapter_type in ("oft", "doft"):
            # PREMISE GUARD (2026-10-01, audit-measured). DOFT's whole justification is that
            # an ORTHOGONAL rotation preserves the frozen weight's row norms, which is what
            # lets the DoRA normalisation be dropped. That holds for the EXACT Cayley solve
            # (measured row-norm relative change 5.5e-08) but NOT for the truncated Neumann
            # series (1.03e-02, i.e. 5 orders of magnitude worse). A config flag would
            # otherwise destroy the premise silently.
            if adapter_type == "doft" and use_cayley_neumann:
                raise ValueError(
                    "adapter_type='doft' requires use_cayley_neumann=False: the Neumann "
                    "series is only approximately orthogonal (measured row-norm drift "
                    "1.03e-02 vs 5.5e-08 for the exact Cayley solve), and DOFT's design "
                    "depends on that invariance. Set use_cayley_neumann=false.")
            # Adapter SURFACE: which Linear-name suffixes get wrapped. "q_proj,k_proj,v_proj"
            # is the keeper's reduced surface; "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj"
            # is the faithful OFT-paper surface (attention + MLP). FORWARD-AFFECTING ->
            # model_kwargs_from_config must carry it, or eval rebuilds a different model.
            oft_suffixes = tuple(t.strip() for t in str(oft_targets).split(",") if t.strip())
            self.oft_targets = oft_suffixes
            for br in self.branches:
                st = inject_oft(
                    br, block_size=block_size, block_share=block_share, oft_scaled=oft_scaled,
                    suffixes=oft_suffixes,
                    dropout_probability=dropout_probability,
                    use_cayley_neumann=use_cayley_neumann,
                    num_cayley_neumann_terms=num_cayley_neumann_terms,
                    include_prefix=br.oft_prefix, verbose=verbose,
                    dora_oft=(adapter_type == "doft"),
                )
                self.oft_stats.append(st)
            self._enforce_oft_weight_dtype(oft_weight_dtype)
        else:
            # adapter_type "dora" (use_dora) or "lora" (plain low-rank, no magnitude).
            self._inject_dora(dora_r, dora_alpha, dora_dropout, dora_targets, verbose,
                              use_dora=(adapter_type == "dora"))

        self.tag_counts = tag_peft_parameters(self, oft_scaled=oft_scaled)
        if verbose:
            print(f"[model] OFT tags set: {self.tag_counts}")

        # ---- MULTI-CHANNEL READOUT (2026-10-02) ---------------------------------
        # One cosine PER READOUT DEPTH instead of one cosine over the final CLS: the
        # single scalar head is WHY "d = 0.01" is ambiguous -- noise, blur and resample
        # all project onto the same 1-D axis, and BT training only constrains their
        # ORDER, never their decomposition. Channels are the LPIPS trick (per-layer
        # distances, learned combination) applied to CLS readouts. The TOTAL stays a
        # scalar (so loss/eval interfaces are unchanged); distance_channels() exposes
        # the per-channel vector.
        self.readout_layers = tuple(int(l) for l in readout_layers) if readout_layers else ()
        if self.readout_layers:
            if len(self.branches) != 1:
                raise ValueError("readout_layers supports a SINGLE branch (one tower); "
                                 f"got {len(self.branches)}")
            if self.branches[0].spec.kind != "cls":
                raise ValueError("readout_layers supports CLS-readout towers (DINOv3) "
                                 f"only, not kind {self.branches[0].spec.kind!r}")
            if len(set(self.readout_layers)) != len(self.readout_layers):
                raise ValueError(f"duplicate readout_layers {self.readout_layers}")
            if any(l < 1 for l in self.readout_layers):
                raise ValueError("readout_layers are 1-indexed transformer layers "
                                 "(hidden_states[L]); 0 is the embedding output")
            # Convex channel weights: logits -> softmax. Zero-init = equal vote, and
            # softmax keeps every channel's contribution NON-NEGATIVE, so the total
            # stays a convex combination of per-channel cosines (ordering-compatible
            # with the single-channel metric when one weight dominates).
            self.channel_weight_logits = nn.Parameter(
                torch.zeros(len(self.readout_layers), dtype=torch.float32))
            if verbose:
                print(f"[model] MULTI-CHANNEL readout at layers {self.readout_layers} "
                      f"({len(self.readout_layers)} channels, learned convex weights)")

        self.assert_trainable_surface(verbose=verbose)

    def _inject_dora(self, r: int, alpha: float, dropout: float, targets, verbose: bool,
                     use_dora: bool = True) -> None:
        """Attach DoRA -- or plain LoRA when use_dora=False (adapter_type "lora",
        2026-09-26: lora-vs-oft isolates low-rank vs orthogonal; dora-vs-oft isolates
        the magnitude capability OFT structurally lacks).

        DoRA splits a pretrained weight into MAGNITUDE and DIRECTION, applies the
        low-rank update to the direction, and learns the magnitude explicitly:
            W' = m * (W + BA) / ||W + BA||_c
        `m` is the per-output-channel magnitude vector (peft stores it as `dora_scale`).

        Why this is the interesting control against OFT: an orthogonal rotation is
        norm-preserving, so it CANNOT rescale a direction or suppress one -- it can only
        reorient. DoRA's learned magnitude is exactly the capability OFT structurally
        lacks, so this arm tests whether that missing capability is what costs us.
        """
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                         target_modules=list(targets), use_dora=use_dora)
        self.branches = nn.ModuleList([get_peft_model(br, cfg) for br in self.branches])
        # Capture the magnitude vectors at init so their DRIFT is measurable. DoRA
        # initialises them to the per-output-channel weight NORM, not to 1.0, so
        # comparing against 1.0 (the obvious-looking choice) would be meaningless.
        # Plain LoRA has no magnitude vectors.
        if use_dora:
            self._dora_mag_init = {
                n: p.detach().clone()
                for n, p in self.named_parameters()
                if "lora_magnitude_vector" in n or "dora_scale" in n
            }
        else:
            self._dora_mag_init = {}
        if verbose:
            n = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            kind = "DoRA" if use_dora else "LoRA"
            print(f"[{kind.lower()}] applied {kind} r={r} alpha={alpha} dropout={dropout} "
                  f"targets={list(targets)}")
            print(f"[{kind.lower()}] trainable {kind} params: {n:,} / {total:,} "
                  f"({100 * n / total:.4f}%)")
            if use_dora:
                print(f"[dora] captured {len(self._dora_mag_init)} magnitude vector(s) at init")

    def _is_adapter_param(self, name: str) -> bool:
        # The multi-channel readout's convex weights travel WITH the adapter: they are
        # trainable model parameters (not tower weights), so they must pass the frozen-
        # tower guard AND be included in adapter_state_dict() or a checkpoint would
        # silently lose the learned channel combination.
        if name.endswith("channel_weight_logits"):
            return True
        return any(m in name for m in ADAPTER_MARKERS[self.adapter_type])

    def _enforce_oft_weight_dtype(self, want: torch.dtype) -> None:
        """Pin the OFT rotation weights to `want` (fp32 by preference).

        These are created with whatever the ambient default dtype happens to be, so
        without this the adapter precision is an accident rather than a decision --
        and a bf16 rotation would make the Cayley/Neumann series visibly less exact.
        """
        cast = 0
        for name, p in self.named_parameters():
            # The DOFT magnitude is part of the adapter's forward math (exp of it multiplies
            # the output), so it is pinned to the same precision as the rotation rather than
            # being left at whatever the ambient default dtype happened to be.
            if (name.endswith("oft_R.weight") or name.endswith("dora_log_multiplier")) \
                    and p.dtype != want:
                p.data = p.data.to(want)
                cast += 1
        got = {p.dtype for n, p in self.named_parameters()
               if n.endswith("oft_R.weight") or n.endswith("dora_log_multiplier")}
        if got != {want}:
            raise RuntimeError(f"OFT weights are {got}, expected exactly {{{want}}}")
        if cast:
            print(f"[model] cast {cast} OFT weight tensor(s) to {want}")

    # -- geometry -------------------------------------------------------------
    @property
    def embed_dim(self) -> int:
        # Ask the BRANCHES, not the registry: the qwen tower's dim depends on qwen_pool.
        return sum(br.embed_dim for br in self.branches)

    def assert_trainable_surface(self, verbose: bool = True) -> None:
        bad = [(n, tuple(p.shape)) for n, p in self.named_parameters()
               if p.requires_grad and not self._is_adapter_param(n)]
        if bad:
            raise RuntimeError(
                f"{len(bad)} parameter tensor(s) outside the {self.adapter_type!r} adapter "
                f"surface require grad; the towers must stay frozen. First few: {bad[:5]}")
        if not any(p.requires_grad for p in self.parameters()):
            raise RuntimeError(
                f"No trainable parameters at all -- {self.adapter_type!r} injection did not take.")
        if verbose:
            n = self.num_trainable
            total = sum(p.numel() for p in self.parameters())
            print(f"[model] trainable {n:,} / {total:,} params ({100 * n / total:.4f}%)")

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # -- forward --------------------------------------------------------------
    @property
    def channel_weights(self) -> torch.Tensor:
        """Softmax of the logits: a convex combination over readout channels."""
        if not self.readout_layers:
            raise RuntimeError("channel_weights requires readout_layers")
        return torch.softmax(self.channel_weight_logits, dim=0)

    def embed_channels(self, x01: torch.Tensor) -> list[torch.Tensor]:
        """(B,3,H,W) in [0,1] -> list of (B, embed_dim) per readout layer.

        Each channel embedding gets the SAME normalisation the single-readout path
        applies (mean-centre + L2 when normalize_embeds), so a per-channel cosine is
        directly comparable to the released metric's cosine.
        """
        if not self.readout_layers:
            raise RuntimeError("embed_channels requires readout_layers")
        feats = self.branches[0].embed_multi(x01, self.readout_layers)
        if self.normalize_embeds:
            feats = [f - f.mean(dim=-1, keepdim=True) for f in feats]
            feats = [f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8) for f in feats]
        return feats

    def embed_spatial(self, x01: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, N, D) per-token map for distance_mode="token_mean".

        Same normalisation contract as embed(), applied PER TOKEN (mean-centre + L2
        over D), branches concatenated over D. A branch whose pool is not a
        *_tokens mode returns (B, D) and the shape mismatch fails loudly here
        rather than producing a silently-wrong broadcast downstream."""
        feats = [br.embed(x01) for br in self.branches]
        for f in feats:
            if f.ndim != 3:
                raise RuntimeError(
                    "embed_spatial requires every branch in a *_tokens qwen_pool "
                    f"(got a pooled (B, D) = {tuple(f.shape)}) -- distance_mode="
                    "token_mean with pooled branches would silently broadcast.")
        if self.per_branch_norm:
            feats = [torch.nn.functional.normalize(f, dim=-1) for f in feats]
        emb = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        if self.normalize_embeds:
            emb = emb - emb.mean(dim=-1, keepdim=True)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb

    def _token_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean over the token map of per-token cos-distance. Clamp BEFORE subtracting
        (the negative-d-at-identical-pairs defect class, model.py's own history)."""
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cos).mean(dim=-1)

    def embed(self, x01: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) in [0,1] -> (B, embed_dim)."""
        feats = [br.embed(x01) for br in self.branches]
        if self.per_branch_norm:
            # L2-normalise EACH branch BEFORE concatenating. Without this, a single cosine
            # over the concat weights the branches by their share of the squared norm --
            # measured, MetaCLIP2 held 71.1% of it while being the WEAKEST branch, so the
            # distance was mostly MetaCLIP2 with the good branches drowned out.
            # Costs no parameters and needs no statistics, unlike whitening.
            feats = [torch.nn.functional.normalize(f, dim=-1) for f in feats]
        emb = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        if self.normalize_embeds:
            emb = emb - emb.mean(dim=-1, keepdim=True)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb

    def distance_channels(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """Per-channel perceptual distances. Returns (channels, total, weights):
        channels (B, C) per-readout cosines, total (B,) = convex combination."""
        if not self.readout_layers:
            raise RuntimeError("distance_channels requires readout_layers")
        fa = self.embed_channels(img_a)
        fb = self.embed_channels(img_b)
        w = self.channel_weights
        # CLAMP BEFORE SUBTRACTING, same defect class the scalar path already fixed (as_loss.py
        # and the 288 release): F.cosine_similarity does not guarantee |cos| <= 1 in floating
        # point, so an IDENTICAL pair could come back as cos = 1 + eps and the distance went
        # NEGATIVE (measured -7.95e-08 on an mc ckpt before this clamp). Harmless for an
        # argmax decision, wrong for a loss: it breaks the "exactly 0.0 at an identical pair"
        # contract and injects a negative gradient term at the minimum.
        d = torch.stack([(1 - F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0))
                         for a, b in zip(fa, fb)], dim=-1)
        return d, d @ w, w

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        if self.readout_layers:
            # The TOTAL is what the loss and every eval see: a learned convex
            # combination of the per-channel cosines (scalar, same interface).
            return self.distance_channels(img_a, img_b)[1]
        if self.distance_mode == "token_mean":
            ea, eb = self.embed_spatial(img_a), self.embed_spatial(img_b)
            return self._token_distance(ea, eb)
        ea, eb = self.embed(img_a), self.embed(img_b)
        return 1 - F.cosine_similarity(ea, eb, dim=-1)

    def forward_pair(self, refs: torch.Tensor, lefts: torch.Tensor,
                     rights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(d(ref,left), d(ref,right)) with the REFS EMBEDDED ONCE. Bit-identical to
        two forward() calls (towers are deterministic in eval), but the reference
        images stop being a third of all tower work -- the trainer calls this
        twice per group through chunked Qwen batches."""
        if self.readout_layers:
            fr, fl, frr = (self.embed_channels(t) for t in (refs, lefts, rights))
            w = self.channel_weights
            dd = torch.stack([(1 - F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0))
                              for a, b in zip(fr, fl)], dim=-1)
            de = torch.stack([(1 - F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0))
                              for a, b in zip(fr, frr)], dim=-1)
            return dd @ w, de @ w
        if self.distance_mode == "token_mean":
            er = self.embed_spatial(refs)
            el = self.embed_spatial(lefts)
            er2 = self.embed_spatial(rights)
            return self._token_distance(er, el), self._token_distance(er, er2)
        er = self.embed(refs)
        el = self.embed(lefts)
        er2 = self.embed(rights)
        d0 = 1 - F.cosine_similarity(er, el, dim=-1)
        d1 = 1 - F.cosine_similarity(er, er2, dim=-1)
        return d0, d1

    def branch_embeddings(self, x01: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-branch embeddings, for diagnosing which branch drives a decision."""
        return {br.spec.key: br.embed(x01) for br in self.branches}

    def train(self, mode: bool = True):  # type: ignore[override]
        """Train mode for adapters only; Branch.train() pins the towers to eval."""
        super().train(mode)
        for br in self.branches:
            br.eval() if not mode else br.train(mode)
        return self

    # -- persistence ----------------------------------------------------------
    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Just the trained adapter tensors -- ~5 MB instead of ~1 GB, and all that matters."""
        return {k: v for k, v in self.state_dict().items() if self._is_adapter_param(k)}

    def load_adapter_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> None:
        own = self.adapter_state_dict()
        if strict:
            missing = set(own) - set(sd)
            extra = set(sd) - set(own)
            if missing or extra:
                raise RuntimeError(
                    f"adapter mismatch: {len(missing)} missing, {len(extra)} unexpected. "
                    f"Check that the backbone set and block_size match the checkpoint.")
        self.load_state_dict(sd, strict=False)
