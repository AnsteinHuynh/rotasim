"""
[Begin Work Zone]
The Qwen3.8-27B VISION TOWER, loaded out of a local llama.cpp mmproj GGUF.

WHY THIS FILE EXISTS: mmproj-Qwen3.8-27B-bf16.gguf (888 MiB) contains the vision tower +
merger and NOTHING else, while the HF repo is 18 shards / 55.6 GB. AutoModel cannot read a
GGUF, and backbones._slice() cannot find this tower on the VLM anyway (it lives at
`full.model.visual`), so the image path is owned here.

VERIFIED ON DISK (scratch/sophie/, nothing in src/ was touched):
  * STRICT LOAD OK: 334 GGUF tensors -> 333 HF params, 0 unmapped / missing / extra /
    shape-mismatched; 460,730,096 params. Every mismatch RAISES -- a silently mis-mapped
    tower still produces plausible embeddings and would never crash.
  * The one llama.cpp deviation: the patch-embed Conv3d is stored as TWO temporal-frame
    tensors -> stack(dim=2). Order-invariant for static images (the processor duplicates
    the frame), so `frames bit-identical=False` in the log is EXPECTED, not a failure.
  * Patch order is MERGED-BLOCK: (block_row, block_col, intra_row, intra_col). Raster
    order is WRONG by max 2.000e+00 on [-1,1] pixels and does not crash. Measured against
    the official Qwen2-VL processor: 288 (324,1536) 5.914e-08, 544 (1156,1536) 5.914e-08,
    224 (196,1536) 5.914e-08. Same bug class as the D-line .view(B,k+1,D) scramble.
  * Positions: a 48x48 learned table (num_position_embeddings 2304) bilinearly
    interpolated per grid + 2D RoPE => resolution-flexible. NO pos surgery, NO baked_pos,
    NO interpolate_pos for this kind. native_size 768 / patch_size 16 are metadata only.
  * dtype: the GGUF is bf16. dtype=float32 gives fp32 ARITHMETIC over bf16-precision
    WEIGHTS (an exact upcast) -- that is what the sq288 class asks for.

usage:
    from dreamsim_oft.qwen_vision import load_vision_tower, qwen_vision_embed, POOL_DIMS
[End Work Zone]
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch

# Case-checked on disk 2026-09-26: the directory is literally `LLAMA-Models`.
QWEN38_MMPROJ_LOCAL = r"I:\MyApps\LLAMA-Models\qwen38-27b\mmproj-Qwen3.8-27B-bf16.gguf"

# vision_config for Qwen/Qwen3.8-27B, verbatim from the HF config.json.
VISION_CFG_27B = {
    "model_type": "qwen3_5",
    "depth": 27,
    "hidden_size": 1152,
    "num_heads": 16,
    "intermediate_size": 4304,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "out_hidden_size": 5120,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [],
}

PS, MS, TP = 16, 2, 2                 # patch, spatial merge, temporal patch
PATCH_DIM = 3 * TP * PS * PS          # 1536 = the width of one pixel_values row
# Which output the metric pools. "merged" is the merger's 5120-d output; "premerge" is the
# last_hidden_state BEFORE the merger, 1152-d. Both were measured frozen at 288 on
# 2026-09-26 (70.43% vs 70.51%, val n=2509) -- a dead heat, so this is a knob, not a bug.
POOL_DIMS = {"merged": 5120, "premerge": 1152}
# Token-map modes (2026-10-04, spatial/LPIPS-class readout): instead of mean-pooling
# the token map to one vector, return the (B, N, D) map itself so the metric can
# compare SPATIALLY ALIGNED tokens. merged_tokens = 32px blocks (5120-d),
# premerge_tokens = 16px patches (1152-d, LPIPS granularity).
TOKEN_POOL_MODES = {"merged_tokens": "merged", "premerge_tokens": "premerge"}

# llama.cpp clip-mmproj naming -> transformers Qwen3_5VisionModel naming.
_BLK = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}
_TOP = {
    "v.patch_embd": "patch_embed.proj",
    "v.position_embd": "pos_embed",
    "v.post_ln": "merger.norm",
    "mm.0": "merger.linear_fc1",
    "mm.2": "merger.linear_fc2",
}
_BLK_RE = re.compile(r"^v\.blk\.(\d+)\.(\w+)(?:\.(.+))?$")


def gguf_to_hf_name(name: str) -> str | None:
    """Map one GGUF tensor name to its HF parameter name, or None if unmappable."""
    m = _BLK_RE.match(name)
    if m:
        idx, part, rest = m.group(1), m.group(2), m.group(3)
        hf = _BLK.get(part)
        if hf is None:
            return None
        return f"blocks.{idx}.{hf}" + (f".{rest}" if rest else "")
    for gg, hf in _TOP.items():
        if name == gg or name.startswith(gg + "."):
            return hf + name[len(gg):]
    return None


def _to_torch(tensor, torch_mod):
    """GGUF tensor -> torch tensor of the same logical layout.

    GGUF stores ne0 (fastest-varying) first, i.e. the REVERSE of the torch shape. We
    flatten in C order and reshape to the reversed shape, so memory order is preserved
    exactly (no transpose is introduced). BF16 arrives as raw 16-bit words and is
    reinterpreted, never numerically converted.
    """
    import numpy as np

    raw = np.asarray(tensor.data)
    ggml_shape = tuple(int(d) for d in tensor.shape)
    torch_shape = tuple(reversed(ggml_shape))
    tt = tensor.tensor_type.name
    flat = np.ascontiguousarray(raw).reshape(-1)
    if tt == "F32":
        return torch_mod.from_numpy(flat.view(np.float32)).reshape(torch_shape)
    if tt == "F16":
        return torch_mod.from_numpy(flat.view(np.float16)).reshape(torch_shape)
    if tt == "BF16":
        if flat.dtype != np.uint16:
            flat = flat.view(np.uint16)
        return torch_mod.from_numpy(flat).view(torch_mod.bfloat16).reshape(torch_shape)
    raise RuntimeError(f"unsupported GGUF tensor type {tt} for {tensor.name}")


def _load_vision_from_hf_dir(hf_dir: str, vision_cfg: dict, dtype, verbose: bool, strict: bool):
    """Load the vision tower straight from an HF-format Qwen3.8-27B checkpoint
    directory (18 safetensors shards). The 333 vision tensors live under
    'model.visual.*' and map 1:1 onto Qwen3_5VisionModel state_dict names after
    stripping the prefix -- unlike the gguf route there is NO name surgery and
    NO temporal-split reassembly. Added 2026-10-04 for the rotasim release: users
    point the tower config at either this directory or the mmproj gguf."""
    import glob
    from safetensors.torch import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = Qwen3_5VisionConfig(**vision_cfg)
    tower = Qwen3_5VisionModel._from_config(cfg, dtype=dtype)
    want = tower.state_dict()
    sd = {}
    for shard in sorted(glob.glob(str(Path(hf_dir) / "model-*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k).to(dtype)
    missing = sorted(set(want) - set(sd))
    extra = sorted(set(sd) - set(want))
    shape_bad = sorted(k for k in set(want) & set(sd)
                       if tuple(want[k].shape) != tuple(sd[k].shape))
    if verbose:
        print(f"[hf-dir] {hf_dir}")
        print(f"[hf-dir] vision tensors: {len(sd)} (HF wants {len(want)})")
        for label, items in (("MISSING", missing), ("EXTRA", extra), ("SHAPE BAD", shape_bad)):
            if items:
                print(f"[hf-dir] !! {label} ({len(items)}): {items[:8]}")
    if strict and (missing or extra or shape_bad):
        raise RuntimeError(
            f"HF-dir vision extraction is not a clean bijection: {len(missing)} missing, "
            f"{len(extra)} extra, {len(shape_bad)} shape-mismatched")
    tower.load_state_dict(sd, strict=True)
    n = sum(p_.numel() for p_ in tower.parameters())
    if verbose:
        print(f"[hf-dir] STRICT LOAD OK: {n/1e6:.1f}M params ({n*dtype.itemsize/2**20:.0f} MiB)")
    return tower, {"n_tensors_hf": len(sd), "params": n, "source": "hf-dir"}


def load_vision_tower(vision_cfg: dict = VISION_CFG_27B, gguf_path: str = QWEN38_MMPROJ_LOCAL,
                      dtype=None, verbose: bool = True, strict: bool = True):
    """Return (tower, info). Raises on any mapping ambiguity when strict.

    Ported from scratch/sophie/qwen38_vision_loader.py (the file that printed
    STRICT LOAD OK). Kept as a copy rather than an import because scratch/ is not a
    package and must not become one.
    """
    import os as _os
    _p = _os.environ.get("QWEN_TOWER_PATH") or gguf_path
    if _os.path.isdir(_p):
        return _load_vision_from_hf_dir(_p, vision_cfg, dtype, verbose, strict)

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    import gguf

    if dtype is None:
        dtype = torch.bfloat16
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(gguf_path)

    reader = gguf.GGUFReader(gguf_path)
    cfg = Qwen3_5VisionConfig(**vision_cfg)
    tower = Qwen3_5VisionModel._from_config(cfg, dtype=dtype)
    want = tower.state_dict()

    sd: dict = {}
    unmapped_gguf: list = []
    temporal_frames: list = []
    for t in reader.tensors:
        # SPECIAL CASE, found by this very gate: llama.cpp stores the patch-embed Conv3d
        # as TWO tensors split along the temporal axis --
        #   v.patch_embd.weight   (1152,3,16,16)
        #   v.patch_embd.weight.1 (1152,3,16,16)
        # where HF wants one (1152,3,2,16,16) for temporal_patch_size=2. Reassembling it
        # is a stack along dim=2. Every OTHER name is 1:1.
        m = re.match(r"^v\.patch_embd\.weight(?:\.(\d+))?$", t.name)
        if m and m.group(1) is not None:
            temporal_frames.append((int(m.group(1)), t))
            continue
        hf = gguf_to_hf_name(t.name)
        if hf is None:
            unmapped_gguf.append(t.name)
            continue
        sd[hf] = _to_torch(t, torch).to(dtype)

    if temporal_frames:
        by_name = {t.name: t for t in reader.tensors}
        first = _to_torch(by_name["v.patch_embd.weight"], torch)
        ordered = [first] + [_to_torch(t, torch) for _, t in sorted(temporal_frames)]
        if verbose:
            same = all(bool(torch.equal(f, ordered[0])) for f in ordered[1:])
            print(f"[gguf] reassembling patch-embed Conv3d from {len(ordered)} temporal "
                  f"frame tensor(s) {tuple(ordered[0].shape)} -> stack(dim=2); "
                  f"frames bit-identical={same} (False is EXPECTED and harmless: static "
                  f"images make the two temporal taps order-invariant)")
        sd["patch_embed.proj.weight"] = torch.stack(ordered, dim=2).to(dtype)

    missing = sorted(set(want) - set(sd))
    extra = sorted(set(sd) - set(want))
    shape_bad = sorted(k for k in set(want) & set(sd) if tuple(want[k].shape) != tuple(sd[k].shape))

    if verbose:
        print(f"[gguf] {gguf_path}")
        print(f"[gguf] {len(reader.tensors)} tensors -> {len(sd)} mapped "
              f"({len(unmapped_gguf)} unmappable), HF wants {len(want)}")
        for label, items in (("UNMAPPED GGUF", unmapped_gguf), ("MISSING in GGUF", missing),
                             ("EXTRA vs HF", extra), ("SHAPE MISMATCH", shape_bad)):
            if items:
                print(f"[gguf] !! {label} ({len(items)}): {items[:8]}")

    if strict and (unmapped_gguf or missing or extra or shape_bad):
        raise RuntimeError(
            f"mmproj->HF mapping is not a clean bijection: {len(unmapped_gguf)} unmapped, "
            f"{len(missing)} missing, {len(extra)} extra, {len(shape_bad)} shape-mismatched")

    tower.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in tower.parameters())
    info = {
        "n_tensors_gguf": len(reader.tensors),
        "n_mapped": len(sd),
        "params": n,
        "gguf_mib": os.path.getsize(gguf_path) / 2**20,
        "cfg": vision_cfg,
    }
    if verbose:
        print(f"[gguf] STRICT LOAD OK: {n/1e6:.1f}M params "
              f"({n*dtype.itemsize/2**20:.0f} MiB in {dtype})")
    return tower, info


def patchify_block(x_hwc: torch.Tensor) -> torch.Tensor:
    """(H,W,C) -> (num_patches, C*TP*PS*PS), MERGED-BLOCK order. DO NOT "improve" IT.

    VERBATIM from scratch/sophie/qwen38_frozen/frozen_probe_qwen38_288.py:54-61 -- the
    function gated bit-exact against the official processor. The reshape / permute /
    unsqueeze / expand sequence IS the proof; raster order is wrong by 2.0.
    """
    h, w, c = x_hwc.shape
    hb, wb = h // (MS * PS), w // (MS * PS)
    x = x_hwc.reshape(hb, MS, PS, wb, MS, PS, c).permute(0, 3, 1, 4, 6, 2, 5)
    x = x.unsqueeze(5).expand(hb, wb, MS, MS, c, TP, PS, PS)
    return x.reshape(hb * wb * MS * MS, c * TP * PS * PS)


_RESIZED_WARN = [0]


def patchify_image(x_chw: torch.Tensor) -> torch.Tensor:
    """(C,H,W) -> (num_patches, C*TP*PS*PS). Owns the %32 policy.

    NON-%32 SIDES ARE RESAMPLED UP TO THE NEXT MULTIPLE OF 32 (2026-10-04, for the
    FR benchmark's native-size cells, e.g. 488x610): that is exactly what the official
    Qwen processor does (round the side up, then patchify), so this is the faithful
    behaviour, not a compromise. The verbatim patchify body TRUNCATES (hb = h // 32)
    on such a side, silently dropping a strip -- that silent-wrong class stays
    impossible: we resample explicitly and log the first occurrence.
    """
    h, w = int(x_chw.shape[-2]), int(x_chw.shape[-1])
    step = MS * PS
    if h % step or w % step:
        H, W = (h + step - 1) // step * step, (w + step - 1) // step * step
        x_chw = torch.nn.functional.interpolate(
            x_chw[None], size=(H, W), mode="bicubic", align_corners=False,
            antialias=True)[0]
        _RESIZED_WARN[0] += 1
        if _RESIZED_WARN[0] == 1:
            print(f"[qwen-patchify] non-{step}-multiple input {h}x{w}: resampled up to "
                  f"{H}x{W} (official-processor behaviour; further occurrences silent)",
                  flush=True)
    return patchify_block(x_chw.permute(1, 2, 0).contiguous())


def qwen_vision_embed(tower, x01: torch.Tensor, pool: str = "merged") -> torch.Tensor:
    """(B,3,H,W) in the tower's OWN normalisation -> (B, POOL_DIMS[pool]) for the pooled
    modes, or (B, N, POOL_DIMS[TOKEN_POOL_MODES[pool]]) for the *_tokens modes.

    This tower's forward is forward(hidden_states, grid_thw): pixels must ALREADY be
    patchified to (num_patches, C*T*16*16), and grid_thw is (num_images, 3) in PATCH
    units (T, H/16, W/16). The tower packs the whole batch into ONE sequence, so the
    pooled output must be split PER IMAGE by its merged/pre-merge token count -- pooling
    the concatenated sequence would mix images from different triplets. The *_tokens
    modes keep the per-image token map (B, N, D) instead of mean-pooling it, for
    spatially-aligned (LPIPS-class) distances; N must MATCH within a pair, which the
    caller must guarantee (same corpus geometry).
    """
    token_mode = pool in TOKEN_POOL_MODES
    base_pool = TOKEN_POOL_MODES[pool] if token_mode else pool
    if base_pool not in POOL_DIMS:
        raise ValueError(f"unknown qwen pool {pool!r}; expected one of "
                         f"{sorted(POOL_DIMS)} + {sorted(TOKEN_POOL_MODES)}")

    pvs, rows, counts = [], [], []
    for b in range(x01.shape[0]):
        xi = x01[b]
        hi, wi = int(xi.shape[-2]), int(xi.shape[-1])
        # ROUND UP TO %32 BEFORE anything else (official-processor behaviour) so the
        # PATCHES and the grid_thw row below are built from the SAME size -- resampling
        # only inside patchify_image would leave grid_thw sized for the ORIGINAL image
        # and the tower would die on hidden_states + pos_embeds (caught on the FR
        # benchmark's native cells 2026-10-04: 1280 vs 1140 patches).
        step = MS * PS
        if hi % step or wi % step:
            Hi = (hi + step - 1) // step * step
            Wi = (wi + step - 1) // step * step
            xi = torch.nn.functional.interpolate(
                xi[None], size=(Hi, Wi), mode="bicubic", align_corners=False,
                antialias=True)[0]
            hi, wi = Hi, Wi
        pvs.append(patchify_image(xi))
        rows.append((1, hi // PS, wi // PS))
        counts.append((hi // (MS * PS)) * (wi // (MS * PS)) if base_pool == "merged"
                      else (hi // PS) * (wi // PS))

    pv = torch.cat(pvs, dim=0)
    grid = torch.tensor(rows, dtype=torch.long, device=x01.device)
    # POSITIONAL hidden_states: exactly the call form the verified probes use
    # (`tower(pv, grid_thw=...)` in frozen_probe_qwen38_288.py:157). The tower's forward
    # is wrapped in @merge_with_config_defaults/@capture_outputs, so do not "tidy" this
    # into an all-keyword call without re-running the smoke.
    out = tower(pv, grid_thw=grid)
    src = out.pooler_output if base_pool == "merged" else out.last_hidden_state
    if src is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Qwen vision tower returned no {pool} output")
    if sum(counts) != src.shape[0]:
        raise RuntimeError(
            f"qwen pooling split {counts} does not sum to the tower's {src.shape[0]} rows "
            f"-- the grid or the pool mapping is wrong, and pooling across images would "
            f"silently mix them.")
    if token_mode:
        if len(set(counts)) != 1:
            raise ValueError(
                f"qwen token-map mode {pool!r} got mixed token counts {counts}: the "
                f"spatial (LPIPS-class) distance needs positionally-matching maps, so "
                f"every image in the call must share one geometry.")
        return torch.stack([t.float() for t in torch.split(src, counts)])
    return torch.stack([t.float().mean(dim=0) for t in torch.split(src, counts)])
