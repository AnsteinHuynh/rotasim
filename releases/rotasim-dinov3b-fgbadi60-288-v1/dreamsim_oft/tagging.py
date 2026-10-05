"""Tag adapter parameters so adv_optm treats them correctly.

adv_optm (Apache-2.0, by Koratahiu) dispatches on underscore attributes
carried by Parameter objects. An OFT rotation parameter is a flat
(r, n_elements) bag of packed upper-triangular coordinates, NOT a matrix, and
the optimizers that do matrix-aware work need to know that or they will
misinterpret its shape:

    adv_optm/util/scaled_optm.py    `_is_oft` x4, `_oft_scale_factor` x1
    adv_optm/util/centered_decay.py `_is_oft`
    adv_optm/util/Kourkoutas.py     `_is_oft`, `_is_lora_A`

Skipping the tagging silently disables those code paths -- training still
runs, it is just not the thing we think we are running.

This module walks `named_parameters()` and classifies each adapter tensor by
name (classification order matters: a name matching several patterns takes the
first hit), then reports a census so callers can log that tagging actually
matched something.

usage:
    from dreamsim_oft.tagging import tag_peft_parameters, describe_tags
    tag_peft_parameters(model, oft_scaled=cfg.oft_scaled)
    describe_tags(model)
"""

from __future__ import annotations

import math

import torch


def _classify(name: str, lora_a_suffix: str, lora_b_suffix: str) -> str | None:
    """Pattern-classify one parameter name. Order = priority."""
    if name.endswith("oft_R.weight"):
        return "oft"
    if name.endswith("dora_log_multiplier"):
        return "doft_magnitude"
    if name.endswith(lora_a_suffix) or ".lora_A." in name:
        return "lora_a"
    if name.endswith(lora_b_suffix) or ".lora_B." in name:
        return "lora_b"
    if "dora_scale" in name or "lora_magnitude_vector" in name:
        return "dora_scale"
    return None


def _block_size_from_n_elements(n_elements: int) -> float:
    """Invert n_elements = b(b-1)/2 for the block size b."""
    return (1 + math.sqrt(1 + 8 * n_elements)) / 2


def tag_peft_parameters(model: torch.nn.Module, oft_scaled: bool = False,
                        lora_a_suffix: str = "lora_down.weight",
                        lora_b_suffix: str = "lora_up.weight") -> dict:
    """Set `_is_oft` / `_oft_scale_factor` (and LoRA A/B markers) by parameter name.

    Returns a small census so the caller can log that tagging actually matched things.
    """
    counts = {"oft": 0, "oft_scaled": 0, "lora_a": 0, "lora_b": 0, "dora_scale": 0,
              "doft_magnitude": 0}

    for name, p in model.named_parameters():
        kind = _classify(name, lora_a_suffix, lora_b_suffix)
        if kind is None:
            continue
        if kind == "oft":
            # adv_optm keys OFT handling off this attribute.
            p._is_oft = True
            counts["oft"] += 1
            if oft_scaled:
                # Hand adv_optm the same 2*sqrt(b-1) factor the forward divides
                # out, so its update scaling and the model stay consistent.
                b = _block_size_from_n_elements(p.shape[-1])
                p._oft_scale_factor = 2 * math.sqrt(b - 1)
                counts["oft_scaled"] += 1
        elif kind == "doft_magnitude":
            # DOFT's per-output-channel log multiplier has no peft-style name,
            # so it needs its own branch or the census reports 0 magnitudes
            # while the adapter actually has one.
            p._is_dora_scale = True
            counts["doft_magnitude"] += 1
        elif kind == "lora_a":
            p._is_lora_A = True
            counts["lora_a"] += 1
        elif kind == "lora_b":
            p._is_lora_B = True
            counts["lora_b"] += 1
        else:  # "dora_scale"
            p._is_dora_scale = True
            counts["dora_scale"] += 1

    return counts


def describe_tags(model: torch.nn.Module) -> None:
    """Print which parameters carry tags, and sanity-check the OFT block geometry."""
    tagged = []
    for name, p in model.named_parameters():
        marks = [a for a in ("_is_oft", "_is_lora_A", "_is_lora_B", "_is_dora_scale")
                 if getattr(p, a, False)]
        if marks or hasattr(p, "_oft_scale_factor"):
            scale = getattr(p, "_oft_scale_factor", None)
            tagged.append((name, tuple(p.shape), marks, scale))

    print(f"[tag] {len(tagged)} tagged parameter tensor(s)")
    for name, shape, marks, scale in tagged[:6]:
        extra = f" scale={scale:.4f}" if scale is not None else ""
        print(f"        {name:<62s} {str(shape):<16s} {','.join(marks)}{extra}")
    if len(tagged) > 6:
        print(f"        ... ({len(tagged) - 6} more)")

    # Geometry check: n_elements must be a triangular number, else the block
    # size recovery above (and adv_optm's own use of it) would be wrong.
    bad = []
    for name, p in model.named_parameters():
        if getattr(p, "_is_oft", False):
            b = _block_size_from_n_elements(p.shape[-1])
            if abs(b - round(b)) > 1e-6:
                bad.append((name, p.shape[-1]))
    if bad:
        print(f"[tag] WARNING: {len(bad)} OFT tensor(s) have a non-triangular "
              f"n_elements, so block size cannot be recovered: {bad[:3]}")
