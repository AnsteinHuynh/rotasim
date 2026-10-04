"""Minimal runnable example: load the released metric, score a pair, backprop.

Run from this repository's root, after setting the backbone path in CONFIG.py
(see README > Installation):

    python example.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # the vendored package

from dreamsim_oft.as_loss import OFTDreamsimFn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = OFTDreamsimFn(device=device,
                           ckpt=str(Path(__file__).resolve().parent / "step001200.pt"))

    # Two images as (N, 3, H, W) floats in [-1, 1], ANY resolution.
    x = torch.rand(2, 3, 512, 512, device=device) * 2 - 1        # e.g. a reconstruction
    y = (x + 0.02 * torch.randn_like(x)).clamp(-1, 1)            # e.g. its target

    d = metric(x, y)
    print("distance :", [round(v, 5) for v in d.tolist()])       # (N,) cosine distances
    print("d(x, x)  :", metric(x, x).tolist())                    # exactly 0.0

    # As a loss: differentiable all the way back to the image tensors.
    xg = x.clone().requires_grad_(True)
    metric(xg, y).mean().backward()
    print("grad     : nonzero =", bool(xg.grad.abs().sum() > 0),
          " max|g| =", f"{float(xg.grad.abs().max()):.3e}")


if __name__ == "__main__":
    main()
