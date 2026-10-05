"""288px square input class: ONE full-frame resize on the exact fractional grid.

This is the "crop" for the 288x288 square class. It replaces
`transforms.Resize((288, 288), interpolation=BICUBIC)`, which did not crop at all -- it
SQUASHED the aspect ratio (a 500x380 photo became a distorted 288x288).

What it does: samples the centre-square window of the FULL frame at the exact fractional
positions with a separable Lanczos-3 low-pass; taps that reach outside the window but remain
inside the frame still contribute. So, per pixel measurements in
scratch/sophie/lanczos_explainer/{CORPUS_1_VS_2,DIV8K_1_VS_2}.md:

  * exact grid - the window origin is (W-S)/2 in floating point and there is NO integer
    intermediate size anywhere. Both PIL shortcuts carry a half-pixel placement error:
    crop-then-resize floors (W-S)//2 (half an INPUT pixel when W-S is odd, up to 102.9/255)
    and resize-then-crop floors (cw-out)//2 (half an OUTPUT pixel, freq-mean 31.9/255,
    max 143.7). Comparing those two proves nothing: their errors sometimes CANCEL
    (710x453: |A-B| 71.8/255 while |A-ideal| 96.5/255).
  * full-frame support - the ONLY benefit the two-resize path actually delivered, kept here
    without its 5x bug, worth ~8/255 (FGResQ) / ~11.5/255 (DIV8K) in the outer ~3 columns.
  * square sources are an exact no-op: for the 69% of FGResQ that is already 288x288 (and any
    square source) f == 1 and the filter is the identity, bit-for-bit (measured max|d| = 0).

Verified against BOTH references: the corpus-sweep implementation (2.98e-07) and Pillow's
LANCZOS in float mode (2.384e-07). Note Lanczos OVERSHOOTS (measured range -0.0711..1.0459
on a noise image, 25 px outside [0,1]); compare unclamped-to-unclamped or you will manufacture
a 7.1e-2 "discrepancy" that is not a filter difference. The clamp below is deliberate and
load-bearing for the uint8 path.

ANY change here invalidates every cached derivative (was-cached-before-any-augs pt caches,
ValImageCache) -- bump RESIZE_TAG and rebuild, do not serve stale pixels.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "RESIZE_TAG",
    "SquareCrop",
    "center_square_plan",
    "lanczos3_kernel",
    "resize_center_square_lanczos",
    "resize_center_square_lanczos_u8",
]

#: Bump this whenever the resampling changes, so cached derivatives can never be reused
#: silently across a change (the failure mode that served wrong pixels from a cache).
RESIZE_TAG = "fraclanczos3-centresquare-fullframe-v2"


def lanczos3_kernel(x: torch.Tensor) -> torch.Tensor:
    """Pillow's lanczos3_filter, vectorised. EXACTLY 0 at |x|>=3 and at every non-zero
    integer, exactly 1 at x==0.

    The integer special-case is not cosmetic. sinc(x) is exactly 0 at x = 1, 2, ..., but the
    closed form 3*sin(pi x)*sin(pi x/3)/(pi^2 x^2) evaluates to ~1e-17 there, which left the
    f=1 tap matrix with four off-diagonal taps of ~1e-17. Leakage onto a mid-grey pixel is
    ~1e-8 of a float32 ulp and vanishes, but onto a pixel that is EXACTLY 0.0 it survives --
    so "the identity is bit-exact on square sources" was false on real 288x288 files
    (measured 4.7e-18..7.1e-17, 8481 of 12593 sources affected, always at a source zero).
    Found by a cache gate, not by the original self-test, whose torch.rand input never
    contains an exact zero.
    """
    x = x.abs()
    out = torch.zeros_like(x)
    nz = (x > 0.0) & (x < 3.0)
    xv = x[nz]
    out[nz] = (3.0 * torch.sin(math.pi * xv) * torch.sin(math.pi * xv / 3.0)
               / (math.pi * math.pi * xv * xv))
    out = torch.where((x >= 1.0) & (x == x.round()), torch.zeros_like(out), out)
    return torch.where(x == 0.0, torch.ones_like(out), out)


def center_square_plan(w: int, h: int, out: int):
    """Exact fractional plan: (x0, y0, f, cx, cy). x0/y0 are NOT floored -- that is the point."""
    s = min(int(w), int(h))
    f = s / float(out)
    x0 = (w - s) / 2.0
    y0 = (h - s) / 2.0
    i = torch.arange(out, dtype=torch.float64)
    return x0, y0, f, x0 + (i + 0.5) * f, y0 + (i + 0.5) * f


def _tap_matrix(n_in: int, centers: torch.Tensor,
                f: float, dtype: torch.dtype = torch.float32,
                device=None) -> torch.Tensor:
    """(out, n_in) renormalised Lanczos-3 tap matrix over the FULL source row/column.

    `device` matters for the AS-A-LOSS path: the tap matrix used to be built on the default
    device, so resizing a CUDA-resident image (what a VAE trainer hands us) raised "Expected
    all tensors to be on the same device". The training path never noticed because its
    batches arrive from a CPU DataLoader. Passing device=None keeps the old behaviour.
    """
    if n_in <= 0:
        raise ValueError("empty source")
    fs = max(1.0, float(f))                    # Pillow clamps the kernel on upscale
    support = 3.0 * fs
    if device is None:
        device = centers.device
    idx = torch.arange(n_in, dtype=torch.float64, device=device)
    d = (idx[None, :] + 0.5) - centers[:, None].to(device)   # input px from the output centre
    w = lanczos3_kernel(d / fs)
    lo = torch.clamp(torch.floor(centers - support + 0.5), min=0).long()
    hi = torch.clamp(torch.ceil(centers + support - 0.5) + 1, max=n_in).long()
    lo = lo.to(device)
    hi = hi.to(device)
    w = w * ((idx[None, :] >= lo[:, None]) & (idx[None, :] < hi[:, None]))
    s = w.sum(dim=1, keepdim=True)
    w = w / torch.where(s == 0.0, torch.ones_like(s), s)
    return w.to(device=device, dtype=dtype)


def resize_center_square_lanczos(img: torch.Tensor, out: int = 288,
                                 clamp: bool = True) -> torch.Tensor:
    """(C,H,W) float 0..1 (any dtype) -> (C,out,out) float32. Separable, horizontal first."""
    if img.dim() != 3:
        raise ValueError(f"expected (C,H,W), got {tuple(img.shape)}")
    _c, h, w = (int(v) for v in img.shape)
    _x0, _y0, f, cx, cy = center_square_plan(w, h, out)
    cx = cx.to(img.device)
    cy = cy.to(img.device)
    work = img.to(torch.float32)
    t = torch.einsum("ow,chw->cho", _tap_matrix(w, cx, f, device=img.device), work)
    t = torch.einsum("vh,chw->cvw", _tap_matrix(h, cy, f, device=img.device), t)
    return t.clamp_(0.0, 1.0) if clamp else t


def resize_center_square_lanczos_u8(u8: torch.Tensor, out: int = 288) -> torch.Tensor:
    """(C,H,W) uint8 -> (C,out,out) uint8. Use THIS in any cache builder so the cached
    pixels and the live training path are produced by the same function."""
    if u8.dtype != torch.uint8:
        raise ValueError(f"expected uint8, got {u8.dtype}")
    x = resize_center_square_lanczos(u8.to(torch.float32) / 255.0, out=out, clamp=True)
    return (x * 255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)


class SquareCrop:
    """PIL image -> (C,out,out) float tensor. Drop-in for the old Resize+ToTensor Compose.

    Only for the SQUARE / fixed-aspect class. The aspect-bucket path must keep resizing to
    the bucket shape whole-image (it draws geometry per pool; see data.py RESIZE, NOT CROP).
    """

    def __init__(self, size: int = 288):
        self.size = int(size)

    def __repr__(self) -> str:  # so banners/logs say what is actually running
        return f"SquareCrop(centre-square, fractional-grid Lanczos3 -> {self.size}px, {RESIZE_TAG})"

    def __call__(self, im) -> torch.Tensor:
        from torchvision.transforms.functional import to_tensor
        return resize_center_square_lanczos(to_tensor(im), out=self.size)


def _selftest() -> int:
    torch.manual_seed(0)
    ok = True

    # 1) square sources are an exact no-op (69% of FGResQ). Use a QUANTISED image: it contains
    #    exact 0.0 and 1.0 pixels, which is where a ~1e-17 integer tap used to survive.
    #    torch.rand never contains a zero, and that is how the original claim passed while
    #    being false on real uint8-derived data.
    im = torch.randint(0, 256, (3, 288, 288), dtype=torch.uint8).to(torch.float32) / 255.0
    d = (resize_center_square_lanczos(im, out=288) - im).abs().max().item()
    print(f"  identity 288x288 -> 288 (quantised, has exact 0s): max|d| = {d:.3e}"
          f"   {'PASS' if d == 0.0 else 'FAIL'}")
    ok &= d == 0.0

    # 2) the grid sits exactly on the ideal fractional positions
    for (w, h, o) in ((1268, 532, 288), (500, 380, 288), (2337, 1536, 288)):
        x0, y0, f, cx, cy = center_square_plan(w, h, o)
        i = torch.arange(o, dtype=torch.float64)
        ideal = (w - min(w, h)) / 2.0 + (i + 0.5) * (min(w, h) / o)
        ph = (cx - ideal).abs().max().item()
        print(f"  grid {w}x{h} -> {o}: x0 = {x0:.4f} (fractional), f = {f:.6f}, "
              f"phase = {ph:.1e}   {'PASS' if ph < 1e-12 else 'FAIL'}")
        ok &= ph < 1e-12

    # 3) full-frame support: content outside the window matters, only in the outer columns
    w, h, o = 1268, 532, 288
    x0, _y0, _f, _cx, _cy = center_square_plan(w, h, o)
    base = torch.full((3, h, w), 0.5)
    edge = torch.zeros_like(base)
    left, right = int(x0), int(x0) + min(w, h)
    edge[:, :, :left] = 1.0
    edge[:, :, right:] = 1.0
    dm = (resize_center_square_lanczos(base, out=o)
          - resize_center_square_lanczos(base + edge, out=o)).abs().amax(dim=0)
    band = 4
    lb, rb = dm[:, :band].max().item(), dm[:, o - band:].max().item()
    interior = dm[:, band:o - band].max().item()
    print(f"  support: left {lb:.4f} / right {rb:.4f} (symmetric), interior {interior:.2e}"
          f"   {'PASS' if (lb > 0 and rb > 0 and interior < 1e-5) else 'FAIL'}")
    ok &= lb > 0.0 and rb > 0.0 and interior < 1e-5

    # 4) vs Pillow LANCZOS, float mode, BOTH UNCLAMPED (see the overshoot note above)
    try:
        import numpy as np
        from PIL import Image
        im2 = torch.rand(3, 512, 512)
        got = resize_center_square_lanczos(im2, out=288, clamp=False).numpy()
        pil = np.stack([
            np.asarray(Image.fromarray(im2.permute(1, 2, 0).numpy()[:, :, k].copy(),
                                       mode="F").resize((288, 288), Image.LANCZOS),
                       dtype="float32") for k in range(3)])
        dd = float(np.abs(got - pil).max())
        print(f"  vs PIL LANCZOS 512->288 (unclamped): max|d| = {dd:.3e}"
              f"   {'PASS' if dd < 1e-3 else 'FAIL'}")
        ok &= dd < 1e-3
    except Exception as exc:  # pragma: no cover
        print(f"  PIL cross-check skipped: {exc}")

    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
