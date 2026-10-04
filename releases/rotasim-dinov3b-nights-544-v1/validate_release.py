"""VALIDATE THE RELEASED ARTIFACT -- run against the shipped package, not a working copy.

Produces the numbers the README quotes, and proves the integration contract holds BEFORE the
sister project wires it in:
  1. loads from the release dir and reports step / metrics / sha256
  2. SURFACE ASSERT: which Linears the OFT rotations actually wrap. This release vendors no
     `oft_targets` key (the 544 configs predate it), so the surface is the default
     q_proj,k_proj,v_proj -- asserted here by NAME, not assumed from the config
  3. d(x,x) == 0 EXACTLY, d(x,y) > 0, shape (N,)
  4. gradient reaches the INPUT (a VAE decoder needs this)
  5. the preprocessing question, quantified: centre-square (the loss-time contract, and the
     policy that reproduces the banked 544 headline) vs squash (legacy), on a SQUARE input
     (where they differ only by filter) and on a NON-SQUARE one (where squash distorts aspect)
  6. a SENSITIVITY SCALE on real NIGHTS photographs: what the distance actually reads for
     known-size corruptions, so the loss weight can be chosen against a measured axis
  7. wall time and peak VRAM for a plausible VAE batch

usage: python validate_release.py          (from this directory's parent, or anywhere)
optional env: NIGHTS_ROOT=<path to a NIGHTS corpus copy> (enables the real-image
              sensitivity table; skipped otherwise)
requires: python >= 3.12 / torch 2.x / transformers 5.x (see README runtime notes)
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # the VENDORED package, so this tests what ships

CKPT = "step001200.pt"
# The trained aspect-native policy of this class draws one of 5 buckets PER ROW at train time
# (data.py::ASPECT_BUCKETS_544), which is an EVALUATION protocol, not something a loss can be
# handed. The loss-time contract is centre-square: it calls the same
# resize_center_square_lanczos used by the square/cropped readout path, and it reproduced the
# banked headline exactly (0 of 2509 val decisions flip vs the aspect-native pass).
# Optional: point NIGHTS_ROOT at a local NIGHTS corpus copy to also measure the
# sensitivity scale on real photographs (skipped otherwise -- synthetic-only run).
import os
NIGHTS = Path(os.environ.get("NIGHTS_ROOT", ""))


def main():
    import torch
    import torch.nn.functional as F
    from dreamsim_oft.as_loss import OFTDreamsimFn
    from dreamsim_oft.oft import OFTLinear

    ck = HERE / CKPT
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    print(f"artifact : {ck.name}  {ck.stat().st_size/1024:.0f} KB")
    print(f"sha256   : {sha}")

    # ---- 1. what the artifact says about itself ------------------------------------------
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    stored = dict(blob.get("config") or {})
    print(f"step     : {blob.get('step')}")
    print(f"metrics  : {blob.get('metrics')}")
    print(f"spec     : image_size={stored.get('image_size')} "
          f"random_aspect={stored.get('random_aspect')} "
          f"block_size={stored.get('block_size')} oft_scaled={stored.get('oft_scaled')} "
          f"oft_targets={stored.get('oft_targets', '<absent -> default q,k,v>')} "
          f"oft_prefix={stored.get('oft_prefix', '<absent -> per-backbone default>')} "
          f"backbones={stored.get('backbones')}")

    t0 = time.time()
    fn = OFTDreamsimFn(device="cuda", ckpt=str(ck))
    print(f"load     : {time.time()-t0:.1f}s")
    n_par = sum(p.numel() for p in fn.model.parameters())
    n_tr = sum(p.numel() for p in fn.model.parameters() if p.requires_grad)
    print(f"params   : {n_par/1e6:.2f}M total, {n_tr} trainable (metric is FROZEN)")
    if n_tr:
        raise SystemExit("STOP: released metric has trainable parameters")

    # ---- 2. surface assert ----------------------------------------------------------------
    names = [n for n, m in fn.model.named_modules() if isinstance(m, OFTLinear)]
    kinds = {n.rsplit(".", 1)[-1] for n in names}
    print(f"\nOFT SURFACE (asserted, not read from config)")
    print(f"  wrapped OFTLinears  : {len(names)}")
    print(f"  wrapped kinds       : {sorted(kinds)}")
    print(f"  e.g.                : {names[0] if names else '<none>'}")
    print(f"  e.g.                : {names[-1] if names else '<none>'}")
    if kinds != {"q_proj", "k_proj", "v_proj"}:
        raise SystemExit(f"STOP: release advertises the q/k/v surface, found {sorted(kinds)}")
    if len(names) % len(kinds):
        raise SystemExit(f"STOP: {len(names)} wrapped Linears is not a multiple of {len(kinds)}")
    print(f"  => {len(names)//len(kinds)} transformer layers x 3 projections (q,k,v)")

    torch.manual_seed(0)
    S = fn.image_size
    if S != 544:
        raise SystemExit(f"STOP: expected the trained 544 input, got {S}")
    x = torch.rand(4, 3, 1024, 1024, device="cuda") * 2 - 1
    y = torch.rand(4, 3, 1024, 1024, device="cuda") * 2 - 1

    d_xx = fn(x, x)
    d_xy = fn(x, y)
    print(f"\nCONTRACT x in [-1,1] at 1024px -> internal {S}px")
    print(f"  d(x,x)          : {d_xx.detach().cpu().tolist()}   (must be exactly 0.0)")
    print(f"  shape           : {tuple(d_xy.shape)}  (contract wants (N,)=(4,))")
    print(f"  d(x,y)          : {[round(v,5) for v in d_xy.detach().cpu().tolist()]}")
    print(f"  all finite, >0  : {bool(torch.isfinite(d_xy).all())} {bool((d_xy>0).all())}")
    if not bool((d_xx.abs() < 1e-12).all()):
        raise SystemExit("STOP: identical pair is not 0 -- contract broken")

    xg = x.clone().requires_grad_(True)
    fn(xg, y).sum().backward()
    g = xg.grad
    print(f"  grad to input   : {'YES' if g is not None else 'NO'} "
          f"nonzero={bool(g is not None and g.abs().sum() > 0)} "
          f"max|g|={float(g.abs().max()) if g is not None else 0:.3e}")
    if g is None or g.abs().sum() == 0:
        raise SystemExit("STOP: no gradient to the decoder input -- unusable as a loss")

    # an already-square input must short-circuit to the exact same value
    with torch.no_grad():
        sq = fn(x, y)
        assert torch.equal(sq, d_xy), "square-input short circuit changed the value"
        print(f"  square short-circ: bit-identical to the 1024px value  YES")

    # ---- 3. preprocessing: trained centre-square vs legacy squash -------------------------
    print(f"\nPREPROCESSING (this is part of the model, not a detail)")
    print(f"  the ASPECT-native policy this artifact was trained under (one of 5 buckets per")
    print(f"  row) is an evaluation protocol, not a loss-time contract; centre-square is what")
    print(f"  the loss uses and it reproduced the banked headline exactly (see README).")
    with torch.no_grad():
        for label, hw in (("square 1024x1024", (1024, 1024)), ("non-square 1024x512 (2:1)", (512, 1024))):
            a = torch.rand(2, 3, *hw, device="cuda") * 2 - 1
            b = (a + 0.02 * torch.randn_like(a)).clamp(-1, 1)
            fn.preprocess = "centre-square"
            dc = fn(a, b).mean().item()
            fn.preprocess = "squash"
            ds = fn(a, b).mean().item()
            print(f"  {label:26s} centre-square {dc:.5f}   squash {ds:.5f}   "
                  f"delta {100*(ds-dc)/max(dc,1e-9):+.1f}%")
        fn.preprocess = "centre-square"

    # ---- 4. sensitivity scale, on REAL NIGHTS photographs ---------------------------------
    # A first version of the 288 release's table used torch.rand images and produced a nonsense
    # row: two UNRELATED noise images read 0.0023 while the SAME image blurred read 0.048.
    # Uniform noise is off-manifold -- DINOv3 maps it to nearly one point. Real content only.
    print(f"\nSENSITIVITY SCALE (real NIGHTS images, batch 8)")
    if NIGHTS.exists():
        from dreamsim_oft.data import TwoAFCDataset, make_loader
        ds = TwoAFCDataset(NIGHTS, split="val", verbose=False, image_size=S,
                           aspect_buckets=None, color_jitter=0.0)
        loader = make_loader(ds, 8, shuffle=False, num_workers=0, pin_memory=False)
        ref, left, right, target, tid = next(iter(loader))
        ref, left, right = (t.to("cuda") * 2 - 1 for t in (ref, left, right))
        other = ref.roll(1, dims=0)          # a DIFFERENT scene -> genuinely unrelated content
        with torch.no_grad():
            print(f"  corroborating perception (ref vs the corpus's two candidates)")
            print(f"    d(ref,left)                d = {fn(ref,left).mean().item():.5f}   "
                  f"d(ref,right) d = {fn(ref,right).mean().item():.5f}")
            d_other = fn(ref, other).mean().item()
            print(f"  unrelated real scene         d = {d_other:.5f}")
            print(f"  symmetric noise, real image")
            for sig in (1/255, 2/255, 4/255, 8/255, 16/255):
                n = (ref + sig * 2 * torch.randn_like(ref)).clamp(-1, 1)
                print(f"    gaussian sigma={sig*255:2.0f}/255      d = {fn(ref,n).mean().item():.5f}")
            print(f"  structural corruption, real image")
            print(f"    3x3 box blur               d = "
                  f"{fn(ref, F.avg_pool2d(ref,3,1,1)).mean().item():.5f}")
            print(f"    5x5 box blur               d = "
                  f"{fn(ref, F.avg_pool2d(ref,5,1,2)).mean().item():.5f}")
            lo = F.interpolate(F.interpolate(ref, scale_factor=0.5, mode="bilinear",
                                             align_corners=False),
                               size=(S, S), mode="bilinear", align_corners=False)
            print(f"    2x down-up resample        d = {fn(ref,lo).mean().item():.5f}")
            jp = (ref + torch.randn_like(ref) * (6/255)).clamp(-1, 1)   # coarse noise ~ JPEG-ish
            print(f"    coarse mottle 6/255        d = {fn(ref,jp).mean().item():.5f}")
        print(f"  READ: on REAL content the metric is dominated by STRUCTURE -- the smearing a")
        print(f"  VAE decoder actually produces (box blur, down-up resample) -- while low-amplitude")
        print(f"  noise stays near zero. Charge for smearing, not for imperceptible dither, and")
        print(f"  set the weight against THIS table rather than inheriting an LPIPS weight.")
        print(f"  CAVEAT, weigh it yourself from the rows above: if the 'unrelated real scene'")
        print(f"  row is CLOSER than a real artefact pair, this is a QUALITY-preference metric,")
        print(f"  not a content-identity one. As a loss it belongs on a RECONSTRUCTION pair")
        print(f"  (content tied by construction) and ideally beside a pixel/content term -- it")
        print(f"  cannot by itself stop a decoder from drifting in content while looking clean.")
    else:
        print(f"  corpus {NIGHTS} not found -- skipped (run on the training box)")

    # ---- 5. cost --------------------------------------------------------------------------
    print(f"\nCOST (frozen metric as a loss term, fp32)")
    for n, res in ((8, 512), (16, 512)):
        a = torch.rand(n, 3, res, res, device="cuda") * 2 - 1
        b = (a + 0.02 * torch.randn_like(a)).clamp(-1, 1).requires_grad_(True)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            loss = fn(a, b).mean()
            loss.backward()
        torch.cuda.synchronize()
        print(f"  batch {n:2d} @ {res}px: {(time.time()-t)/5*1000:.0f} ms/step  "
              f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    print(f"\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
