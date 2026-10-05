"""Self-test for the OFT ckpt bundle: loads BOTH checkpoints through the LOSS API
(as_loss.OFTDreamsimFn) and checks the documented contract. Run from anywhere with the
bundle's parent on PYTHONPATH-free access: python validate_bundle.py <bundle_dir>.

Checks per ckpt:
  * model builds from the ckpt's OWN config block (adapter step + params printed)
  * d(x, x) == 0 on random images (max abs < 1e-6)
  * d(a, b) > 0 for different images and is finite
  * gradient reaches BOTH arguments (backward on a constant scalar)
Prints [PASS]/[FAIL] per check; exit 1 on any failure.
"""

import sys
from pathlib import Path

import torch


def main(bundle: str) -> int:
    bundle = Path(bundle).resolve()
    sys.path.insert(0, str(bundle))
    from dreamsim_oft.as_loss import OFTDreamsimFn  # noqa: E402

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}")
    ok_all = True
    cks = sorted(p.name for p in bundle.glob("step*.pt"))
    if not cks:
        print("[FAIL] no step*.pt checkpoints in bundle")
        return 1
    for name in cks:
        ck = bundle / name
        if not ck.is_file():
            print(f"[FAIL] missing {name}")
            ok_all = False
            continue
        fn = OFTDreamsimFn(device=dev, ckpt=str(ck))
        g = torch.Generator().manual_seed(0)
        n = 8
        # STRUCTURED images (smooth random gradients + mild noise), not white noise:
        # two white-noise images are nearly identical to a CLS embedding (d ~ 0.002),
        # which makes the range/grad checks vacuous. 512x384 = non-square on purpose
        # (exercises the centre-square crop).
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 384), torch.linspace(0, 1, 512),
                                indexing="ij")
        base = torch.stack([yy, xx, (yy * xx)], dim=0).unsqueeze(0)  # (1,3,384,512)
        x = base + 0.05 * torch.rand((n, 3, 384, 512), generator=g)
        y = base.flip(-1) + 0.05 * torch.rand((n, 3, 384, 512), generator=g)
        x = (x * 2 - 1).to(dev)   # [-1,1] contract
        y = (y * 2 - 1).to(dev)
        d_self = fn(x, x)
        xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
        d_xy = fn(xr, yr)
        d_xy.sum().backward()     # grads do not exist until backward (first version's bug)
        checks = {
            "d(x,x)==0": float(d_self.detach().abs().max()) < 1e-6,
            "d(x,y)>0 finite": bool((d_xy.detach() > 0).all() and torch.isfinite(d_xy).all()),
            "grad reaches both args": xr.grad is not None and float(xr.grad.abs().sum()) > 0
                                      and yr.grad is not None and float(yr.grad.abs().sum()) > 0,
            "per-image shape (N,)": tuple(d_xy.shape) == (n,),
        }
        for k, v in checks.items():
            print(f"[{name}] {k}: {'PASS' if v else 'FAIL'}")
            ok_all &= v
        print(f"[{name}] d(x,y) = {[round(float(v), 4) for v in d_xy.detach()]}")
    print("[validate_bundle] " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent)))
