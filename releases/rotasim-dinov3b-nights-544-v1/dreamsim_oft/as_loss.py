"""Our trained OFTv2 metric as a LOSS for a generative model.

DROP-IN REPLACEMENT for sinkosaur3-vae-studio's `sinkvae.losses.dreamsim.DreamsimFn`.
The interface is deliberately identical, so the swap in `sinkvae/losses/stack.py` is:

    # from sinkvae.losses.dreamsim import DreamsimFn
    # self.dreamsim_fn = DreamsimFn(device=device, cache_dir=...)
    from dreamsim_oft.as_loss import OFTDreamsimFn
    self.dreamsim_fn = OFTDreamsimFn(device=device,
                                     ckpt=r"<release dir>\\step000300.pt")

Everything else in the stack (`loss.dreamsim` weight, `record("dreamsim", ...)`, the
`perceptual_res` handling, the per-image (N,) contract) keeps working unchanged.

RELEASED ARTIFACT (fgresq-perceptual-288-v1, 2026-10-01): ONE tower, DINOv3-B/16, with
trained OFTv2 rotations (block_size 16, all six linear kinds, 622,080 params) at 288px.
FGResQ 2AFC: val 76.44% / test 73.83% (strict non-tie 74.20%). It beats the released
DreamSim ensemble on that corpus by +2.52pp (cluster CI [+1.16,+3.78]) and all 24 pyiqa
metrics scored there. Read the release README before wiring it in.

CONTRACT (must match DreamsimFn exactly):
    prep(x)        (N,3,H,W) in [-1,1] -> (N,3,S,S) in [0,1]
    __call__(a,b)  -> (N,) per-image distances. 0.0 at an identical pair, lower = more
                   similar. Differentiable w.r.t. BOTH arguments.

PREPROCESSING IS PART OF THE MODEL, NOT A DETAIL. The released checkpoint was TRAINED
under centre-square crop + fractional-grid Lanczos3 (`resize_center_square_lanczos`), which
is the DEFAULT here. The older `preprocess="squash"` mode (full-frame bicubic resize to a
square) STRETCHES the aspect ratio -- that is precisely the bug this project fixed on
2026-10-01 -- and it is kept only for the legacy 544 3-tower artifact. On the FGResQ corpus
the two differ measurably even frozen (centre-square 69.71 vs bicubic-squash 70.11, cluster
CI [-2.07,+1.11]); switching silently scores a slightly different metric. Both modes are
differentiable: the centre-square transform is a separable einsum against a constant tap
matrix, so gradients reach the decoder.

WHY `prep` OUTPUTS [0,1] AND DOES NOT NORMALISE: each branch applies its OWN normalisation
internally (ImageNet for DINOv3, 0.5/0.5 for SigLIP2, OpenAI-CLIP for MetaCLIP2).
Normalising here would double-normalise and silently wreck the features. This is the same
reason DreamsimFn does not normalise either.

MEMORY NOTE, and it is the reason this class exists rather than a fresh API: in the VAE
trainer the SOURCE image is a constant. `__call__` therefore detects a non-grad-requiring
argument and computes its embedding under `no_grad`, which roughly halves activation memory
with no change in the returned value. This is automatic; callers need do nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import PerceptualModel, model_kwargs_from_config


class OFTDreamsimFn:
    """Frozen perceptual distance backed by a trained OFTv2 checkpoint."""

    def __init__(self, device: str = "cuda", ckpt: str | None = None,
                 image_size: int | None = None, amp: str = "off", dtype=torch.float32,
                 preprocess: str = "centre-square", verbose: bool = True):
        """
        ckpt      : the released `step000300.pt` (or any `adapters.pt` / `checkpoint.pt`
                    from a dreamsim_oft run). The file MUST carry a `config` block; we
                    rebuild the model from it rather than from defaults, because the
                    forward-affecting settings (oft_scaled, use_cayley_neumann,
                    per_branch_norm, image_size) change the maths while leaving every
                    parameter NAME and SHAPE identical -- so a wrong rebuild loads cleanly
                    and computes a different model. There is precedent: a hand-listed
                    rebuild once produced a Neumann-series model scored over exact-Cayley
                    weights and returned 68% instead of ~96%.
        image_size: input resolution the TOWERS see. DEFAULT None = use the checkpoint's
                    OWN stored image_size, which is the only safe default: resolution is
                    forward-affecting (it decides the patch grid), so overriding it scores a
                    metric that was never validated. Pass an int only deliberately.
        preprocess: "centre-square" (DEFAULT, the trained class) | "squash" (legacy).
        amp       : "off" | "bf16". bf16 here is a SPEED choice, not a memory one -- measured
                    at 224/batch1, bf16 used MORE activation memory than fp32 (721 vs 234
                    MiB), probably autocast retaining fp32 copies for backward nodes.
                    The released metric was validated in fp32; prefer "off".
        """
        if not ckpt:
            raise ValueError("ckpt is required: pass the path to step000300.pt")
        if preprocess not in ("centre-square", "squash"):
            raise ValueError(f"preprocess must be 'centre-square' or 'squash', got {preprocess!r}")

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "adapter" not in ck:
            raise ValueError(f"{ckpt} does not look like a dreamsim_oft artifact "
                             f"(no 'adapter' key)")
        stored = ck.get("config") or {}
        if not stored:
            raise ValueError(
                f"{ckpt} carries no 'config'. The forward-affecting settings are then "
                f"unknown and any rebuild is a guess. Refusing to load a possibly "
                f"different model.")

        kwargs = model_kwargs_from_config(stored)
        if image_size is not None:
            kwargs["image_size"] = int(image_size)
        elif not kwargs.get("image_size"):
            raise ValueError(
                f"{ckpt} stores no image_size and none was passed; the patch grid would be "
                f"a guess. Pass image_size explicitly.")

        self.model = PerceptualModel(dtype=dtype, device=device, verbose=verbose,
                                     **kwargs).to(device)
        self.model.load_adapter_state_dict(ck["adapter"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.image_size = int(self.model.image_size)
        self.amp = amp
        self.preprocess = preprocess
        self.step = ck.get("step")
        self.metrics = ck.get("metrics")
        self.per_branch_norm = bool(getattr(self.model, "per_branch_norm", False))

        if verbose:
            print(f"[oft-metric] loaded step={self.step} image_size={self.image_size} "
                  f"preprocess={self.preprocess} per_branch_norm={self.per_branch_norm} "
                  f"amp={amp}")
            if self.metrics:
                print(f"[oft-metric] training metrics on record: {self.metrics}")

    # -- interface ------------------------------------------------------------
    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) in [-1,1] -> (N,3,S,S) in [0,1]. Resizes; does NOT normalise."""
        x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        if self.preprocess == "centre-square":
            if x01.shape[-1] == self.image_size and x01.shape[-2] == self.image_size:
                return x01
            # Centre-square crop + fractional-grid Lanczos3, the class the checkpoint was
            # TRAINED under. Separable einsum against a constant tap matrix => differentiable.
            from .resize import resize_center_square_lanczos
            return torch.stack([resize_center_square_lanczos(im, out=self.image_size)
                                for im in x01])
        if x01.shape[-1] != self.image_size or x01.shape[-2] != self.image_size:
            x01 = F.interpolate(x01, size=(self.image_size, self.image_size),
                                mode="bicubic", align_corners=False, antialias=True)
        return x01.clamp(0.0, 1.0)  # bicubic can overshoot; ToTensor never would

    @torch.no_grad()
    def _embed_frozen(self, x01: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(self.amp == "bf16" and self.device == "cuda")):
            return self.model.embed(x01)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a/b: (N,3,H,W) in [-1,1] -> (N,) distances. Differentiable w.r.t. both, EXCEPT
        an argument that does not require grad is evaluated under no_grad (that is the
        source image in a VAE loss, and detaching it halves the activation cost)."""
        pa, pb = self.prep(a), self.prep(b)

        if pa.requires_grad:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(self.amp == "bf16" and self.device == "cuda")):
                ea = self.model.embed(pa)
        else:
            ea = self._embed_frozen(pa)

        if pb.requires_grad:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(self.amp == "bf16" and self.device == "cuda")):
                eb = self.model.embed(pb)
        else:
            eb = self._embed_frozen(pb)

        # CLAMP THE SIMILARITY INTO [-1,1] BEFORE SUBTRACTING. F.cosine_similarity does not
        # guarantee |cos| <= 1 in floating point: an identical pair came back as 1 + 1.19e-07
        # on one element, so the distance was NEGATIVE. Harmless for an argmax decision -- the
        # only thing the research harness ever asked -- and WRONG for a loss: it breaks the
        # "exactly 0.0 at an identical pair" contract and injects a negative gradient term at
        # the optimum. Inside the range the clamp is the identity (verified: the FP16/FP32
        # distances and every study number in this project are unaffected); at the boundary
        # the gradient is 0, which is the correct behaviour at the minimum of a distance.
        cos = F.cosine_similarity(ea, eb, dim=-1).clamp(-1.0, 1.0)
        return (1.0 - cos).flatten()


def smoke_test(ckpt: str, image_size: int | None = None, device: str = "cuda") -> None:
    """Load the artifact and prove the contract holds. Run this BEFORE wiring anything up."""
    import time

    t0 = time.time()
    fn = OFTDreamsimFn(device=device, ckpt=ckpt, image_size=image_size)
    x = torch.rand(2, 3, image_size, image_size, device=device) * 2 - 1
    y = torch.rand(2, 3, image_size, image_size, device=device) * 2 - 1

    d_xy = fn(x, y)
    d_xx = fn(x, x)
    print(f"  shape        : {tuple(d_xy.shape)}  (contract wants (N,) = (2,))")
    print(f"  d(x,x)       : {d_xx.detach().cpu().tolist()}  (must be ~0)")
    print(f"  d(x,y)       : {d_xy.detach().cpu().tolist()}")
    print(f"  d(x,y) > 0   : {bool((d_xy > 0).all())}")
    print(f"  finite       : {bool(torch.isfinite(d_xy).all())}")

    # gradient must reach the INPUT (that is the whole point for a VAE decoder)
    xg = x.clone().requires_grad_(True)
    fn(xg, y).sum().backward()
    g = xg.grad
    print(f"  grad to input: {'YES' if g is not None else 'NO'}  "
          f"(nonzero: {bool(g is not None and g.abs().sum() > 0)})")
    print(f"  loaded in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import sys
    smoke_test(sys.argv[1] if len(sys.argv) > 1 else
               r"runs\544-warmstart-smoke\adapters.pt",
               int(sys.argv[2]) if len(sys.argv) > 2 else 544)
