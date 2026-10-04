# HOWTO — rotasim-dinov3b-diffiqa-v1

This bundle is **self-contained**: the checkpoint, a vendored `dreamsim_oft/`
library, a validator, and the benchmark CSV. Nothing else to install beyond
torch + numpy + pillow.

## Files

| file | what |
|---|---|
| `step001300.pt` | the released adapter checkpoint (length-curve argmax) |
| `dreamsim_oft/` | vendored library, including the loss API `as_loss.py` |
| `validate_bundle.py` | self-test — run it (`python validate_bundle.py .`) |
| `VALIDATION_cpu.log` | the log of that self-test |
| `FR_benchmark_results.csv` | per-dataset cells behind the GM12 number |
| `LICENSE` / `WEIGHTS-LICENSE.md` | code license / weights license |

## Use it as a loss (drop-in)

```python
import sys; sys.path.insert(0, r"<path to this bundle>")
from dreamsim_oft.as_loss import OFTDreamsimFn

fn = OFTDreamsimFn(device="cuda", ckpt=r"<bundle>/step001300.pt")
d = fn(recon, target)          # (N, 3, H, W) float in [-1, 1]  ->  (N,)
```

- Input is **[-1, 1]**, any resolution; the centre-square crop at **288** is part
  of the model's preprocessing (it reads the scale from the checkpoint).
- Output is a cosine distance, non-negative, **differentiable w.r.t. both
  arguments** (gradients reach a decoder).
- The frozen **DINOv3-B/16** tower resolves automatically. DINOv3 lives behind a
  gated HF repo — accept its license once (`huggingface-cli login`), or point the
  checkpoint config's tower path at a local copy.

## Read this before using it as a loss

This is a **quality-preference** metric, not absolute similarity: unrelated clean
scenes can read *closer* than a clean-vs-degraded pair of one scene. As a loss it
must ride the reconstruction pair and be paired with a pixel/content term — never
standalone.

## Validate

```bash
python validate_bundle.py .        # d(x,x)==0, positive finite d, grads to both args
```
