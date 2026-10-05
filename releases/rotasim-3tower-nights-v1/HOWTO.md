# HOWTO — rotasim-3tower-nights-v1

*[Written by AI]*

This bundle is **self-contained**: the checkpoint, a vendored `dreamsim_oft/`
library, a validator, and the benchmark CSV.

## Files

| file | what |
|---|---|
| `step001200.pt` | the released 3-tower adapter checkpoint |
| `dreamsim_oft/` | vendored library, including the loss API `as_loss.py` |
| `validate_bundle.py` | self-test — run it (`python validate_bundle.py .`) |
| `VALIDATION_cpu.log` | the log of that self-test |
| `FR_benchmark_results.csv` | the FR cells behind the table row |
| `config.json` | the checkpoint's config, curated |
| `src_hashes.json` | sha256[:16] of each vendored module |
| `LICENSE` / `WEIGHTS-LICENSE.md` | code license / weights license |

## Use it as a loss (drop-in)

```python
import sys; sys.path.insert(0, r"<path to this bundle>")
from dreamsim_oft.as_loss import OFTDreamsimFn

fn = OFTDreamsimFn(device="cuda", ckpt=r"<bundle>/step001200.pt")
d = fn(recon, target)          # (N, 3, H, W) float in [-1, 1]  ->  (N,)
```

- Input is **[-1, 1]**, any resolution; the **224-square** resize is part of the
  model's preprocessing (it reads the scale from the checkpoint).
- Output is a cosine distance, non-negative, differentiable w.r.t. both arguments.
- **All three frozen towers** resolve at load: DINOv3-B/16 (gated HF repo — accept
  the license once), SigLIP2-base/16 and MetaCLIP2-B/16 (Hub downloads).

## Read this before using it as a loss

This is a **quality-preference** metric: unrelated clean scenes can read *closer*
than a clean-vs-degraded pair of one scene. As a loss it must ride the
reconstruction pair, paired with a pixel/content term — never standalone.

## Validate

```bash
python validate_bundle.py .        # d(x,x)==0, positive finite d, grads to both args
```
