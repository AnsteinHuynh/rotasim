# HOWTO — rotasim-qwen38vit-diffiqa-v1

**Self-contained except for the frozen tower**, which is not shipped (size +
licensing). Everything else — checkpoint, vendored `dreamsim_oft/`, validator —
is in this directory.

## Files

| file | what |
|---|---|
| `step002800.pt` | the released adapter checkpoint (length-ladder argmax) |
| `dreamsim_oft/` | vendored library, including the loss API `as_loss.py` |
| `validate_bundle.py` | self-test — run it (`python validate_bundle.py .`) |
| `VALIDATION_cpu.log` | the log of that self-test |
| `tower_path.example.json` | how to point the bundle at your Qwen tower |
| `FR_benchmark_results.csv` | per-dataset cells behind the GM12 number |
| `LICENSE` / `WEIGHTS-LICENSE.md` | code license / weights license |

## Point it at the frozen tower (required, once)

The 460.7M **Qwen3.8-27B vision tower** is user-supplied. Either source works and
both loaders are gated bit-identical (333/333 tensors):

- an **HF checkpoint directory** (e.g. `Qwen/Qwen3.8-27B`), or
- the **llama.cpp mmproj gguf** file.

Set it via the config key `qwen_tower_path` (copy `tower_path.example.json` to
`tower_path.json`, or edit the checkpoint config) or the env var
`QWEN_TOWER_PATH`. The HF-dir route needs `safetensors` + `transformers` (≥5);
the mmproj route needs the `gguf` package.

## Use it as a loss (drop-in)

```python
import sys; sys.path.insert(0, r"<path to this bundle>")
from dreamsim_oft.as_loss import OFTDreamsimFn

fn = OFTDreamsimFn(device="cuda", ckpt=r"<bundle>/step002800.pt")
d = fn(recon, target)          # (N, 3, H, W) float in [-1, 1]  ->  (N,)
```

- Input **[-1, 1]**; centre-square crop at **288** from the checkpoint.
- **Mixed-size eval batches must be scored per image** (`QWEN_FWD_CHUNK=1`): the
  packed-varlen forward assumes one grid size per call. Non-multiple-of-32 inputs
  resample up to the next multiple of 32 (official Qwen processor behaviour).
- Differentiable w.r.t. both arguments.

## Read this before using it as a loss

A **quality-preference** metric (same caveat as the DINO bundle): unrelated clean
scenes can read closer than a degraded pair of one scene — ride the
reconstruction pair with a pixel/content term; never standalone.

## Validate

```bash
QWEN_TOWER_PATH=<your tower> python validate_bundle.py .
```
