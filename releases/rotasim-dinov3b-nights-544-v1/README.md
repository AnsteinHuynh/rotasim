---
license: apache-2.0
base_model: facebook/dinov3-vitb16-pretrain-lvd1689m
tags:
  - perceptual-similarity
  - image-quality
  - full-reference
  - oft
  - parameter-efficient-finetuning
  - dinov3
library_name: pytorch
---

# RotaSim — ROtations Trained as Adapters for perceptual SIMilarity

**RotaSim-dinov3b-nights-544** is a full-reference perceptual similarity metric: it takes two
images and returns a scalar distance that agrees with human judgments of perceptual similarity.
It is **not a finetune** — the entire trained artifact is a bag of 36 **OFTv2 block-orthogonal
rotations** (block size 32) injected into the q/k/v attention projections of a **frozen
DINOv3-B/16** (85.66 M parameters, untouched), read out as a cosine distance between adapted
CLS embeddings. That is **428,544 trainable parameters (0.50 %)** — a **1.7 MB** checkpoint you
can drop into any training loop.

| | n | accuracy |
|---|---|---|
| NIGHTS **val** | 1,720 | **94.59 %** |
| NIGHTS **test** | 1,824 | **94.79 %** |
| LIVE SRCC | — | **0.9201** |
| TID2013 SRCC | — | **0.8242** |
| TID2008 SRCC | — | **0.8116** |
| CSIq SRCC | — | **0.9405** |

The NIGHTS numbers are recomputed from this exact checkpoint, fp32, one protocol, with per-row
dumps retained. The IQA correlations (LIVE / TID2008 / TID2013 / CSIQ) put it **above published
LPIPS and DISTS rows on all four datasets** (see *Full-reference IQA benchmarks* below and
`fr_benchmark_results.csv`).

Trained on **NIGHTS only** (20,019 two-alternative-forced-choice rows, Bradley–Terry loss,
1,200 steps) — no IQA dataset was seen during training, so the benchmark rows above are
leak-clean.

## Why this checkpoint

It was selected for release on three measured grounds:

1. **Downstream parity.** In a sister project that uses the metric as a *training loss* for an
   SDXL VAE, this checkpoint matched the other release candidates within FID/SFID error.
2. **Best external-transfer scores** of all internally tracked candidates on the classic
   FR-IQA benchmark table (the full internal sweep of checkpoints and fusions ships in
   `fr_benchmark_results_full.csv`).
3. **Simplest provenance**: trained on NIGHTS alone, single tower, no mixture recipe to
   reproduce.

## What it is worth, honestly

* It does **not** beat the released DreamSim ensemble on NIGHTS. DreamSim v0.2.0 scores
  **96.16 %** on these same 1,824 test rows, i.e. this artifact is **−1.37 pp** against a much
  larger ensemble. Do not present it as a state-of-the-art NIGHTS claim.
* Its value is **cost and shape**: one frozen DINOv3-B/16, 0.50 % trainable parameters,
  **1.7 MB** on disk, differentiable all the way to the input, and it **ties the 2-tower
  (DINOv3-B + SigLIP2) sibling's test number (94.79 = 94.79) at one third of the adapter
  parameters** (428,544 vs 1,285,632).
* It is a **quality-preference metric**, not a content-identity one (see *Caveats*).

## Installation

Requires Python ≥ 3.12, PyTorch (CUDA recommended), `transformers`, `huggingface_hub`.
Validated runtimes: `python 3.14.3 / torch 2.13.0+cu132 / transformers 5.5.4` and
`python 3.14.3 / torch 2.11.0+cu128 / transformers 5.16.1`.

> **transformers 4.x does NOT work** — the OFT surface walk fails with
> `RuntimeError: inject_oft found NO nn.Linear matching suffixes=(...)`. Use 5.x.

```bash
pip install torch transformers huggingface_hub   # or your usual stack
git clone https://github.com/AnsteinHuynh/rotasim.git
cd rotasim/releases/rotasim-dinov3b-nights-544-v1
```

### Getting the backbone (one-time, required)

Meta keeps DINOv3 behind a **gated** Hugging Face repo: you must be logged in and have accepted
the license once. Then either

```bash
huggingface-cli login
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
    --local-dir C:/myPath/dinov3-vitb16-pretrain-lvd1689m
```

and point `DINOV3_VITB16_PATH` in **`CONFIG.py`** (repo root) at that folder:

```python
# CONFIG.py
DINOV3_VITB16_PATH = r"C:/myPath/dinov3-vitb16-pretrain-lvd1689m"
```

The folder is the standard HF snapshot layout — it should contain `model.safetensors`,
`config.json`, `preprocessor_config.json`, etc. (exactly what `huggingface-cli download`
produces).

…or leave `DINOV3_VITB16_PATH = ""` and the package will fetch the tower straight from the Hub
by repo id, which works if you ran `huggingface-cli login` and accepted the gate in the browser.

## Quickstart

`x` and `y` are `(N, 3, H, W)` float tensors in `[-1, 1]` at **any** resolution; the output is
`(N,)` cosine distances.

```python
import sys, torch
sys.path.insert(0, ".")                     # this bundle
from dreamsim_oft.as_loss import OFTDreamsimFn

metric = OFTDreamsimFn(device="cuda", ckpt="step001200.pt")
# image_size (544) and preprocess ("centre-square") come from the checkpoint itself

recon  = torch.rand(4, 3, 512, 512, device="cuda") * 2 - 1
target = recon + 0.02 * torch.randn_like(recon)
loss = metric(recon, target).mean()         # differentiable all the way back to `recon`
loss.backward()                             # gradients reach the input
```

`example.py` in this repo is a runnable version of the above.

Contract, verified in `VALIDATION.log` against the shipped copy:

| property | value |
|---|---|
| input | `(N, 3, H, W)` float in `[-1, 1]`, any resolution, CUDA |
| output | `(N,)` cosine distance, non-negative |
| `d(x, x)` | exactly `0.0` (floored; identical input short-circuits) |
| gradient to input | yes, non-zero to the decoder |
| trainable parameters | none — the released metric is fully frozen |
| preprocessing | `centre-square` (fractional Lanczos3, full-frame, differentiable) |

## Full-reference IQA benchmarks (leak-clean)

Protocol: pyiqa-bench pair lists and labels, **native resolution** (no crop/resize), one
forward per unique image, PLCC after the standard 4-parameter logistic fit. All reference rows
are the published pyiqa numbers re-measured under this protocol. Trained on NIGHTS only —
none of these four datasets were seen in training.

| metric | CSIQ SRCC | LIVE SRCC | TID2008 SRCC | TID2013 SRCC |
|---|---|---|---|---|
| ssim | 0.837 | 0.910 | 0.624 | 0.627 |
| ms_ssim | 0.913 | 0.951 | 0.854 | 0.786 |
| lpips | 0.923 | 0.924 | 0.715 | 0.745 |
| lpips-vgg | 0.883 | 0.932 | 0.654 | 0.670 |
| dists | 0.930 | 0.948 | 0.665 | 0.708 |
| **rotasim-dinov3b-nights-544** | **0.941** | **0.920** | **0.812** | **0.824** |
| topiq_fr | 0.967 | 0.976 | 0.923 | 0.917 |

Full table with PLCC/KRCC and every reference metric: `fr_benchmark_results.csv`. The complete
internal sweep (every candidate checkpoint and score-fusion variant tried before this release)
is `fr_benchmark_results_full.csv`.

## Framing and geometry: measured, not assumed

The trained policy draws a per-row aspect bucket and **resizes the full frame** (one of five
buckets around 544 px: `448x688, 480x640, 544x544, 640x480, 688x448`). A loss cannot be handed
a per-row random geometry, so the loss-time contract is the trained centre-square readout.
Measured on this checkpoint, joined per row by id:

| split | input policy | n | accuracy | vs the aspect pass |
|---|---|---|---|---|
| val | 5-bucket aspect (evaluator default) | 1,720 | 94.59 % | — |
| val | 544 centre-square crop (loss-time policy) | 1,720 | 94.59 % | 22 flips (1.28 %), **+0.000 pp** |
| test | 5-bucket aspect | 1,824 | 94.79 % | — |
| test | 544 centre-square crop | 1,824 | 94.79 % | 24 flips (1.32 %), **+0.000 pp** |
| test | 5-bucket aspect, alternative geometry seed | 1,824 | 94.63 % | 23 flips, −0.164 pp |

**No new preprocessing mode is needed to use this artifact.** Accuracy is not uniform across
the five buckets (spread 2.61 pp, 93.77 → 96.37 %) — report a bucket table with any claim.

`preprocess` is part of the model, not a cosmetic detail — a caller who overrides
`preprocess="squash"` gets a metric that is neither calibrated nor monotone with the shipped
one (measured: +569.8 % distance error on square input, −31.1 % on 2:1 input). Leave it at the
checkpoint's own value.

## What the metric actually responds to (measured on real photographs)

Use these to set a loss weight instead of inheriting one from LPIPS:

| perturbation | d |
|---|---|
| identical | 0.00000 |
| gaussian sigma 1/255 | 0.00002 |
| gaussian sigma 4/255 | 0.00067 |
| gaussian sigma 16/255 | 0.01767 |
| 3x3 box blur | 0.01718 |
| 2x down-up resample | 0.02512 |
| 5x5 box blur | 0.05512 |
| corpus candidate pair, vs ref | 0.32159 / 0.39125 |
| **unrelated real scene** | **0.91281** |

Reading: on real content the metric is **structure-dominated** — the smearing a VAE decoder
actually produces costs far more than imperceptible dither. And the honest counter-row sits
right there: an unrelated clean scene reads *farther* than any degraded pair of one scene,
which is why this is a **quality-preference** metric and why it belongs beside a pixel/content
term when used as a loss.

## Cost as a loss term (measured, fp32)

| batch | input | peak VRAM |
|---|---|---|
| 8 | 512 px | 4,409 MiB |
| 16 | 512 px | 8,098 MiB |

Model load ≈ 4–13 s; 85.66 M frozen parameters in fp32 (`amp=off` — the checkpoint's numerics
are fp32 at inference).

## Provenance

| | |
|---|---|
| checkpoint | `step001200.pt` (the ckpt's own `step` field is `1200`) |
| backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m`, 85.66 M frozen |
| adapter | `adapter_type=oft`, `block_size=32`, `oft_scaled=True`, fp32 weights, exact Cayley solve, dropout 0.0 |
| surface | **q/k/v only** (36 wrapped Linears = 12 layers x 3; asserted by module name in the validator, not read from config) |
| params | 428,544 trainable / 86,088,960 total (**0.4978 %**) |
| objective | Bradley–Terry loss, `bt_tau=0.05` |
| optimiser | AdamW, lr 3e-4 → 3e-6 cosine, warmup 100 steps, batch 20 triplets (60 images/step) |
| data | NIGHTS 2AFC (20,019 rows), `image_size=544`, aspect-bucketed, `color_jitter=0.03`, `seed=1234` |
| numerics | train bf16 / inference fp32; `gradient_checkpointing=True`; `baked_pos=True`; `normalize_embeds=True` |

`config.json` is the checkpoint's own config block (training keys unchanged).
`src_hashes.json` pins the sha256[:16] of every vendored module.

## Validate it yourself

`validate_release.py` re-checks size, sha256, step, OFT surface, identity contract, gradients,
preprocessing and cost **against this directory on your machine**:

```powershell
python validate_release.py
# optionally, to also measure the sensitivity table on real photographs:
$env:NIGHTS_ROOT = "<path to a NIGHTS corpus copy>"; python validate_release.py
```

`VALIDATION.log` is the output of that script on the release box
(sha256 `2f7a0d2e019344a25d56179c36c8c20ecc25f3facf1a4a86c1b061869e03db0f`).

## Files

```
step001200.pt                  the released checkpoint (carries its own `config` block)
dreamsim_oft/                  vendored inference + loss package (13 modules)
CONFIG.py                      YOUR backbone path settings (see Installation)
config.json                    the checkpoint's config, curated
validate_release.py            re-runs every claim in this README against THIS directory
VALIDATION.log                 output of that script on the release box
src_hashes.json                sha256[:16] of each vendored module
fr_benchmark_results.csv       curated FR-IQA benchmark table (reference rows + this ckpt)
fr_benchmark_results_full.csv  the full internal sweep (all candidate ckpts and fusions)
example.py                     minimal runnable usage example
LICENSE                        Apache-2.0 (adapter weights + code)
LICENSE-DINOV3                 Meta's DINOv3 License (the frozen backbone)
```

## License

* The **adapter weights, vendored code and documentation** in this repository are
  **Apache-2.0** (see `LICENSE`).
* The **frozen DINOv3-B/16 backbone is Meta's DINOv3 License** and is NOT redistributed here —
  you download it yourself from the gated Hub repo. That license permits use, modification and
  redistribution (including commercially) under its conditions: redistribute the backbone and
  derivatives only under the same agreement, acknowledge Meta's DINO Materials in any
  publication using it, and do not use it for military/warfare, nuclear, espionage or weapons
  purposes. Full text: `LICENSE-DINOV3`.

## Caveats

* **Quality preference, not content identity.** It can rank an unrelated clean scene as *more
  different* than a degraded pair of one scene — as a loss it belongs on a **reconstruction
  pair** (content tied by construction) and beside a pixel/content term; it cannot by itself
  stop a decoder from drifting in content while looking clean.
* **Trained on NIGHTS.** Other corpora (e.g. restoration-quality datasets) are different
  distribution regimes; do not compare accuracy numbers across them.
* **Single-run artifact.** Any fine-grained cross-run comparison below ~1.5 pp carries a
  measured run-level noise term (~0.6–0.7 pp RMS) and needs replication.
* **Backbone gating.** The checkpoint is useless without the gated DINOv3 download — plan for
  that in CI/offline use.

## Citation

```bibtex
@misc{rotasim2026,
  author  = {Anstein Huynh},
  title   = {RotaSim: ROtations Trained as Adapters for perceptual SIMilarity},
  year    = {2026},
  url     = {https://github.com/AnsteinHuynh/rotasim},
  note    = {rotasim-dinov3b-nights-544, adapter checkpoint step001200}
}
```
