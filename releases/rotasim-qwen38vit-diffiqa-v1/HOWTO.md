# OFT perceptual metric bundle -- oft-ckpt-bundle-2026-10-01-night
The best artifact this project has measured, plus a conservative alternative, both as
adapter-only checkpoints with the vendored library. **READ `## src patches` below before you
wire the multichannel one in -- it needed two fixes to be usable as a loss.**

## Files

| file | what |
|---|---|
| `step000400_mc3warm400.pt` | PRIMARY -- best measured artifact |
| `step000300_incumbent300.pt` | shipped alongside: different readout / trade-off point |
| `dreamsim_oft/` | vendored library (inference-relevant modules, INCLUDING the two patches documented below). PYTHONPATH the bundle root or just run from it. |
| `validate_bundle.py` / `VALIDATION.log` | self-test: loads each ckpt through the LOSS API, checks `d(x,x)==0`, positive/finite distances, gradient to BOTH arguments. RUN IT. |

## Usage (5 lines, unchanged from the previous bundle)

```python
import sys; sys.path.insert(0, r"<path to this bundle>")
from dreamsim_oft.as_loss import OFTDreamsimFn
fn = OFTDreamsimFn(device="cuda", ckpt=r"<bundle>\step000400_mc3warm400.pt")
d = fn(recon_minus1_1, target_minus1_1)   # (N,3,H,W) in [-1,1] -> (N,)
```

Contract notes that bite: input is **[-1,1]**; preprocessing is part of the model
(`preprocess="centre-square"`, 288 is the trained scale); it is a **quality-preference** metric,
not absolute similarity -- unrelated clean scenes can read closer than a degraded pair of one
scene, so ride the reconstruction pair and pair it with a pixel/content term.

## SCOREBOARD (equal-thirds, 2509 held-out rows, ONE process, sv2 carve)

| panel / arm | fgresq (837, sv2) | bapps (836) | diffiqa (836) | COMPOSITE | fgresq full (2509) |
|---|---|---|---|---|---|
| `s2_panel_sv2.json` / frozen | 0.7037 | 0.7859 | 0.5323 | **0.6740** | 0.6971 |
| `s2_panel_sv2.json` / incumbent@300 | 0.7467 | 0.8421 | 0.4964 | **0.6951** | 0.7644 |
| `s2_panel_sv2.json` / r3@382 | 0.7073 | 0.8170 | 0.5969 | **0.7071** | 0.7067 |
| `s2_panel_sv2.json` / r5@400 | 0.7097 | 0.7883 | 0.6268 | **0.7082** | 0.7122 |
| `s2_panel_sv2.json` / mc3@200 | 0.7360 | 0.8337 | 0.6280 | **0.7326** | 0.7453 |
| `s2_panel_sv2.json` / mc3@400 | 0.7491 | 0.8385 | 0.6663 | **0.7513** | 0.7461 |

Columns: fgresq = first 837 rows of the **sv2 (pool-disjoint)** val carve; bapps = 836 rows of
the official BAPPS val split; diffiqa = 836 rows of the official DiffIQA Validation split;
COMPOSITE = their unweighted mean; fgresq full = the whole 2509-row sv2 val (the number
comparable to the banked 76.44). Single draws; per-cell noise ~+-1.5-2pp, so read the big gaps.

**Harness validation:** `frozen` fgresq-full reproduces the banked 69.71 and `incumbent@300`
reproduces the banked 76.44 EXACTLY. An earlier version of this panel used the sv1 (pool-LEAKY)
carve, which inflated trained arms by ~1.5-3pp (the incumbent's fgresq cell read 77.42 there vs
74.67 clean); this table is the clean one.

## What to take from the table

1. **The mixture arms now beat the incumbent on the composite**, and the champion does it while
   MATCHING the incumbent on the fgresq cell (see the table) -- at the cost of ~1.8pp on fgresq
   FULL-val. It gains **+17pp on DiffIQA** over the incumbent (which is *below the frozen
   baseline* there).
2. **The gain is NOT the learned head.** At step 400 the head's logits were
   [-0.0126, -0.0055, +0.0169] -> weights ~0.329/0.332/0.339, a whisker from uniform. The win
   comes from the multi-layer DISTANCE plus the warm-started rotations. The scalar warm-start
   control in the table (same warm start, same data, readout OFF) is the arm that attributes it
   -- read those two rows together before crediting the mechanism.
3. **Your downstream verdict still stands as the decider.** Our S3 finding was that the FGResQ
   specialist wins your VAE pipeline. A **||g|| ordering** ("the incumbent's gradient magnitude
   predicts your FID") was reported to us as incumbent 0.9456 > r5 0.6280 > r3 0.0226; the r5 and
   r3 values have since been **RETRACTED BY THEIR OWN AUTHOR** (a probe-harness bug: one gradient
   per input param of the SUMMED loss). Only the incumbent's **0.9456** -- a single-term
   measurement -- should be quoted, and mc3's ||g|| is UNKNOWN. Nothing here changes the S3
   verdict; this bundle is simply a strictly broader metric to A/B against the specialist.

## DOWNSTREAM OUTCOME (added 2026-10-01 07:15, after the sister project's A/B)

**This bundle's mc3 was A/B'd against the incumbent as a VAE training loss, and mc3 LOST** --
exactly as this README pre-registered it would. Their full-path protocol (genuinely unfrozen
encoder, 288px, pyiqa fid/sfid, 400 steps, equal-|g| weights):

| arm @400 | FID@256 | sfid@256 | FID@512 | sfid@512 |
|---|---|---|---|---|
| incumbent (scalar) seed 1 | 33.10 | 122.58 | **24.05** | **103.62** |
| incumbent seed 2 | -- | -- | 24.77 | 105.26 |
| **mc3@400 (this bundle)** | 33.46 | 123.26 | 25.24 | 105.25 |
| gold-star optimizer (incumbent stack), 2 draws | 32.30 | 122.78 | 24.75 / 24.88 | 105.11 / 105.80 |

mc3 is the worst of the five draws (+1.19 FID@512 vs the incumbent seed 1). Their full-path
run-to-run spread is **+-0.72 FID / +-1.6 sfid**, so the gap is ~1.7x the spread: a SMALL loss, not
a rout. The calibrated reason is now measured: mc3's decoder-subset **||g|| = 0.379 vs the
incumbent's 0.974** (single-term probe, validated against their banked table).

**THEREFORE: use `step000300_incumbent300.pt` as the training loss.** It is the project's best
checkpoint by the only criterion that crowns it (downstream VAE quality). `step000400_mc3warm400.pt`
is the best SCOREBOARD artifact this project has (S2 composite 0.7513, +17pp DiffIQA) and a strictly
broader metric -- a candidate to A/B *if* a later arm recovers the FGResQ fidelity that this one
traded away (the queued `mcA` arm: same readout, FGResQ-only training).
## src patches applied for this bundle (summary; full text in `PATCHES.md`)

Two defects, both of the "silently scores a different model" class:
(1) `as_loss` ignored a multichannel checkpoint's head entirely (measured 74.30 vs the
checkpoint's own 72.90 on 1000 val rows); (2) `distance_channels` could return a NEGATIVE
distance at an identical pair (-7.95e-08 measured) because the cosine was not clamped.
Both fixed and verified; `PATCHES.md` has the details, the measurements and the scale caveat.

## Provenance

- champion: `runs\20261001-020022-rotasim-mc3-warm-b60fg60di-blk16-qkvoupdown-seed1234\ckpt\step000400.pt` (see `PATCHES.md`/this file for the recipe: DINOv3-B/16 frozen tower +
  OFTv2 blk16 all-six, 622,080 adapter params, warm-started from the FGResQ incumbent's step-300
  peak, then trained on shard-cached pixels with per-corpus native resolutions).
- corpus pixels: fgresq = VSR-ultra 288 fp32 (generated by the nvvsr app at output_size 288);
  DiffIQA = native 512 u8; BAPPS = native 256 u8. NO val/test rows from any corpus were trained
  on; the fgresq val carve is sv2 (pool-disjoint), BAPPS/DiffIQA use their official val splits.
- evaluation: one process, fp32, `eval_s2_fast.py` (tower built once, adapters swapped).
