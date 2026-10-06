# rotasim-dinov3b-diffiqa-v1

*[Written by AI]*

**RotaSim** — ROtations Trained as Adapters for perceptual SIMilarity.
A perceptual similarity metric: a **frozen DINOv3-B/16 tower** (86M) plus **OFTv2
orthogonal rotation adapters** (all six linear kinds, block 32) on the DiffIQA
PNY+SNY corpus with 3-rater vote-gap weighting. **1.29M trainable parameters.**

This was the first **preference-axis champion** of RotaSim; the current
**recommended** checkpoint is `rotasim-dinov3b-fgbadi60-v1`.

## Two-axis results (always quote both — the axes are near-independent)

**S2 preference panel** (2AFC accuracy on fixed-seed cells, our protocol):

| fg (FGResQ sv2) | ba (BAPPS) | di (DiffIQA holdout) | mean-of-3 |
|---|---|---|---|
| — (see caveat) | — | — | **0.7250** @ step 1300 (argmax) |

Caveat: per-cell values at step 1300 are not banked on disk; the composite is.
The 400-step point of this lineage scored fg .7061 / ba .7847 / di .6232.

**GM12 — synthetic-FR grand mean** ({CSIQ, LIVE, TID2008, TID2013} x {PLCC, SRCC, KRCC},
raw Pearson, LIVE on the published `dmos_realigned` labels):

| CSIQ | LIVE | TID2008 | TID2013 | **GM12** |
|---|---|---|---|---|
| .810/.843/.658 | .829/.910/.737 | .733/.725/.544 | .712/.688/.508 | **.7247** (@1300) / .7306 (@1250 argmax) |

Per-dataset cells: `FR_benchmark_results.csv` in this directory. Published-protocol
lpips-vgg sits at ~.7372 GM12 — the @1250 argmax reaches within 0.007 of it with
~11x fewer trained parameters than lpips-vgg's head+trunk adaptation.

## Training recipe

DiffIQA PNY+SNY (74,730 rows), Bradley-Terry on 2AFC triplets, per-row weight
`min(1, gap / median_decisive_gap)` from the 3-rater vote gaps ("unlocked
information"); 60 slots/step, 1,300 steps (78k triplets, length curve argmax —
decline confirmed after), lr 1.2911e-4 constant, seed 1234, fp32, gold-star
optimizer (iterative orthogonal gradient, Sinkhorn, snr_cond). Run:
`runs/20261003-211208-rotasim-diffiqa60-native512-blk32-qkvoupdown-seed1234`.

## Use as a loss

Same drop-in contract as the earlier bundles (see HOWTO.md):
`dreamsim_oft.as_loss.OFTDreamsimFn`. It is a **quality-preference metric** —
unrelated scenes read "close", so ride the reconstruction pair with a pixel/content
term; never use standalone.

## Validation

`python validate_bundle.py` — identity d(x,x)==0, positive finite distances,
gradients reaching both arguments, per-image shape. PASS on this checkpoint
(2026-10-04, GPU protocol).
