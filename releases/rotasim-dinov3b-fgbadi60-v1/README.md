# rotasim-dinov3b-fgbadi60-v1

*[Written by AI]*

**RotaSim** — ROtations Trained as Adapters for perceptual SIMilarity.
A perceptual similarity metric: a **frozen DINOv3-B/16 tower** (86M) plus **OFTv2
orthogonal rotation adapters** (all six linear kinds, block 32) trained on a
**three-corpus gap-weighted mixture** — FGResQ + BAPPS + DiffIQA, 60 slots each
per step, Bradley-Terry with per-row human-gap weighting. **1.29M trainable
parameters.**

This is the **recommended RotaSim checkpoint** — the leader on both the preference
mean and the full-reference-benchmark mean.

## Two-axis results (always quote both — the axes are near-independent)

**Preference axis — fgresq / bapps / diffiqa mean** (2AFC accuracy on fixed-seed
cells, our protocol):

| fg (FGResQ sv2) | ba (BAPPS) | di (DiffIQA holdout) | mean-of-3 |
|---|---|---|---|
| .7384 | .8553 | .6663 | **0.7533** @ step 825 |

**Fidelity axis — FR-benchmark mean** ([IQA-PyTorch full-reference
benchmark](https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/FR_benchmark_results.csv))
({CSIQ, LIVE, TID2008, TID2013} × {PLCC, SRCC, KRCC}, raw Pearson, LIVE on the published
`dmos_realigned` labels):

| CSIQ | LIVE | TID2008 | TID2013 | mean |
|---|---|---|---|---|
| .879/.890/.702 | .928/.949/.793 | .752/.720/.538 | .762/.707/.524 | **.7619** (@825) |

Per-dataset cells: `FR_benchmark_results.csv` in this directory. **Convention:** every row,
ours and the references, is scored on the published LIVE convention (`dmos_realigned.mat`),
so these cells are directly comparable to the IQA-PyTorch table (published lpips-vgg mean
~0.7372). The other LIVE variant (`dmos.mat`) is *not* interchangeable — see the repo README.

## Why this checkpoint

Step 825 of a 900-step run. The run's two axes disagree about the best step, which
is itself the interesting part:

* the **preference mean peaks earlier**, at step 650/700 (0.7617), then oscillates
  in a ~.748–.758 band;
* the **FR-benchmark mean peaks late and flat**: .7619@825, **.7627@850**, then
  .7592@875 → .7554@900.

Both curves are a **nested continuation of one run** (consecutive rungs of a single
trajectory), so those differences carry no cross-run term. Step 825 is the released
checkpoint; on the published LIVE labels the FR-mean argmax is step 850, a 0.08pp
difference — inside run noise, so 825 and 850 are effectively tied at the top. The
preference-mean argmax of the same run is step 650.

## Training recipe

FGResQ (60 slots) + BAPPS (60) + DiffIQA (60) per step — Bradley-Terry on 2AFC
triplets with per-row human-gap weighting (`ROTASIM_GAP_WEIGHT`, DiffIQA on the
PNY+SNY plan); warm-started from the DiffIQA di-gap champion `step001300`; all six
linear kinds, block 32, lr 8.979e-5 constant with soft-LR compensation, warmup 30,
fp32 adapter weights with bf16 forward; gold-star optimizer (iterative orthogonal
gradient, Sinkhorn, `snr_cond`); 900 steps, seed 1234. Run:
`runs/20261005-044553-rotasim-dinov3b-fgbadi60-mixgap-scalar-warm1300-blk32-lrfix-seed1234`.

## Use as a loss

Same drop-in contract as the earlier bundles (see `HOWTO.md`):
`dreamsim_oft.as_loss.OFTDreamsimFn`. It is a **quality-preference metric** —
unrelated clean scenes read "close", so ride the reconstruction pair with a
pixel/content term; never use it standalone. **This checkpoint has not been shown
to be a better VAE-training loss** — one measured rung of its own run (step 650)
scored FID 37.33 / sFID 152.33 downstream, worse than the project's best 34.88.
Gate it on your own objective.

## Validation

`python validate_bundle.py .` — loads the checkpoint through the loss API, checks
`d(x,x)==0`, positive finite distances, gradients to **both** arguments, and
per-image shape. PASS on this checkpoint (`VALIDATION_cpu.log`, CPU protocol).

## Caveats

* **Quality preference, not content identity** (see the drop-in warning).
* **Single-run artifact.** Cross-run differences below ~1.5 pp carry a measured
  run-level noise term (~0.6–0.7 pp RMS) and need replication.
* **Training-stream note.** The FGResQ component of this mixture memorises its
  training rows (train accuracy .727 → .852 across ~2.5 epochs while the held-out
  fgresq cell stays flat/drifts down); DiffIQA and BAPPS do not. It has not visibly
  cost either headline axis, but extra FG epochs are spent on rows this metric will
  not be scored on.
