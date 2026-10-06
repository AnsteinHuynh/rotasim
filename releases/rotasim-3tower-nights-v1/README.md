# rotasim-3tower-nights-v1

*[Written by AI]*

**RotaSim** — ROtations Trained as Adapters for perceptual SIMilarity.
A three-tower NIGHTS metric: **DINOv3-B/16 + SigLIP2-base/16 + MetaCLIP2-B/16**
(266M frozen) with **OFTv2 rotations** (q/k/v, block 32, 108 wrapped Linears) per
tower, concatenated to 2048-d and scored by cosine. **1,285,632 trainable
parameters (0.4833%)**, a 5,062 KB checkpoint.

> **Legacy / provenance.** This is the older 224-square 3-tower class, retired in
> 2026-09 in favour of the 544 aspect-native 1-tower line. It ships because it was
> asked for, and it is the only *trained* 3-tower artifact here. Read it as a
> reproduction artifact, not a recommendation.

## NIGHTS results (its home class)

| | n | accuracy |
|---|---|---|
| NIGHTS **val** | 1,720 | **95.64 %** |
| NIGHTS **test** | 1,824 | **95.61 %** |

Recomputed from this checkpoint, fp32, one protocol; val is also banked in the
checkpoint (`val_acc = 0.9563953488372093`, `val_n 1720`).

**The adapters are what carry it.** The *same three frozen towers*, concatenated and
scored with **no adapters at all**, bank 86.80 % val / 87.01 % test — so the
1.29 M OFT parameters are worth **+8.84 pp val / +8.60 pp test**. Nothing about the
towers changed, which makes this the cleanest "OFT is real" statement in the repo.

Not a SOTA NIGHTS claim: the released DreamSim v0.2.0 **ensemble** scores **96.16 %**
on these same 1,824 test rows (measured), i.e. this artifact is −0.55 pp against a
much larger ensemble.

## Full-reference benchmark

The towers use **fixed position embeddings**, so the native-resolution embedding path
that the other bundles use is impossible here; these cells are the model's own
224-square class (`--own224`). LIVE is on the published `dmos_realigned` labels.
(CSIQ/LIVE/TID2008/TID2013 = PLCC/SRCC/KRCC)

| CSIQ | LIVE | TID2008 | TID2013 | mean |
|---|---|---|---|---|
| .527/.742/.534 | .536/.749/.554 | .408/.372/.253 | .479/.526/.366 | **.4993** |

The weak TID2008/TID2013 columns are the cost of scoring a 224-square
natural-image metric on synthetic-distortion corpora — do not read the .4993 as this
model's quality; the NIGHTS numbers above are its home measurement.

## Preference panel (S2), own 224 geometry

2AFC accuracy on the same preference cells the other releases use (FGResq sv2 837 rows /
BAPPS 836 / DiffIQA 836), scored at **this model's own 224-square** pipeline:

| fg | ba | di | composite |
|---|---|---|---|
| .7252 | .8421 | .5048 | **.6907** |

Caveat: this is a 224-class score. The other rows in the repo README use the 288-class
panel, so the composites are **not directly comparable** — the DiffIQA cell in particular
is scored on a 224 crop of 512px sources and reads low for that reason.

## Use as a loss

Same drop-in contract as the other bundles (see `HOWTO.md`):
`dreamsim_oft.as_loss.OFTDreamsimFn`. Resolves all three towers at runtime. It is a
**quality-preference** metric — ride the reconstruction pair with a pixel/content
term.

## Validation

`python validate_bundle.py .` — identity `d(x,x)==0`, positive finite distances,
gradients to both arguments, per-image shape. PASS on this checkpoint
(`VALIDATION_cpu.log`). The checkpoint's own config builds **108 wrapped Linears
(36 × 3 towers), 1,285,632 trainable parameters**.

## Caveats

* Three towers must resolve at load time (DINOv3 is gated; SigLIP2 / MetaCLIP2
  download from the Hub) — heavier setup than the 1-tower bundles.
* Fixed 224-square input: the full frame is resized to a square, nothing is cropped.
* Single-run artifact; cross-run differences below ~1.5 pp need replication.
