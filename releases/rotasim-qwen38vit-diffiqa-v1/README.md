# rotasim-qwen38vit-diffiqa-v1

*[Written by AI]*

**RotaSim** — ROtations Trained as Adapters for perceptual SIMilarity.
A perceptual similarity metric: the **frozen vision tower of Qwen3.8-27B**
(460.7M, ViT, 27 blocks, read at its merged 5120-d pooled output) plus **OFTv2
orthogonal rotation adapters** on qkv/proj/fc1/fc2 (108 Linears, block 32).
**2.32M trainable parameters.** The 27B LLM is not involved — vision tower only.

This is the **FR-axis challenger**: it loses the preference panel to
rotasim-dinov3b-diffiqa-v1 but BEATS it on synthetic-FR correlation. The two axes
are near-independent; quote both or quote neither.

## Two-axis results

**S2 preference panel** (2AFC accuracy, fixed-seed cells):

| fg | ba | di | mean-of-3 |
|---|---|---|---|
| .7360 | .8266 | .5646 | **0.7090** @ step 2800 (ladder argmax) |

Ladder: frozen 0.6911 / @400 0.6971 / @1600 0.7047 / @2400 0.7086 / **@2800 0.7090** /
@3200 0.7074 / @3900 0.7011 (decline confirmed past the argmax).

**GM12 — synthetic-FR grand mean** (PLCC/SRCC/KRCC x 4 datasets, our protocol):

| CSIQ | LIVE | TID2008 | TID2013 | **GM12** |
|---|---|---|---|---|
| .745/.922/.759 | .585/.920/.747 | .693/.818/.627 | .661/.779/.585 | **0.7368** |

That GM12 is ABOVE the DINO champion's 0.7248 and 0.0004 below published-protocol
lpips-vgg — and step 2800 was selected on the COMPOSITE, so the FR win is not
cherry-picked. Read: **rank-strong (SRCC/KRCC lead everywhere), linear-weak (PLCC
collapses on LIVE)** — it orders distortions correctly but its distance scale is
not linearly monotone in severity.

## Training recipe

Same corpus/weighting as rotasim-dinov3b-diffiqa-v1 (DiffIQA PNY+SNY, vote-gap
weighted BT), at 20 slots/step; 2,800 steps = 56k triplets (the composite peaked
here and declined through the 78k-triplet parity point). fp32, lr 1.2911e-4
constant, seed 1234, gold-star optimizer, packed-sequence forwards (per-image
attention via cu_seqlens). Run:
`runs/20261004-003539-rotasim-diffiqa20qwen-native512-blk32-qkvprojfc12-seed1234`.

## The frozen tower — YOU provide it (not in this release)

The checkpoint carries only adapters + config (20 MB). Point the loader at either
source via the model-config key `qwen_tower_path` or the env var `QWEN_TOWER_PATH`:

```json
{"qwen_tower_path": "C:/path/to/mmproj-Qwen3.8-27B-bf16.gguf"}
```

* a **file** = the llama.cpp mmproj gguf (strict name-mapped, 333/333 tensors);
* a **directory** = the HF-format `Qwen/Qwen3.8-27B` checkpoint (the 333
  `model.visual.*` tensors are extracted from the safetensors shards; no gguf
  dependency, standard `transformers` + `safetensors` only).

The two sources are verified **bit-identical** (333/333 tensors, max diff 0.0).

## Caveats

* Quality-preference metric: pair with a pixel/content term (see HOWTO.md).
* fp32 forward of a 460M tower is heavy; budget ~20x the DINO checkpoint's eval cost.
* Score mixed-size batches per-image (set `QWEN_FWD_CHUNK=1`); non-%32 inputs are
  resampled up to the next multiple of 32 exactly like the official Qwen processor.
* Single-draw cross-run numbers: the +1.2pp GM12 lead over the DINO champion is
  unreplicated (repo-measured run-level noise ~0.7pp).

## Validation

`python validate_bundle.py` — PASS on this checkpoint (2026-10-04, GPU protocol).
