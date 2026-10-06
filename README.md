# RotaSim

**ROtations Trained as Adapters for perceptual SIMilarity.**

*[Written by a human, proofread by AI]*

Hello~!

# What is this

These are models that tell how similar image A is to image B, by telling you a number.

Inspired by Dreamsim. Saw that Dreamsim was training with DinoV1/V2, wanted to upgrade by training a new model with DinoV3 [`rotasim-dinov3b-nights-v1`](releases/rotasim-dinov3b-nights-v1/).  

# How

You take a vision model, and then you train it with pictures: you give it three pictures; one is the reference, and one is the 'good' one and the other is the 'bad' one. What's good or bad is set by what real life humans voted for. 

## How special is this model?

You're unlocking the hidden potential of any vision models! 

So normally, when vision models are trained to do a task, they specialize in one area, and are meant to be used for what they are designed for.

For Dino's case, it looks at a picture and describes what's in it with numbers that captures its understanding of what the picture shows

For Qwen3.8's case, it's meant to provide tokens embedding, or 'words' representations for the LLM to see pictures
These models can tell a car from a bike, but have no opinion about whether a photo looks good or bad.

So you train it with a reference, a good, and a bad.

## The way of Training and Rotations
When you train these models, you're either finetuning it (changing the weights of the entire model), or adding a learned Lora (a model on top of a model)

I'm sure you've seen some ai generated pictures that look 'burnt' or over saturated.

Why?

It's the AI trying to use a hammer on everything.

Regular loras add information about the training to the base model
if you add too much information, the model basically forgets itself. In other words, it starts forgetting whatever the original information was taught to it, so for example, forgetting what a bike is, and only remembering bikes it saw during training. If it sees a motorcycle, it would think it's a bike instead of a motorcycle. 
Finetuning has this same problem - it starts forgetting.


The fix is OFT LoRA with orthogonal rotations. It's an adapter like LoRA which preserves the geometry of the base model. 

So think of a scrambled rubiks cube (the base model). You want a perfectly solved rubiks cube (The thing you're trying to train for)
A regular lora / fine tune could learn to: rotate the cube, paint the cube, put different colored stickers on the cube, add extra cubes, remove cubes.

But with orthogonal rotations? It only rotates! A rotation adds nothing extra and can always be undone.
So the adapter model is like a set of instructions to tell how the model rotates to achieve the perfect cube.

So really, all you're training is the instructions list. For this project these models are really tiny, because they're rotational instructions on how to turn a base model into a model that works for you.

# Methods
These were trained using ALL the latest SOTA methods I could think of. I dug through the community's best tricks, I tested, tested, and tested again which terms added or helped.
Some of the methods that went into this model I could think of off the top of my head:

OFTv2, Orthogonal Finetuning, the adapter method this project is built on. I was inspired mostly by how good these loras are from seeing initial results from my own sd1.5 and sdxl training from kohya-ss https://github.com/bmaltais/kohya_ss and OneTrainer https://github.com/Nerogar/OneTrainer and also seeing how oneTrainer's community also highly praised OFT. These also have papers backing it (OFT, NeurIPS 2023; OFTv2, CVPR 2024)

SinkSGD_ADV, Sinkhorn optimizer, an advanced optimizer from Koratahiu https://github.com/Koratahiu/Advanced_Optimizers

Diffiq, an advanced dataset, the dataset is legitimately 'hard' https://github.com/ChrisDud0257/AFINE

## Pre-trained models
DinoV3 ViT-B/16, a discriminator model, in my opinion, an advanced form of YOLO but at the modern end of 2025.

qwen3.8, the vision tower, 2026 grade trained model - needs no introduction of how amazing qwen38 is for locally run models

## Optimizations
[in-progress]

A lot of optimizations had to be done while still keeping the training excellent, including things you normally don't consider 'training'

The evaluations. I estimate I still spend 15-30% of the project's time on evaluating and validating checkpoints and checkpoints in between

The caching of images. Cuts down gpu-cpu level re-computations of stuff you've already calculated

The cache for the cache



# Releases and Benchmarks

*For all benchmark numbers: higher is better.*

| Model | Tower | Base/Trained | Dataset | Our benchmark mean ([FGResQ](https://github.com/sxfly99/FGResQ)/[BAPPS](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/tree/main)/[DiffIQA](https://github.com/ChrisDud0257/AFINE)) | [FR-benchmark mean](https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/FR_benchmark_results.csv) | TID2013 (PLCC/SRCC/KRCC) | NIGHTS (Val/Test) |
|---|---|---|---|---|---|---|---|
| [fgbadi60](releases/rotasim-dinov3b-fgbadi60-v1/) **(recommended)** | [DINOv3-B/16](https://github.com/facebookresearch/dinov3) | 86M/1.29M | FGResQ + BAPPS + DiffIQA | **0.7533** (.7384/.8553/.6663) | **0.7619** | .762/.707/.524 | — |
| [diffiqa](releases/rotasim-dinov3b-diffiqa-v1/) | [DINOv3-B/16](https://github.com/facebookresearch/dinov3) | 86M/1.29M | DiffIQA | 0.7250 (.7061/.7919/.6770) | 0.7247 | .712/.688/.508 | — |
| [qwen-diffiqa](releases/rotasim-qwen38vit-diffiqa-v1/) | Qwen3.8-27B-mmproj | 460.7M/2.32M | DiffIQA | 0.7090 (.7360/.8266/.5646) | 0.7505 | .661/.779/.585 | — |
| [nights](releases/rotasim-dinov3b-nights-v1/) | [DINOv3-B/16](https://github.com/facebookresearch/dinov3) | 86M/0.43M | NIGHTS | 0.6903 (.7395/.8577/.4737) | 0.7961 | .765/.824/.628 | 94.59/94.79 |
| [3tower-nights](releases/rotasim-3tower-nights-v1/) | [DINOv3-B/16](https://github.com/facebookresearch/dinov3) + [SigLIP2-base/16](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) + [MetaCLIP2-B/16](https://huggingface.co/docs/transformers/en/model_doc/metaclip_2) | 266M/1.29M | NIGHTS | — | 0.4993 | .479/.526/.366 | 95.64/95.61 |
| [pyiqa](https://github.com/chaofengc/IQA-PyTorch) topiq | — | 23.5M/12.5M | KADID-10K | 0.6755 (.7503/.7943/.4821) | 0.8929 | .916/.917/.744 | — |
| pyiqa ssim | — | 0/0 | — | 0.6277 (.7300/.6962/.4569) | 0.6762 | .656/.627/.455 | — |
| pyiqa ms-ssim | — | 0/0 | — | 0.6241 (.7216/.6890/.4617) | 0.7848 | .782/.786/.605 | — |
| pyiqa l1 | — | 0/0 | — | 0.6149 (.7097/.7010/.4342) | 0.4967 | .423/.488/.347 | — |
| pyiqa psnr | — | 0/0 | — | 0.6122 (.7216/.6842/.4306) | 0.6513 | .660/.687/.496 | — |
| pyiqa lpips | — | 2.47M/1.2K | BAPPS | 0.6744 (.7037/.8289/.4904) | 0.7571 | .753/.744/.548 | — |
| [dreamsim](https://github.com/ssundaram21/dreamsim) (ensemble) | DINOv2-B/16 + CLIP-B/16 + OpenCLIP-B/16 | 264M/1.77M | NIGHTS | 0.6748 (.7133/.8325/.4785) | 0.7720 | .746/.813/.615 | 96.9/96.2 |
| dreamsim (dino_vitb16) | DINOv2-B/16 | 92.6M/0.59M | NIGHTS | 0.6919 (.7551/.8349/.4856) | 0.7715 | .712/.832/.631 | 95.6/94.8 |

Each row is that release's selected checkpoint (argmax on its own selection axis). The
Qwen tower is **not shipped** — `-mmproj` is the loader variant; point it at your own copy.
Benchmark corpora are linked in the column header; the nights models are trained on
[NIGHTS](https://github.com/ssundaram21/dreamsim/tree/main/dataset). `3tower-nights` is a
legacy 224-square artifact (closed class), shown for provenance. The two `dreamsim` rows are
its published NIGHTS numbers; all other cells are measured by us under one protocol.

For scale: the Qwen model's FR mean (0.7505) sits above published-protocol lpips-vgg
(~0.7372), and the fgbadi60 model (0.7619) above both — all with ~11× fewer trained
parameters than lpips-vgg's head+trunk adaptation.

**LIVE labels — every row is on the published convention.** pyiqa's LIVE dataset ships two
DMOS files, and the published IQA-PyTorch table used `dmos_realigned.mat` (`dmos_new`), not
the `dmos.mat` that pyiqa's own `scripts/process_live.py` builds. Every cell here — our
checkpoints *and* the baseline rows — is scored against that same realigned ground truth, so
the pyiqa baseline rows now reproduce the published table **exactly** (topiq_fr 0.8929, to the
digit). This only moves the LIVE cells (and the FR mean that contains them); CSIQ, TID2008 and
TID2013 are unaffected. Do not splice in a number scored against the other LIVE variant — the
two are **not** comparable on LIVE or on the FR mean.

Each release directory is **self-contained**: checkpoint, a vendored
`dreamsim_oft/` package, README with the full two-axis numbers, a HOWTO, and a
`validate_bundle.py` self-test (`python validate_bundle.py <bundle_dir>`).
Start with the bundle READMEs:

- [releases/rotasim-dinov3b-fgbadi60-v1/README.md](releases/rotasim-dinov3b-fgbadi60-v1/README.md)
- [releases/rotasim-dinov3b-diffiqa-v1/README.md](releases/rotasim-dinov3b-diffiqa-v1/README.md)
- [releases/rotasim-qwen38vit-diffiqa-v1/README.md](releases/rotasim-qwen38vit-diffiqa-v1/README.md)
- [releases/rotasim-dinov3b-nights-v1/README.md](releases/rotasim-dinov3b-nights-v1/README.md)
- [releases/rotasim-3tower-nights-v1/README.md](releases/rotasim-3tower-nights-v1/README.md)

# Technicals
[in-progress]

# Surprises and 'gotchas' during this project
[in-progress]



# RotaSim - recap by AI

**ROtations Trained as Adapters for perceptual SIMilarity.**

*[Written and summarized by AI]*

RotaSim metrics are perceptual similarity models built from a **frozen vision
tower + OFTv2 orthogonal rotation adapters**: instead of finetuning a backbone,
each adapted linear layer learns a block-diagonal orthogonal rotation of its
input (a Cayley transform of a skew-symmetric parameter), leaving the pretrained
weights untouched. The released artifacts are tiny — **428K–2.3M trainable
parameters** on top of towers the user already has.

The core empirical finding of the project: **the two axes of perceptual-metric
quality are near-independent.**

- **Preference axis** — 2AFC accuracy on human preference panels: the mean of our
  three holdout cells, **fgresq / bapps / diffiqa**.
- **Fidelity axis** — correlation with human DMOS on the [IQA-PyTorch full-reference
  benchmark](https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/FR_benchmark_results.csv):
  the mean of its 12 cells ({CSIQ, LIVE, TID2008, TID2013} × {PLCC, SRCC, KRCC}).

Training for one does not buy the other. Quoting any single-axis number alone
misrepresents every model in the table — always read both.

## Quickstart (as a differentiable loss)

```python
from dreamsim_oft.as_loss import OFTDreamsimFn

fn = OFTDreamsimFn(device="cuda", ckpt="releases/rotasim-dinov3b-diffiqa-v1/step001300.pt")
d = fn(x, y)   # (N,) perceptual distances in [0, 2]; differentiable w.r.t. both args
```

Images are float tensors in `[-1, 1]`, any aspect ratio (centre-square crop at
the trained resolution). See the bundle HOWTOs for details, including the Qwen
tower resolution (`qwen_tower_path` config key or `QWEN_TOWER_PATH` env).

**Important:** these are **quality-preference metrics**. Unrelated clean scenes
can read *closer* than a clean-vs-degraded pair of one scene, so as a training
loss the distance must ride the reconstruction pair together with a
pixel/content term (both bundle READMEs carry this caveat).

## Honest-caveats corner

- The headline numbers are **single training draws**; measured run-level noise
  on these protocols is ~0.7pp. Differences between models smaller than that
  should not be over-read.
- Panel metrics do not crown a loss. In a downstream VAE-training A/B, the nights
  model did **not** improve decoder FID — its strong FR correlations did not
  transfer — so treat neither axis as a verdict about loss quality.
- Checkpoints are the adapter tensors only; each release's frozen tower is
  supplied by the user (never shipped).

## License & credits

- **Code:** Apache-2.0 (see [LICENSE](LICENSE), [NOTICE.md](NOTICE.md)).
- **Adapter checkpoints:** license depends on the training corpus, and each
  bundle states it in its own `WEIGHTS-LICENSE.md`: the two `*-diffiqa-v1`
  checkpoints are **CC-BY-NC-4.0** (DiffIQA / A-FINE terms restrict derived data
  to non-commercial research — cite that paper); the `*-nights-v1` checkpoint
  is **Apache-2.0** (NIGHTS / DreamSim has no non-commercial clause).
- **Frozen towers are not shipped.** DINOv3 (Apache-2.0) resolves
  automatically; the Qwen release points at your own copy of the Qwen3.8-27B
  checkpoint or its llama.cpp mmproj file (see its README).
