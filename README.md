# RotaSim

**ROtations Trained as Adapters for perceptual SIMilarity.**

*[Written by a human, proofread by AI]*

Hello~!

# What is this

These are models that tell how similar image A is to image B, by telling you a number.

Inspired by Dreamsim. Saw that Dreamsim was training with DinoV1/V2, wanted to upgrade by training a new model with DinoV3 [`rotasim-dinov3b-nights-544-v1`](releases/rotasim-dinov3b-nights-544-v1/).  

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



# Benchmarks
[cleaned-up benchmarks in-progress]

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
misrepresents every model below — always read both.

## Releases

| model | tower | trainable | preference (fgresq/bapps/diffiqa mean) | fidelity ([FR-benchmark mean](https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/FR_benchmark_results.csv)) |
|---|---|---|---|---|
| [`rotasim-dinov3b-fgbadi60-288-v1`](releases/rotasim-dinov3b-fgbadi60-288-v1/) **(recommended)** | DINOv3-B/16 (86M, frozen) | 1.29M | **0.7533** (fg .7384 / ba .8553 / di .6663) | **0.7515** (@825) |
| [`rotasim-dinov3b-diffiqa-v1`](releases/rotasim-dinov3b-diffiqa-v1/) | DINOv3-B/16 (86M, frozen) | 1.29M | 0.7250 (argmax@1300) | 0.7180 (.7248@1250) |
| [`rotasim-qwen38vit-diffiqa-v1`](releases/rotasim-qwen38vit-diffiqa-v1/) | Qwen3.8-27B vision tower (460.7M, frozen, not shipped) | 2.32M | 0.7090 (argmax@2800) | 0.7368 |
| [`rotasim-dinov3b-nights-544-v1`](releases/rotasim-dinov3b-nights-544-v1/) | DINOv3-B/16 (86M, frozen) | 0.43M | 0.6903 | 0.8133 |

For scale: published-protocol lpips-vgg sits at ~0.7372 FR-benchmark mean — the Qwen
model matches it to within 0.0004, and the new fgbadi60 model lands above it, all with
~11× fewer trained parameters than lpips-vgg's head+trunk adaptation. Caveat: our LIVE
cells use our own DMOS labels, so these cross-row gaps are indicative, not exact.

Each release directory is **self-contained**: checkpoint, a vendored
`dreamsim_oft/` package, README with the full two-axis numbers, a HOWTO, and a
`validate_bundle.py` self-test (`python validate_bundle.py <bundle_dir>`).
Start with the bundle READMEs:

- [releases/rotasim-dinov3b-diffiqa-v1/README.md](releases/rotasim-dinov3b-diffiqa-v1/README.md)
- [releases/rotasim-qwen38vit-diffiqa-v1/README.md](releases/rotasim-qwen38vit-diffiqa-v1/README.md)

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
  to non-commercial research — cite that paper); the `*-nights-544-v1` checkpoint
  is **Apache-2.0** (NIGHTS / DreamSim has no non-commercial clause).
- **Frozen towers are not shipped.** DINOv3 (Apache-2.0) resolves
  automatically; the Qwen release points at your own copy of the Qwen3.8-27B
  checkpoint or its llama.cpp mmproj file (see its README).
