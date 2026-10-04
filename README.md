# RotaSim

**ROtations Trained as Adapters for perceptual SIMilarity.**

RotaSim metrics are perceptual similarity models built from a **frozen vision
tower + OFTv2 orthogonal rotation adapters**: instead of finetuning a backbone,
each adapted linear layer learns a block-diagonal orthogonal rotation of its
input (a Cayley transform of a skew-symmetric parameter), leaving the pretrained
weights untouched. The released artifacts are tiny — **428K–2.3M trainable
parameters** on top of towers the user already has.

The core empirical finding of the project: **the two axes of perceptual-metric
quality are near-independent.**

- **Preference axis** — 2AFC accuracy on human preference panels (our S2 panel
  over FGResQ / BAPPS / DiffIQA holdout cells).
- **Fidelity axis** — correlation with human DMOS on synthetic full-reference
  IQA benchmarks (our GM12 grand mean over {CSIQ, LIVE, TID2008, TID2013} ×
  {PLCC, SRCC, KRCC}).

Training for one does not buy the other. Quoting any single-axis number alone
misrepresents every model below — always read both.

## Releases

| model | tower | trainable | preference (S2 composite) | fidelity (GM12) |
|---|---|---|---|---|
| [`rotasim-dinov3b-diffiqa-v1`](releases/rotasim-dinov3b-diffiqa-v1/) **(recommended)** | DINOv3-B/16 (86M, frozen) | 1.29M | **0.7250** (argmax@1300) | 0.7180 (.7248@1250) |
| [`rotasim-qwen38vit-diffiqa-v1`](releases/rotasim-qwen38vit-diffiqa-v1/) | Qwen3.8-27B vision tower (460.7M, frozen, not shipped) | 2.32M | 0.7090 (argmax@2800) | **0.7368** |
| [`rotasim-dinov3b-nights-544-v1`](releases/rotasim-dinov3b-nights-544-v1/) | DINOv3-B/16 (86M, frozen) | 0.43M | 0.6903 | 0.8133 |

For scale: published-protocol lpips-vgg sits at ~0.7372 GM12 — the Qwen model
reaches within 0.0004 of it; the DINO model within 0.012 with ~11× fewer
trained parameters than lpips-vgg's head+trunk adaptation.

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
