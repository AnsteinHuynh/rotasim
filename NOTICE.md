# NOTICE

RotaSim builds on the following work. This file records provenance and credit;
each item's own license governs its use.

## Methods implemented

- **OFT** — "Orthogonal Finetuning: A General Post-Training Method for Language
  and Vision Models" (Liu et al., NeurIPS 2023, arXiv:2306.07280). The
  block-diagonal orthogonal-rotation adapter and its Cayley parametrisation.
- **OFTv2** — "Controlling Text-to-Image Diffusion by Orthogonal Finetuning"
  (CVPR 2024, arXiv:2312.09266). The Neumann-series approximation of the
  Cayley transform used on the long-input paths.
- **SOFT scaling** — the `2*sqrt(block_size-1)` pre-Cayley divisor that keeps
  learning rates block-size invariant: design by Koratahiu (OneTrainer PR
  #1315). The OFT implementation in this repo is an independent implementation
  written from the papers.
- **DOFT** (optional DoRA-magnitude variant) — design by Koratahiu (OneTrainer
  PR #1335); reimplemented here from the published formula.

## Dependencies

- **adv_optm** (Apache-2.0, Koratahiu) — the SinkSGD_adv optimizer family used
  to train the released checkpoints; its OFT-aware parameter tagging contract
  (`_is_oft` etc.) is mirrored in `dreamsim_oft/tagging.py`. Training-only.

## Training data (checkpoints)

- **DiffIQA** (A-FINE, CVPR 2025; github.com/ChrisDud0257/AFINE) —
  non-commercial, research-only dataset terms, including for derived data;
  hence the CC-BY-NC-4.0 license on the released adapter weights. Please cite
  the A-FINE paper when using the checkpoints.

## Frozen towers (not shipped)

- **DINOv3-B/16** (Meta AI, Apache-2.0) — resolved at runtime.
- **Qwen3.8-27B vision tower** (Qwen, Apache-2.0) — user-supplied via
  `qwen_tower_path` / `QWEN_TOWER_PATH` (HF checkpoint directory or llama.cpp
  mmproj gguf; the two loaders are gated bit-identical).
