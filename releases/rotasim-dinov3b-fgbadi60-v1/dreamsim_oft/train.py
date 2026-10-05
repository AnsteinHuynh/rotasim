"""
[Begin Work Zone]
Standalone trainer: OFTv2 adapters + adv_optm.SinkSGD_adv on the NIGHTS 2AFC task.

Why not DreamSim's train.py: it needs pytorch_lightning + wandb + torchmetrics, none
of which are in this venv, and its LoRA plumbing targets a fused `qkv` that none of
our three backbones has. Why not OneTrainer: it is diffusers-based and has no concept
of triplet-ranking perceptual-metric training. So we keep the two things that matter
(OFTv2's rotation and adv_optm) and own the loop.

What this writes into a timestamped run folder:
    adapters.pt      the trained oft_R rotations (+ step, config)
    steps.jsonl      one line per logged step: loss, 2AFC acc, grad norm, lr, vram
    config.json      the exact config, for reproducibility
Console output is additionally mirrored to <project>/debug/*.jsonl by logutil.

usage:
    <venv python> src/dreamsim_oft/train.py --synthetic --steps 300          # smoke run
    <venv python> src/dreamsim_oft/train.py --dataset-root I:\\...\\nights --epochs 6
    <venv python> src/dreamsim_oft/train.py --help
[End Work Zone]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .data import SyntheticTwoAFCDataset, TwoAFCDataset, make_loader
from .loss import HingeLoss, build_criterion, two_afc_scored
from .model import PerceptualModel
from .oft import OFTLinear, oft_identity_error, oft_rotation_stats
from .tagging import describe_tags

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    # data
    dataset_root: str | None = None
    # Which loader to build. "nights" = the CSV 2AFC loader (original). "fgresq" =
    # FGResQDataset: REAL photographs, JSON annotations, pairs with 0/0.5/1 labels.
    dataset: str = "nights"
    # mix3 only: roots of the two extra corpora (dataset_root stays the FGResQ root,
    # which eval-side selection also uses). Empty string -> the canonical I: paths.
    bapps_root: str = ""
    diffiqa_root: str = ""
    # FGResQ only: drop scene_id 3/4 (no pristine reference exists there, ~2.1k rows).
    exclude_scene34: bool = True
    # FGResQ train/val carve version. 1 = LEGACY hash-of-row-paths (pool-LEAKY: the
    # ~8% val rows come from pools that also contribute train rows -- the carve the
    # 77.91% keeper was measured on). 2 = pool-disjoint: the hash runs over ref_image
    # alone, so ~8% of POOLS hold the whole val carve and no pool straddles it.
    # DATA-SIDE ONLY (never changes the forward math or any parameter key), but it
    # decides which rows a val number was measured on, so it IS _RESUME_LOCKED: a
    # resume adopts the checkpoint's version to keep its val curve interpretable.
    split_version: int = 1
    # D1 dense pool batching: batch_size counts POOLS; each pool contributes its ref +
    # pool_k candidates through ONE model.embed() forward, and every supervised
    # candidate pair inside the pool becomes one BT logit. Eval stays on the existing
    # pairwise val loader for continuity.
    pool_batch: bool = False
    pool_k: int = 6
    # D2 pool objective. "bt" = the D1 pair-vote Bradley-Terry over the upper triangle
    # (unchanged). "listnet" = ListNet cross-entropy: per-pool target distribution over
    # candidates built from score_norm, compared against log_softmax of the standardized
    # distances. Needs no pair rows (every candidate has score_norm) and no tau sweep.
    pool_loss: str = "bt"             # "bt" | "listnet"
    # ListNet logit temperature. 0.0 = per-pool STANDARDIZATION of the distances
    # (z = (d - mean_valid) / (std_valid + 1e-6)), the D2 default: scale-free, no sweep.
    # >0 = explicit temperature z = -d / listnet_t (kept only for a future ablation).
    listnet_t: float = 0.0
    # Target sharpener: t = softmax(scores / listnet_tau_t over valid candidates), then
    # optionally vote-weighted and renormalized. 1.0 = use score_norm as-is.
    listnet_tau_t: float = 1.0
    # Weight the target distribution by per-candidate vote share (candidate vote count /
    # max vote count in the pool). Candidates appearing in more judged rows are better
    # calibrated, so they get more of the target mass.
    listnet_vote_weight: bool = True
    synthetic: bool = False
    synthetic_n: int = 256
    num_workers: int = 4
    # LEGACY (ignored 2026-09-26): the trainer no longer evals, so there is no val
    # split to pre-decode. Key kept so older run configs (resume via --config-file)
    # still parse. Post-run scoring: scripts/eval_sweep.py.
    preflight_decode: bool = True
    # LEGACY (ignored 2026-09-26): there is no val loader to resize -- the trainer
    # never evals. Key kept for old-config parse compatibility.
    eval_batch_size: int = 0
    # optimisation
    lr: float = 3e-4
    # Cosine anneal FLOOR. 0.0 = anneal to zero (the original behaviour). A non-zero
    # value lets a warm restart settle at a small LR instead of stopping dead.
    lr_final: float = 0.0
    epochs: int = 6
    batch_size: int = 16
    grad_accum: int = 1
    # Objective. 'hinge' is DreamSim's original (margin=0.05, flat region once
    # satisfied); 'bt' is the Bradley-Terry / logistic loss, which keeps a gradient on
    # every pair. See loss.py.
    loss_type: str = "hinge"
    margin: float = 0.05          # hinge only
    bt_tau: float = 0.05          # bt only; FIXED, not learned (SinkSGD discards scale)
    # Vote-confidence weighting (BT only): per-triplet loss weight w = 2*|y-0.5|, so a
    # TIE (y=0.5, FGResQ only) contributes zero loss instead of pushing d0==d1. A no-op
    # on binary-target data (NIGHTS). Does NOT change the forward math, so it is not
    # _RESUME_LOCKED; the TRAIN criterion only -- the (library-only) eval criterion
    # stayed unweighted so a val_loss was never vote-boosted.
    # val_loss remains comparable across runs.
    bt_confidence_weight: bool = False
    # CONFIDENCE-SOFTENED LABELS (2026-10-01): 0.0 = OFF (bit-identical to the old hard
    # 0/1 target). >0 gives the scale of the human |score_normA - score_normB| at which a
    # label is treated as fully decisive; below it the pool-BT target is pulled toward 0.5
    # in proportion, so a near-tie stops pushing a direction the humans barely expressed.
    # Default 0.07 is the MEDIAN human gap on this corpus (median 0.0708, p10 0.0147), so
    # about half the pairs stay fully weighted. Only the D1/BT pool path uses it; the
    # binarised `target` (and therefore every accuracy number and every eval probe) is
    # untouched. NOTE bt_confidence_weight CANNOT do this -- it reads the binarised y.
    soft_gap_ref: float = 0.0
    weight_decay: float = 0.0
    warmup_steps: int = 50
    lr_schedule: str = "cosine"          # cosine | linear | constant
    # On RESUME the schedule is normally measured from the RESUME step (segment-relative),
    # so warmup and the decay restart from there. That is right for a deliberate warm
    # restart, but WRONG for merely continuing an interrupted run: the LR would fall to the
    # floor and climb back to peak, silently changing the recipe. Setting this True measures
    # the schedule from step 0 instead, so a resumed run continues the ORIGINAL curve
    # exactly. (Verified: at step 2600 of a 5214-step linear run both give 1.548e-4 when
    # this is on, vs a drop to ~5.7e-6 and a 100-step re-warmup when it is off.)
    lr_absolute_schedule: bool = False
    grad_clip: float | None = None
    seed: int = 1234
    # ---- adapter family -----------------------------------------------------
    # 'oft'  = Orthogonal Finetuning v2 (a rotation; norm-preserving, cannot rescale)
    # 'dora' = weight-Decomposed low-rank Adaptation (learns magnitude AND direction)
    # The DoRA arm exists to test whether OFT's structural inability to rescale or
    # suppress a direction is what limits accuracy on this task.
    adapter_type: str = "oft"
    dora_r: int = 16
    dora_alpha: float = 8.0
    dora_dropout: float = 0.3
    dora_targets: str = "q_proj,k_proj,v_proj"

    # model
    backbones: tuple = ("dinov3_vitb16", "siglip2_base16", "metaclip2_b16")
    block_size: int = 32
    block_share: bool = False
    # OFT adapter surface: comma-separated Linear-name suffixes (parsed like dora_targets).
    # "q_proj,k_proj,v_proj" = the keeper's reduced surface;
    # "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj" = the faithful OFT-paper surface
    # (attention + MLP; 72 tensors / 2.61M params at block_size 64 on DINOv3-B).
    # FORWARD-AFFECTING -> mirrored in model.model_kwargs_from_config.
    oft_targets: str = "q_proj,k_proj,v_proj"
    oft_scaled: bool = False
    oft_dropout: float = 0.0
    # DOFT magnitude LR (adapter_type="doft" only). 0.0 = same as the base lr. The
    # magnitude's update under SinkSGD is a pure sign step bounded by its own lr, so the
    # reach it can travel is |log m| = steps * lr_mag EXACTLY -- the dose is set by this
    # number, not by the gradient. At the base lr over 300 steps that is only +-3.9%.
    doft_mag_lr: float = 0.0
    # With oft_scaled on and spectral_normalization off, nothing in adv_optm compensates
    # the SOFT divisor, so the effective rotation LR silently drops ~11x. This restores
    # SOFT's DOCUMENTED behaviour (effective LR consistent across block sizes) by
    # scaling the OFT param group's LR by 2*sqrt(block_size-1). Set false to instead get
    # the ~11x gentler rotation at the nominal LR.
    soft_lr_compensation: bool = True
    # OFT adapter weight precision. fp32 is the recommended setting: the Cayley /
    # Neumann construction is numerically delicate, and the rotations are only ~1.3M
    # params so keeping them in fp32 costs essentially nothing.
    oft_weight_dtype: str = "float32"
    # Exact Cayley solve (I-Q)(I+Q)^-1 instead of the truncated 5-term Neumann series.
    # WHY: the Neumann series is a Taylor expansion valid only for ||Q|| < 1, and the
    # Sinkhorn-normalised updates push ||Q|| far past that -- measured, the reported
    # raw drift reached 1264 where Cayley is mathematically bounded by 2, i.e. the
    # series had diverged, and ||R^T R - I|| climbed to 0.156 (these were no longer
    # rotations). The exact solve is orthogonal for ANY ||Q|| and costs one 32x32
    # solve per layer (~108 per step), which is negligible.
    # NOTE: the exact branch parametrises R as the transpose of the Neumann branch
    # (a reparametrisation, not a bug). Checkpoints are NOT portable across this flag.
    use_cayley_neumann: bool = False
    normalize_embeds: bool = True
    # L2-normalise EACH branch BEFORE concatenating (model.py PerceptualModel.embed).
    # WHY IT EXISTS: concatenating raw branch features and taking ONE cosine weights the
    # branches by their share of the SQUARED norm, not by merit. Measured on NIGHTS, the
    # mis-weighted concat gave MetaCLIP2 -- the WEAKEST branch -- 71.1% of the voice and
    # the best branch (SigLIP2) only 10.0%.
    # MEASURED on the frozen model, PAIRED (McNemar) over val+test with scripts/eval_baseline.py
    # --dump-preds + scratch/sophie/mcnemar.py: +1.35pp, 95% CI [+0.57, +2.14], p=0.0009.
    # NOTE this is NOT the same as normalize_embeds: that mean-centres and L2-normalises the
    # WHOLE 2048-d vector AFTER the concat, which is exactly what fails to balance branches.
    # FORWARD-AFFECTING with no change to any parameter key or shape, so it is in
    # model_kwargs_from_config AND _RESUME_LOCKED. Default False keeps old checkpoints exact.
    per_branch_norm: bool = False
    # ---- MULTI-CHANNEL READOUT (2026-10-02) -----------------------------------
    # WHICH transformer layer outputs feed the distance, e.g. (4, 8, 12) = three
    # per-layer cosines combined by LEARNED convex weights; the TOTAL is what the loss
    # sees (scalar, unchanged interface) and model.distance_channels() exposes the
    # per-channel vector. Empty tuple = the released single-readout head (exact legacy
    # behaviour). FORWARD-AFFECTING in the strongest sense -> model_kwargs_from_config
    # AND _RESUME_LOCKED. Single CLS tower (DINOv3) only; PerceptualModel validates.
    readout_layers: tuple = ()
    dtype: str = "float32"
    # ---- input resolution ----------------------------------------------------
    # The resolution the towers are fed. NIGHTS is 768x768 natively, so 224 is a 3.4x
    # downsample and 544 keeps far more of the real detail.
    # FORWARD-AFFECTING IN THE STRONGEST SENSE: it fixes the patch grid, and SigLIP2 /
    # MetaCLIP2 carry LEARNED position tables sized for exactly one grid. At 644 they do not
    # merely degrade, they raise; backbones.resize_position_embeddings resamples those tables
    # bicubically. DINOv3 needs none of that (RoPE) and accepts any size.
    # NOTE: 544 makes those two towers run at a resolution they were NEVER trained at, so
    # their features are off-distribution. In model_kwargs_from_config AND _RESUME_LOCKED.
    image_size: int = 224
    # Mixed precision. "off" | "bf16". bf16 needs no GradScaler (unlike fp16: its exponent
    # range matches fp32). The OFT rotation explicitly DISABLES autocast internally and runs
    # fp32 regardless -- see OFTRotationModule.forward; the Cayley solve is numerically
    # delicate and an earlier Neumann variant diverged outright.
    amp: str = "off"
    # ---- the 544 random-aspect class ------------------------------------------
    # ON: each NIGHTS triplet gets ONE (h,w) from ASPECT_BUCKETS_544 (h*w ~= 544^2,
    # dims %16, aspect 1:2..2:1) plus ONE crop rect from its 768x768 source, shared by
    # ref/left/right (the triplet's meaning depends on the shared view). Batches are
    # grouped by bucket (AspectBatchSampler). Val geometry is FROZEN per index so
    # in-run evals are comparable. OFF: the legacy square resize path, bit-compatible
    # with every 224-class run.
    random_aspect: bool = False
    # Page-lock loader batches for overlapped H2D copies (see data.make_loader). Flag
    # exists so it can be A/B'd end to end in pixels/second.
    pin_memory: bool = True
    # WDDM stall mitigation (forensic report 2026-09-26, VRAM_WEDGE_TIMELINE.md): after an
    # eval the prefetched queue is torn down and re-primed; with 8 workers x prefetch 2
    # that is 16 batches of host commit, and the producer side then wedged ~30 s/batch for
    # the rest of the run (65% of wall time on the 09:31 run). persistent_workers keeps
    # the workers alive across epochs; prefetch_factor 1 keeps the queue shallow.
    # DEFAULTS PRESERVE LEGACY BEHAVIOUR (False / None); these change only the input
    # pipeline, never the forward math or which rows are used.
    persistent_workers: bool = False
    prefetch_factor: int = 0     # 0 = legacy (torch default 2 when workers>0)
    # Dump the first N images exactly as the trainer feeds them to the model (post-crop,
    # post-jitter, [0,1], the literal input tensors) into <run_dir>/preview/ as PNGs.
    # Per-tower normalization happens INSIDE each Branch, so this IS "what the trainer
    # sees". 0 disables; default 10.
    preview_images: int = 10
    # Per-image color augmentation, +/- this fraction on brightness/contrast/saturation
    # (multiplicative) and hue (additive on the [-0.5,0.5] scale). Applied independently
    # to each image of a triplet INCLUDING the reference. 0 disables.
    color_jitter: float = 0.0
    # Horizontal-flip probability for a WHOLE triplet/pool, TRAIN split only (kerok directive
    # 2026-10-01 morning): a perceptual similarity metric should be flip-symmetric and the
    # corpora ship no mirrored examples. All images of a sample flip together (flipping a
    # subset would corrupt the 2AFC label), the draw is deterministic in (seed, epoch) like
    # the geometry plan, and the evaluator never flips. 0.0 = disabled, bit-identical to the
    # pre-flip pipeline. See dreamsim_oft.data._hflip_sample.
    hflip: float = 0.0
    # Gradient checkpointing on the towers (~30% compute for a large activation-memory
    # cut; measured 0.4 GiB/triplet at the 544 budget, so batch 16 lands ~7.5 GiB).
    # Towers must then run in train() mode -- safe, dropout surface verified zero.
    gradient_checkpointing: bool = False
    # DINOv3's train-mode stochastic RoPE coordinate augmentation (pos_embed_rescale).
    # True = keep it ON (positional regularizer, user's choice for the 544 class; the
    # towers are then stochastic during TRAINING ONLY -- eval mode is always clean).
    # False = force off for bit-determinism. Default False = the 224-class behaviour.
    # FORWARD-AFFECTING (stochastically) -> in _RESUME_LOCKED.
    dinov3_rope_augment: bool = False
    # SigLIP: use the tower's built-in per-batch bicubic position-table interpolation
    # instead of pre-resizing to one square grid. Bit-identical at 224; required for
    # non-square batches. DINOv3 needs nothing (RoPE). Default False for 224 compat.
    interpolate_pos: bool = False
    # SigLIP, supersedes interpolate_pos: feed the image at its TRUE aspect and serve
    # the position table per grid from a lazily-baked cache (bicubic resample ONCE per
    # grid, deterministic). No runtime interpolation flag. Mutually exclusive with
    # interpolate_pos. FORWARD-AFFECTING -> model_kwargs_from_config + _RESUME_LOCKED.
    baked_pos: bool = False
    # Qwen3.8 vision tower pooling: "merged" (5120-d merger output) or "premerge"
    # (1152-d last_hidden_state). FORWARD-AFFECTING (the embedding the metric compares)
    # with identical adapter parameter names, so it is in model_kwargs_from_config AND
    # _RESUME_LOCKED. Inert for every other backbone kind.
    qwen_pool: str = "merged"
    # SinkSGD_adv -- names/defaults mirror the installed adv_optm signature
    momentum: float = 0.0
    nesterov: bool = False
    nesterov_coef: float | None = None
    normed_momentum: bool = False
    snr_cond: bool = False
    orthogonal_sinkhorn: bool = False
    sinkhorn_iterations: int = 5
    cautious_wd: bool = False
    geometric_wd: bool = False
    centered_wd: float = 0.0
    centered_wd_mode: str = "float8"
    spectral_normalization: bool = False
    orthogonal_gradient: str = "disabled"   # disabled | flattened | iterative
    state_precision: str = "auto"
    stochastic_rounding: bool = True
    nnmf_factor: bool = False
    vector_reshape: bool = False
    compiled_optimizer: bool = False
    # bookkeeping
    log_every: int = 10
    # LEGACY EVAL KEYS (all ignored 2026-09-26, kerok directive after probe A):
    # NO evals happen during training -- not mid-run, not at the end. The one
    # in-process eval left in probe A's lifecycle (the final full-split eval) was the
    # one that wedged it, after two clean subprocess evals; best.pt carried no step
    # in its name; and mid-run candles were never decision-grade. Protocol now:
    #   train -> save versioned ckpts -> finish/stop -> scripts/eval_sweep.py
    # which scores every checkpoint under ONE protocol (fp32, geometry seed 1235)
    # and prints a {step, val_acc, n} table; best step = a row in that table.
    # Fields kept so older run configs (--config-file on resume) still parse.
    eval_every: int = 0
    eval_batches: int = 0
    eval_subprocess: bool = False
    # A 4-hour run must survive being paused. `save_every` writes a resumable
    # checkpoint.pt (adapters + optimizer state + position); `resume` picks it back up.
    # Without this, killing the process loses everything, which is exactly what happened
    # on the first real run.
    save_every: int = 50
    # EVERY save_every checkpoint is also copied to <run_dir>/ckpt/stepNNNNNN.pt with
    # the step number in its NAME, so nothing is ever overwritten. Rationale: the
    # 224-class run at step 1150/1100 could no longer be tested after checkpoint.pt was
    # replaced -- a mid-run val curve needs the checkpoints that earned it.
    # checkpoint.pt itself keeps being rewritten (it is what --resume expects).
    versioned_checkpoints: bool = True
    # PAUSE FLAG (kerok 2026-09-26, sister-project convention). Drop a file with this
    # name (default "PAUSE") into the project root or the launch directory and the trainer
    # notices at the END of the current step, writes checkpoint.pt + adapters.pt + a
    # named ckpt/stepNNNNNN.pt, retires the flag to "<name>.done" and exits 0 -- instead
    # of having to wait for the next save_every boundary or hard-killing the process.
    # Checked every step for free (one Path.exists()); set to "" to disable.
    pause_flag: str = "PAUSE"
    resume: str | None = None
    # Data stream on resume. DEFAULT (False): the loop fast-forwards through the batches the
    # previous segment already consumed, so training continues exactly where it left off.
    # True: RE-SHUFFLE and start the stream at batch 0, so the resumed segment trains on a
    # fresh random draw from the whole training set rather than the tail of the previous
    # ordering. This is the right choice for a warm restart, where the extra steps should
    # sample broadly.
    # CAVEAT worth stating: this deliberately re-shows data the model has already trained on.
    # After epoch 0 every triplet has been seen at least once, so "fresh sample" can only
    # ever mean "drawn anew", never "never seen" -- there is no unseen data left to give it.
    # NOT in _RESUME_LOCKED: changing it on resume is the entire point.
    resume_reshuffle: bool = False
    # Seed for that fresh permutation. None -> cfg.seed + 7919, a fixed offset chosen so it
    # collides with neither the per-epoch seeds (cfg.seed + 0, +1, ...) nor cfg.seed itself.
    reshuffle_seed: int | None = None
    max_steps: int = 0
    run_name: str = "train"
    tag: str = ""
    extra: dict = field(default_factory=dict)


def make_run_dir(cfg: TrainConfig) -> Path:
    root = PROJECT_ROOT / "runs"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{cfg.run_name}"
    if cfg.tag:
        name += f"-{cfg.tag}"
    d = root / name
    n = 1
    while d.exists():
        d = root / f"{name}-{n}"
        n += 1
    d.mkdir(parents=True, exist_ok=False)
    return d


def soft_lr_scale(model: PerceptualModel) -> float:
    """The factor that makes Scaled OFT (SOFT) a true REPARAMETERISATION.

    OneTrainer's own tooltip for SOFT says it exists so that "the effective learning rate
    remains consistent across different block sizes". SOFT divides the skew weights by
    s = 2*sqrt(block_size-1) before the Cayley transform, so at a fixed LR the applied
    rotation becomes ~s times smaller -- measured 11.06x at block_size 32.

    OneTrainer records s as `_oft_scale_factor` for adv_optm to undo, BUT that value is
    read in exactly ONE place (scaled_optm.py:183) which is reachable only when
    spectral_normalization=True. With spectral off -- our config -- nothing compensates
    and SOFT silently reduces the effective rotation LR by ~11x, the OPPOSITE of its
    documented purpose.

    This returns s so the OFT param group's LR can be multiplied by it, restoring the
    documented invariant without switching on spectral normalisation.
    """
    sizes = {m.oft_R.block_size for m in model.modules() if isinstance(m, OFTLinear)}
    if not sizes:
        return 1.0
    if len(sizes) > 1:
        print(f"[soft] WARNING: mixed OFT block sizes {sorted(sizes)}; using the smallest "
              f"for the LR compensation. This is approximate.")
    b = min(sizes)
    if b <= 1:
        return 1.0
    return 2.0 * math.sqrt(b - 1)


def build_optimizer(model: PerceptualModel, cfg: TrainConfig):
    from adv_optm import SinkSGD_adv

    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("no trainable parameters to optimize")

    # ---- PARAM-GROUP SPLIT (2026-10-01) -------------------------------------
    # `oft_lr_scale` is OUR key (applied below as g["lr"] = lr * g.get("oft_lr_scale", 1.0));
    # adv_optm ignores unknown group keys, and the key is PER-GROUP. It used to sit on the
    # single group holding every trainable tensor, so on any model whose module carries more
    # than one adapter family EVERY family would be scaled by 2*sqrt(block_size-1) --
    # measured on a combined OFT+DoRA+LoRA module: oft_R 5.2915, dora magnitude 5.2916,
    # lora_A 5.2915, lora_B 5.2915. Harmless while adapter families are mutually exclusive
    # (model.py's if/else) and silently corrupting the moment they are not -- DoRA's
    # magnitude, whose step under SinkSGD is a pure +/-lr sign step, would move 7.75x too
    # fast at block_size 16. Give the rotations their OWN group so the scale stays local.
    name_of = {id(p): n for n, p in model.named_parameters()}
    oft_params, other_params = [], []
    for p in params:
        (oft_params if name_of.get(id(p), "").endswith("oft_R.weight")
         else other_params).append(p)

    groups: list[dict] = []
    oft_group: dict | None = None
    if oft_params:
        oft_group = {"params": oft_params}
        groups.append(oft_group)
    if other_params:
        groups.append({"params": other_params})
    if not groups:
        raise RuntimeError("no trainable parameters to optimize")

    scale = 1.0
    if (oft_group is not None and cfg.adapter_type in ("oft", "doft")
            and cfg.oft_scaled and cfg.soft_lr_compensation):
        scale = soft_lr_scale(model)
        oft_group["oft_lr_scale"] = scale
        print(f"[soft] Scaled OFT is ON with spectral_normalization="
              f"{cfg.spectral_normalization}. Nothing in adv_optm compensates on that "
              f"path, so the OFT param group LR is multiplied by {scale:.4f} "
              f"(2*sqrt(block_size-1)) to keep the effective rotation rate consistent, "
              f"which is SOFT's documented purpose. Set soft_lr_compensation=false to get "
              f"the ~{scale:.1f}x gentler rotation instead.")
        print(f"[soft] param groups: {len(groups)} -- oft_lr_scale={scale:.4f} applies to "
              f"{len(oft_params)} rotation tensor(s) ONLY"
              + (f"; {len(other_params)} other adapter tensor(s) keep lr={cfg.lr:g}"
                 if other_params else ""))

    # DOFT's magnitude group gets its OWN LR (the dose knob). Under SinkSGD the magnitude
    # moves as a pure +-lr sign step, so |log m| <= steps * lr_mag: this ratio sets how far
    # the per-channel multiplier can travel, independent of gradient magnitude.
    if other_params and cfg.adapter_type == "doft" and getattr(cfg, "doft_mag_lr", 0.0) > 0:
        if not cfg.lr:
            raise RuntimeError("doft_mag_lr needs a non-zero base lr to scale against")
        mag_ratio = float(cfg.doft_mag_lr) / float(cfg.lr)
        groups[-1]["lr_scale"] = mag_ratio
        print(f"[doft] magnitude group lr = {cfg.doft_mag_lr:g} ({mag_ratio:.4f}x base) "
              f"=> reach |log m| <= {cfg.doft_mag_lr:.4g} x steps "
              f"(~{cfg.doft_mag_lr * 300:.3f} over 300 steps)")

    return SinkSGD_adv(
        params=groups,
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
        nesterov_coef=cfg.nesterov_coef,
        normed_momentum=cfg.normed_momentum,
        snr_cond=cfg.snr_cond,
        orthogonal_sinkhorn=cfg.orthogonal_sinkhorn,
        sinkhorn_iterations=cfg.sinkhorn_iterations,
        cautious_wd=cfg.cautious_wd,
        geometric_wd=cfg.geometric_wd,
        centered_wd=cfg.centered_wd,
        centered_wd_mode=cfg.centered_wd_mode,
        spectral_normalization=cfg.spectral_normalization,
        orthogonal_gradient=cfg.orthogonal_gradient,
        state_precision=cfg.state_precision,
        stochastic_rounding=cfg.stochastic_rounding,
        nnmf_factor=cfg.nnmf_factor,
        vector_reshape=cfg.vector_reshape,
        compiled_optimizer=cfg.compiled_optimizer,
    )


def lr_at(step: int, total: int, cfg: TrainConfig, segment_start: int = 0) -> float:
    """Learning rate at `step`.

    Warmup and the cosine are measured from `segment_start` (the step this run SEGMENT
    began at -- 0 for a fresh run, or the resume step). That matters for warm restarts:
    with absolute-step warmup, resuming at step 1200 and asking for 25 warmup steps would
    skip warmup entirely and drop straight onto a cosine already 65% of the way through.

    The schedule interpolates between `lr_final` and `lr` rather than always decaying to
    zero, so an anneal target can be non-zero.
    """
    lo = float(getattr(cfg, "lr_final", 0.0) or 0.0)
    hi = float(cfg.lr)
    rel = step - segment_start
    span = max(1, total - segment_start)
    warm = max(0, int(cfg.warmup_steps))

    if warm and rel < warm:
        # linear warmup from the anneal floor up to the peak
        return lo + (hi - lo) * (rel + 1) / warm
    if cfg.lr_schedule == "constant" or span <= warm:
        return hi
    prog = min(1.0, (rel - warm) / max(1, span - warm))
    # NOTE: an unrecognised schedule used to fall through to cosine silently. That is the
    # same class of bug as oft_scaled/use_cayley_neumann -- a forward-affecting setting that
    # quietly does something other than what was asked. Reject it loudly instead.
    if cfg.lr_schedule == "cosine":
        frac = 0.5 * (1 + math.cos(math.pi * prog))
    elif cfg.lr_schedule == "linear":
        # Straight line from peak to the anneal floor. Compared with cosine at the same
        # span, linear is LOWER in the first ~60% and HIGHER in the last ~40%: cosine is
        # flat at the start and plunges at the end. At prog=0.364 (step 500 of 1200)
        # cosine gives 0.708*peak where linear gives 0.636*peak; at prog=0.818 (step 1000)
        # cosine gives 0.079*peak where linear gives 0.182*peak.
        frac = 1.0 - prog
    else:
        raise ValueError(
            f"unknown lr_schedule {cfg.lr_schedule!r}; expected 'cosine', 'linear' or "
            f"'constant'. Refusing to silently substitute a different schedule.")
    return lo + (hi - lo) * frac


def save_checkpoint(path: Path, model, opt, step: int, epoch: int, cfg: TrainConfig,
                    metrics: dict | None = None, final: bool = False, lean: bool = False) -> None:
    """Write a resumable checkpoint: adapter rotations + optimizer state + position.

    The optimizer state matters -- SinkSGD_adv carries momentum / anchor / sinkhorn
    state, so restoring only the weights would silently restart the update rule.

    `lean=True` writes adapters only (no optimizer state), for the portable artifact.
    """
    torch.save({
        "adapter": model.adapter_state_dict(),
        "optimizer": None if lean else opt.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": asdict(cfg),
        "metrics": metrics,
        "oft_stats": model.oft_stats,
        "tag_counts": model.tag_counts,
        "final": final,
    }, path)


def read_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Load a checkpoint file. Separate from restoring so the stored CONFIG can be read
    and reconciled BEFORE the model is built -- which is required, not merely tidy:
    oft_scaled changes the forward math, so a model built with the wrong setting is
    already wrong by the time weights are loaded into it."""
    if Path(path).is_dir():
        raise IsADirectoryError(
            f"--resume expects the checkpoint FILE, not a directory. Try "
            f"{Path(path) / 'checkpoint.pt'}")
    return torch.load(path, map_location=device, weights_only=False)


def restore_from_checkpoint(ck: dict, model, opt, device: str):
    """Restore adapter + optimizer state from an already-loaded checkpoint.

    Returns (step, epoch, metrics). The optimizer state matters -- SinkSGD_adv carries
    momentum / anchor / sinkhorn state, so restoring only the weights would silently
    restart the update rule.
    """
    model.load_adapter_state_dict(ck["adapter"])
    if ck.get("optimizer") is not None:
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:
            print(f"[resume] WARNING: could not restore optimizer state ({e}); "
                  f"continuing with weights only -- momentum/anchor state restarts")
    else:
        print("[resume] WARNING: this checkpoint carries NO optimizer state. It is a lean "
              "adapters-only artifact (adapters.pt), so SinkSGD_adv's momentum / anchor / "
              "sinkhorn state RESTARTS FROM ZERO. The weights continue but the UPDATE RULE "
              "does not -- this is not a faithful resume. Use checkpoint.pt instead.")
    step = int(ck.get("step", 0))
    epoch = int(ck.get("epoch", 0))
    print(f"[resume] restored weights: step={step} epoch={epoch} "
          f"adapter tensors={len(ck['adapter'])}")
    return step, epoch, ck.get("metrics")


# Keys that MUST come from the checkpoint on resume. Changing any of these mid-run
# either alters the forward math without changing parameter keys (oft_scaled,
# normalize_embeds), changes the update rule (momentum, nesterov, orthograd, compile,
# spectral), or changes the adapter geometry (block_size, backbones).
_RESUME_LOCKED = (
    "backbones", "block_size", "block_share", "oft_scaled", "oft_dropout",
    # adapter_type was MISSING here until 2026-10-01: it decides WHICH adapter is injected
    # (oft | doft | lora | dora) and therefore the forward math and the parameter-name set.
    # A resume that changed it would build the new adapter first and only then trip
    # load_adapter_state_dict(strict=True) -- or, on a lenient path, load an empty adapter.
    "adapter_type",
    "oft_weight_dtype", "normalize_embeds", "per_branch_norm", "image_size", "dtype",
    # Adapter surface + tower pooling (2026-09-26): both are forward-affecting with
    # identical parameter names. oft_targets was missing here since it was added -- a
    # resume with a different surface would build first and trip
    # load_adapter_state_dict(strict=True) second; the reconciliation should own it.
    "oft_targets", "qwen_pool",
    # Multi-channel readout (2026-10-02): decides WHICH layers feed the distance and
    # whether channel_weight_logits exists at all -- forward-affecting both ways.
    # Normalised to tuple on comparison below (JSON stores it as a list).
    "readout_layers",
    # soft_gap_ref (2026-10-01) changes the TRAINING SIGNAL, not the forward math, so no
    # parameter shape or name would catch a changed value: a resume that turned it on (or
    # changed the scale) would silently train against a different objective than the one
    # its step count describes. Latent until now because no config set it.
    "soft_gap_ref",
    "momentum", "nesterov", "nesterov_coef", "normed_momentum", "snr_cond",
    "orthogonal_sinkhorn", "sinkhorn_iterations", "cautious_wd", "geometric_wd",
    "centered_wd", "centered_wd_mode", "spectral_normalization",
    "orthogonal_gradient", "state_precision", "stochastic_rounding",
    "nnmf_factor", "vector_reshape", "compiled_optimizer",
    "dinov3_rope_augment", "interpolate_pos", "baked_pos", "gradient_checkpointing",
    "split_version",
)


def reconcile_resume_config(cfg: TrainConfig, stored: dict) -> TrainConfig:
    """Force the architecture/optimizer config to match the checkpoint.

    Everything in _RESUME_LOCKED is adopted FROM the checkpoint, because resuming with
    different values silently changes the model or the update rule. Schedule and
    bookkeeping keys (lr, epochs, eval_every, num_workers, ...) stay as the caller asked,
    so you can still extend a run or change the LR deliberately.
    """
    if not stored:
        print("[resume] WARNING: checkpoint carries no config; cannot verify that the "
              "current settings match how it was trained.")
        return cfg
    diffs = []
    for k in _RESUME_LOCKED:
        if k not in stored:
            continue
        want = getattr(cfg, k)
        got = stored[k]
        if k == "backbones":
            want, got = tuple(want), tuple(got)
        if k == "readout_layers":
            want, got = tuple(want or ()), tuple(got or ())
        if want != got:
            diffs.append((k, want, got))
            setattr(cfg, k, got)
    if diffs:
        print("[resume] ============================================================")
        print("[resume] The checkpoint was trained with DIFFERENT settings than the ones")
        print("[resume] requested on this command line. Adopting the checkpoint's values,")
        print("[resume] because training continues under the OLD configuration:")
        for k, want, got in diffs:
            print(f"[resume]     {k}: requested {want!r}  ->  using checkpoint {got!r}")
        print("[resume] To change any of these on purpose, start a NEW run instead;")
        print("[resume] resuming under a different model/optimizer config would silently")
        print("[resume] train something other than what the artifact was.")
        print("[resume] ============================================================")
    else:
        print("[resume] config matches the checkpoint on all locked keys")
    return cfg


def amp_context(amp: str, device: str):
    """Autocast context for mixed precision.

    bf16 is chosen over fp16 deliberately: its exponent range matches fp32, so it needs no
    GradScaler and cannot overflow on the way to the loss. The OFT rotation disables autocast
    internally and stays fp32 (see OFTRotationModule.forward), so the delicate Cayley solve is
    unaffected by this setting either way.
    """
    if amp in (None, "off"):
        return contextlib.nullcontext()
    if amp != "bf16":
        raise ValueError(f"unknown amp {amp!r}; expected 'off' or 'bf16'. Refusing to guess.")
    if device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model: PerceptualModel, loader, device: str, max_batches: int = 0,
             criterion=None, amp: str = "off") -> dict:
    """2AFC accuracy + loss over the loader. Loss is weighted by triplet count so a
    short final batch does not count as much as a full one.

    LIBRARY FUNCTION -- train() no longer calls it (2026-09-26, kerok: NO evals
    during training). Post-run scoring lives in scripts/eval_baseline.py and
    scripts/eval_sweep.py, which run as their own process: the one in-process eval
    left in probe A's lifecycle (the final full-split pass) was the one that wedged,
    while its two subprocess evals were followed by perfectly clean steps.
    """
    model.eval()
    # (2026-09-26 forensic reversal: the empty_cache() that used to sit here was
    # REMOVED. On WDDM it releases on the driver side outside torch's accounting and the
    # ensuing host-commit churn wedged the input pipeline for the rest of the run --
    # stalls began 7 s after the first eval in 3 independent runs. Peak-VRAM OOM risk is
    # handled by the >2 GiB-guarded anti_creep() trim in the training loop instead.)
    if criterion is None:
        # Do NOT do this silently: a BT run that forgot to pass its criterion would
        # report a HINGE loss under a "val_loss" label. Accuracy is unaffected (it is
        # computed from the distances, not the criterion), but the number would be
        # mislabelled. Callers always pass their criterion explicitly; this is the guard.
        print("  [eval] WARNING: no criterion passed; val_loss will be HINGE(margin=0.05) "
              "regardless of the training loss. Pass criterion=... for a matching number.")
        criterion = HingeLoss(margin=0.05, device=device, reduction="sum")
    correct = 0.0
    scored = 0
    n = 0
    loss_sum = 0.0
    crit = criterion
    total = len(loader) if hasattr(loader, "__len__") else 0
    t0 = time.time()
    if total:
        print(f"  [eval] start: {total} batches (max_batches={max_batches or 'all'})", flush=True)
    for i, (ref, left, right, target, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        ref, left, right, target = (t.to(device) for t in (ref, left, right, target))
        with amp_context(amp, device):
            d0 = model(ref, left)
            d1 = model(ref, right)
            loss_sum += crit(d0 - d1, target).item()
        c, s = two_afc_scored(d0, d1, target)   # ties excluded from both
        correct += c.item()
        scored += s.item()
        n += target.shape[0]
        acc = correct / max(1, scored)
        avg_loss = loss_sum / max(1, n)
        rate = (i + 1) / max(1e-9, time.time() - t0)
        prog = f"  [eval] {i + 1}/{total} batches" if total else f"  [eval] {i + 1} batches"
        print(f"{prog} | {rate:.1f} it/s | triplets {n} | acc so far {acc * 100:.1f}% | loss {avg_loss:.4f}",
              flush=True)
    if total:
        print(f"  [eval] done in {time.time() - t0:.1f}s | acc {correct / max(1, scored) * 100:.2f}% "
              f"| loss {loss_sum / max(1, n):.4f} | n={n}", flush=True)
    if device == "cuda":
        # Reset the peak meter so the next "peak MiB" log line reflects TRAINING, not the
        # eval. (2026-09-26 forensic reversal: the empty_cache() here was REMOVED -- see
        # the note at the pre-eval site; it is the suspected WDDM wedge trigger.)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    return {"val_acc": correct / max(1, scored), "val_loss": loss_sum / max(1, n), "val_n": n}


def host_mem_sample() -> dict:
    """System memory pressure for steps.jsonl (forensic §6.6, 2026-09-26).

    One record per log tick turns the next wedge from a forensic session into a
    lookup: host RAM load, available RAM and commit charge sit next to torch's own
    peak in every logged step. Windows GlobalMemoryStatusEx via ctypes -- no psutil
    dependency; returns {} off-Windows or on any failure, so the log schema stays
    additive and can never break a run.
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_uint64),
                        ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        st = _MemoryStatusEx()
        st.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return {}
        return {"host_load_pct": int(st.dwMemoryLoad),
                "host_avail_mib": round(st.ullAvailPhys / 2**20, 1),
                "host_commit_pct": round(
                    100.0 * (1.0 - st.ullAvailPageFile / max(1, st.ullTotalPageFile)), 2)}
    except Exception:
        return {}


@torch.no_grad()
def adapter_health(model: PerceptualModel) -> dict:
    """Adapter-specific evidence that training is actually MOVING the adapter.

    OFT and DoRA share nothing measurable, so this is the one place that knows the
    difference. Calling the OFT rotation probe on a DoRA model would find no OFTLinear
    modules and log a confident 0.0 forever -- a dead health signal that looks like a
    stalled optimizer.
    """
    if model.adapter_type in ("oft", "doft"):
        r = oft_rotation_stats(model)
        out = {"oft_applied": r["applied"], "oft_raw": r["raw"],
               "oft_orth_err": r["orthogonality_err"]}
        if model.adapter_type == "doft":
            # DOFT also learns a per-output-channel magnitude, so the rotation probe alone
            # would under-report what is moving. dora_log_multiplier is ZERO-init, so
            # max|log m| IS the drift away from identity and exp() gives the multiplier
            # range actually in force -- logged every tick so a magnitude that never moves
            # (which would make the arm a plain OFT run wearing a different name) is visible
            # in steps.jsonl rather than inferred from checkpoints.
            mags = [p for n, p in model.named_parameters()
                    if n.endswith("dora_log_multiplier")]
            if mags:
                mx = max(float(p.abs().max()) for p in mags)
                out["doft_logmult_max"] = mx
                out["doft_mult_min"] = math.exp(-mx)
                out["doft_mult_max"] = math.exp(mx)
        return out
    # DoRA: lora_B is zero-initialised, so BA starts at 0 (adapter == identity) and its
    # norm is a clean "has anything moved" signal. The magnitude vectors are NOT
    # initialised to 1 -- peft sets them to the per-output-channel weight NORM -- so
    # drift is measured against their captured init (PerceptualModel._dora_mag_init).
    # Comparing them to 1.0, the obvious-looking choice, would report a huge constant.
    params = dict(model.named_parameters())
    b_norm = 0.0
    for n, p in params.items():
        if "lora_B" in n:
            b_norm = max(b_norm, float(p.norm()))

    mag_dev = 0.0
    n_mag = 0
    for n, init in getattr(model, "_dora_mag_init", {}).items():
        cur = params.get(n)
        if cur is None:
            continue
        # The init snapshot was taken at injection time, i.e. on CPU before .to(device).
        init = init.to(device=cur.device, dtype=cur.dtype)
        denom = init.abs().clamp_min(1e-8)
        mag_dev = max(mag_dev, float(((cur.detach() - init).abs() / denom).max()))
        n_mag += 1
    return {"dora_loraB_max_norm": b_norm, "dora_mag_rel_dev": mag_dev,
            "dora_mag_tensors": float(n_mag)}


class _PrefetchedIterator:
    """Build AND prime the next epoch's loader in a background thread.

    WHY (kerok directive 2026-09-26 "fix it"): on the pool/aspect paths a FRESH DataLoader
    is constructed per epoch, because the sampler's per-epoch geometry must be pickled into
    newly spawned workers. Under Windows spawn each worker re-imports torch, so teardown +
    respawn cost ~16 s of GPU-idle time at EVERY epoch boundary -- 10.5% of wall on the b6p
    run (11.3-17.2 s dt vs a 2.3 s median; forensic report
    scratch\\20260926\\STALL_INVESTIGATION_HOST_COMMIT.md).

    All of that work is CPU/IO, so it can overlap with the current epoch's GPU steps. This
    thread runs `iter(loader)`, which does the three expensive things in the same order the
    boundary used to: (1) the sampler's `_plan()` -- deterministic in seed+epoch, which also
    assigns the epoch's geometry to the PARENT dataset, (2) worker spawn (workers snapshot
    that geometry), (3) filling the prefetch queue.

    SEMANTICS ARE UNCHANGED: the plan is a pure function of (seed, epoch), the geometry
    assignment is idempotent, and the same assignments happen in the same order relative to
    the worker spawn -- only the wall-clock timing moves earlier. Batch order, batch
    membership and crops are identical to the synchronous build.

    Failure is non-fatal: the caller falls back to the synchronous build.
    """

    def __init__(self, build, epoch: int):
        self.epoch = epoch
        self.error: BaseException | None = None
        self.took_s: float | None = None
        self._it = None
        self._ready = threading.Event()
        self._t0 = time.time()
        self._t = threading.Thread(target=self._run, args=(build,), daemon=True,
                                   name=f"loader-prefetch-e{epoch}")
        self._t.start()

    def _run(self, build):
        try:
            self._it = iter(build())
            self.took_s = time.time() - self._t0
        except BaseException as e:  # surfaced in the caller, never swallowed
            self.error = e
        finally:
            self._ready.set()

    def take(self):
        """Block until the background build is done, then hand over the live iterator."""
        self._ready.wait()
        if self.error is not None:
            raise self.error
        print(f"  [prefetch] epoch {self.epoch} loader primed in {self.took_s:.1f}s "
              f"(overlapped with the previous epoch)")
        return self._it


class _PauseRequested(Exception):
    """The pause-flag file appeared. Caught to save and exit 0, not as an error."""


def _pause_flag_path(cfg: "TrainConfig", run_dir: Path) -> Path | None:
    """Return the pause flag's path if it exists, else None.

    Candidates, in order:
      1. <cwd>/<name>             -- the directory the trainer was launched from
      2. <project root>/<name>    -- run_dir is <root>/runs/<run>, so parents[1] is <root>
    so `touch PAUSE` in the repo root works regardless of the launch directory. One
    Path.exists() per training step is free relative to a ~2-4 s step.
    """
    name = (getattr(cfg, "pause_flag", "PAUSE") or "").strip()
    if not name:
        return None
    cands = [Path.cwd() / name]
    try:
        cands.append(run_dir.resolve().parents[1] / name)
    except (IndexError, OSError):
        pass
    for c in cands:
        if c.exists():
            return c
    return None


def train(cfg: TrainConfig, device: str | None = None, run_dir: Path | None = None) -> Path:
    torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.dtype)
    # Resuming writes back into the ORIGINAL run folder so steps.jsonl continues as one
    # history rather than splitting across a new timestamped dir on every restart.
    if run_dir is None:
        if cfg.resume:
            run_dir = Path(cfg.resume).resolve().parent
            print(f"[resume] continuing in existing run dir {run_dir}")
        else:
            run_dir = make_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run dir : {run_dir}")
    print(f"device  : {device}  dtype={cfg.dtype}")

    # -- data -----------------------------------------------------------------
    aspect = cfg.random_aspect
    if cfg.synthetic:
        # Non-square smoke shape: exercises SigLIP pos-interpolation, DINOv3 RoPE at a
        # long grid, bf16, checkpointing -- everything the 544 class adds -- without
        # needing bucketed batches (synthetic triplets all share the shape).
        shape = (672, 448) if aspect else None
        train_ds = SyntheticTwoAFCDataset(n=cfg.synthetic_n, seed=cfg.seed, shape=shape)
        print(f"[data] SYNTHETIC run: {len(train_ds)} train random triplets"
              + (f" at {shape}" if shape else ""))
    else:
        if not cfg.dataset_root:
            raise SystemExit("--dataset-root is required unless --synthetic is given")
        from dreamsim_oft.data import ASPECT_BUCKETS_544
        buckets = ASPECT_BUCKETS_544 if aspect else None
        if cfg.dataset == "fgresq":
            from dreamsim_oft.data import FGResQDataset
            train_ds = FGResQDataset(cfg.dataset_root, split="train",
                                     image_size=cfg.image_size, aspect_buckets=buckets,
                                     color_jitter=cfg.color_jitter, hflip=cfg.hflip,
                                     shape_seed=cfg.seed,
                                     exclude_scene34=cfg.exclude_scene34,
                                     split_version=cfg.split_version)
        elif cfg.dataset == "nights":
            train_ds = TwoAFCDataset(cfg.dataset_root, split="train", image_size=cfg.image_size,
                                     aspect_buckets=buckets, color_jitter=cfg.color_jitter,
                                     hflip=cfg.hflip,
                                     shape_seed=cfg.seed)
        elif cfg.dataset == "bapps":
            # Multi-corpus ingredient #1: BAPPS 2afc train, 151,400 human judgments over
            # cnn|mix|traditional (38,120 / 56,640 / 56,640). target = judge = P(p1 preferred),
            # which IS the loss's "x1 more similar" convention (see dreamsim_oft/bapps.py).
            from dreamsim_oft.bapps import BAPPSDataset
            train_ds = BAPPSDataset(cfg.dataset_root, split="train",
                                    image_size=cfg.image_size, hflip=cfg.hflip)
        elif cfg.dataset == "diffiqa":
            # Multi-corpus ingredient #2: DiffIQA cc-type rows only (PNY/PSY/SNY/SSY,
            # ~121K train rows). The *YY types (50.6%) put the REFERENCE in a candidate
            # slot -- d(ref,ref)=0 makes them structurally unscoreable for this metric
            # family (measured 2026-09-30, full Test: PYY exactly 0% frozen AND trained;
            # see dreamsim_oft/diffiqa.py's docstring). target = 1 - gt.
            from dreamsim_oft.diffiqa import DiffIQADataset
            train_ds = DiffIQADataset(cfg.dataset_root, split="train",
                                      image_size=cfg.image_size, hflip=cfg.hflip)
        elif cfg.dataset == "mix3":
            # Confluence: equal-thirds mixture of FGResQ (sv2, excl34 per flags) + BAPPS +
            # DiffIQA-cc, all pairwise. dataset_root = the FGResQ root; the other two
            # roots come from bapps_root/diffiqa_root (empty -> canonical I: paths).
            from dreamsim_oft.mixdata import Mix3Dataset
            from dreamsim_oft.bapps import BAPPSDataset
            from dreamsim_oft.diffiqa import DiffIQADataset
            from dreamsim_oft.data import FGResQDataset
            fgr = FGResQDataset(cfg.dataset_root, split="train",
                                image_size=cfg.image_size, aspect_buckets=buckets,
                                color_jitter=cfg.color_jitter, hflip=cfg.hflip,
                                shape_seed=cfg.seed,
                                exclude_scene34=cfg.exclude_scene34,
                                split_version=cfg.split_version)
            bap = BAPPSDataset(cfg.bapps_root or r"I:\MyApps\sd-train\BAPPS", split="train",
                               image_size=cfg.image_size, hflip=cfg.hflip)
            dif = DiffIQADataset(cfg.diffiqa_root or r"I:\MyApps\sd-train\DiffIQA",
                                 split="train", image_size=cfg.image_size, hflip=cfg.hflip)
            train_ds = Mix3Dataset(fgr, bap, dif, seed=cfg.seed)
        else:
            raise SystemExit(f"unknown dataset {cfg.dataset!r}; expected nights|fgresq|bapps|diffiqa|mix3")

    # -- D1 dense pool batching (TRAINING side only; eval stays pairwise) -------
    pool_ds = None
    if cfg.pool_batch:
        from functools import partial
        from dreamsim_oft.data import collate_pools
        if cfg.synthetic:
            from dreamsim_oft.data import SyntheticPoolDataset
            pool_ds = SyntheticPoolDataset(n_pools=cfg.synthetic_n, k=cfg.pool_k,
                                           seed=cfg.seed,
                                           shape=(672, 448) if aspect else None)
            print(f"[data] SYNTHETIC pool mode: {len(pool_ds)} pools x k={cfg.pool_k}")
        elif cfg.dataset == "fgresq":
            from dreamsim_oft.data import FGResQPoolDataset
            pool_ds = FGResQPoolDataset(cfg.dataset_root, image_size=cfg.image_size,
                                        aspect_buckets=buckets,
                                        color_jitter=cfg.color_jitter,
                                        hflip=cfg.hflip,
                                        k=cfg.pool_k, exclude_scene34=cfg.exclude_scene34,
                                        split_version=cfg.split_version,
                                        soft_gap_ref=cfg.soft_gap_ref)
        else:
            raise SystemExit("--pool-batch is only implemented for --dataset fgresq "
                             "(or --synthetic)")
        pool_collate = partial(collate_pools, k=cfg.pool_k)

    if aspect and not cfg.synthetic:
        from dreamsim_oft.data import AspectBatchSampler
        train_loader = make_loader(train_ds, cfg.batch_size, num_workers=cfg.num_workers,
                                   pin_memory=cfg.pin_memory,
                                   persistent_workers=cfg.persistent_workers,
                                   prefetch_factor=(cfg.prefetch_factor or None),
                                   batch_sampler=AspectBatchSampler(
                                       train_ds, cfg.batch_size, seed=cfg.seed, epoch=0))
    else:
        train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                   seed=cfg.seed, pin_memory=cfg.pin_memory,
                                   persistent_workers=cfg.persistent_workers,
                                   prefetch_factor=(cfg.prefetch_factor or None))
    # (2026-09-26, kerok: the val dataset/loader, the fgresq val decode cache, the
    # preflight decode and the eval_batch_size rebuild are GONE -- no evals happen in
    # this process, ever. Post-run scoring: scripts/eval_sweep.py, which runs as its
    # own process after train -> finish/stop.)

    # -- resume: reconcile config BEFORE building the model -------------------
    # Order matters. oft_scaled/normalize_embeds change the forward MATH without changing
    # any parameter key, so a model constructed with the wrong setting would be wrong
    # before a single weight was loaded -- and would then load silently.
    resume_ck = None
    if cfg.resume:
        resume_ck = read_checkpoint(cfg.resume)
        cfg = reconcile_resume_config(cfg, resume_ck.get("config") or {})

    # -- model ----------------------------------------------------------------
    model = PerceptualModel(
        keys=cfg.backbones, block_size=cfg.block_size, block_share=cfg.block_share,
        oft_scaled=cfg.oft_scaled, dropout_probability=cfg.oft_dropout,
        oft_weight_dtype=getattr(torch, cfg.oft_weight_dtype),
        use_cayley_neumann=cfg.use_cayley_neumann,
        adapter_type=cfg.adapter_type, dora_r=cfg.dora_r, dora_alpha=cfg.dora_alpha,
        dora_dropout=cfg.dora_dropout,
        dora_targets=tuple(t.strip() for t in cfg.dora_targets.split(",") if t.strip()),
        oft_targets=cfg.oft_targets,
        normalize_embeds=cfg.normalize_embeds, per_branch_norm=cfg.per_branch_norm,
        image_size=cfg.image_size,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dinov3_rope_augment=cfg.dinov3_rope_augment,
        interpolate_pos=cfg.interpolate_pos,
        baked_pos=cfg.baked_pos,
        qwen_pool=cfg.qwen_pool,
        readout_layers=tuple(cfg.readout_layers or ()),
        dtype=dtype, device=device,
    ).to(device)
    describe_tags(model)
    # Print the FORWARD-AFFECTING knobs as the constructed model actually holds them, read
    # off the module rather than off cfg. These keys change the forward maths while leaving
    # every parameter name and shape identical, so a config/constructor mismatch is
    # undetectable in the saved artifact and would train a quietly different model
    # (oft_scaled and use_cayley_neumann both bit this project before). Reading them back
    # off `model` is the only check that cannot agree with a buggy constructor.
    print(f"[model] forward knobs: per_branch_norm={model.per_branch_norm} "
          f"normalize_embeds={model.normalize_embeds} "
          f"use_cayley_neumann={model.use_cayley_neumann} oft_scaled={model.oft_scaled} "
          f"block_size={model.block_size} adapter_type={model.adapter_type} "
          f"image_size={model.image_size} qwen_pool={model.qwen_pool} "
          f"readout_layers={getattr(model, 'readout_layers', ())} amp={cfg.amp}")
    from dreamsim_oft.data import ASPECT_BUCKETS_544 as _AB
    _cls = (f"ASPECT {cfg.image_size}px, {len(_AB)} buckets, redrawn per epoch (seed+epoch)"
            if cfg.random_aspect else
            f"SQUARE {cfg.image_size}x{cfg.image_size}px, no aspect draw")
    print(f"[model] INPUT CLASS: {_cls}")
    print(f"[model] class knobs: grad_ckpt={cfg.gradient_checkpointing} "
          f"dinov3_rope_augment={cfg.dinov3_rope_augment} "
          f"siglip_interp_pos={cfg.interpolate_pos} siglip_baked_pos={cfg.baked_pos} "
          f"random_aspect={cfg.random_aspect} "
          f"color_jitter={cfg.color_jitter} hflip={cfg.hflip}")
    model.train()

    opt = build_optimizer(model, cfg)
    crit = build_criterion(cfg.loss_type, device, reduction="sum",
                           margin=cfg.margin, tau=cfg.bt_tau,
                           confidence_weight=getattr(cfg, "bt_confidence_weight", False))
    print(f"loss    : {cfg.loss_type}"
          + (f" (margin={cfg.margin})" if cfg.loss_type == "hinge" else f" (tau={cfg.bt_tau})")
          + (f" confidence_weight=ON (ties contribute 0)" if getattr(cfg, "bt_confidence_weight", False) else ""))
    if pool_ds is not None:
        print(f"pool    : loss={cfg.pool_loss} k={cfg.pool_k}"
              + (f" listnet_t={cfg.listnet_t} (standardization)"
                 if cfg.pool_loss == "listnet" and cfg.listnet_t <= 0 else "")
              + (f" listnet_t={cfg.listnet_t}" if cfg.pool_loss == "listnet" and cfg.listnet_t > 0 else "")
              + (f" tau_t={cfg.listnet_tau_t} vote_weight={cfg.listnet_vote_weight}"
                 if cfg.pool_loss == "listnet" else ""))

    if pool_ds is not None:
        from dreamsim_oft.data import PoolSampler
        steps_per_epoch = math.ceil(len(PoolSampler(pool_ds, cfg.batch_size, seed=cfg.seed,
                                                    epoch=0)) / max(1, cfg.grad_accum))
    else:
        steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum))
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    print(f"steps   : {steps_per_epoch}/epoch, {total_steps} total (grad_accum={cfg.grad_accum})")
    print(f"optim   : SinkSGD_adv lr={cfg.lr} momentum={cfg.momentum} "
          f"normed_momentum={cfg.normed_momentum} ortho_sinkhorn={cfg.orthogonal_sinkhorn} "
          f"centered_wd={cfg.centered_wd} spectral={cfg.spectral_normalization}")

    steps_fh = (run_dir / "steps.jsonl").open("a", encoding="utf-8")
    cfg_path = run_dir / "config.json"
    if cfg_path.exists() and cfg.resume:
        # A resume must not erase the record of how the run was ORIGINALLY launched.
        (run_dir / "config.resume.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
        print("[resume] keeping original config.json; this launch -> config.resume.json")
    else:
        cfg_path.write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    step = 0
    micro = 0
    start_epoch = 0
    skip_batches = 0
    # Explicit ordering for the FIRST epoch of a resume, so the consumed prefix is never
    # decoded. None means "use the normal shuffled loader".
    resume_indices = None
    epoch = 0
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0.0
    running_n = 0
    running_loss_n = 0        # running_loss denominator (BT: pairs; listnet: pools)
    running_px = 0            # pixels decoded-and-trained since the last log line
    t_log = time.time()
    last_log_step = 0
    # (2026-09-26, kerok: best.pt during-run tracking REMOVED. It overwrote itself
    # with a filename that carried neither its step nor its score, and computing it
    # required in-run evals -- the machinery that wedged runs. Best step is now a ROW
    # in the post-run sweep table: scripts/eval_sweep.py over ckpt/stepNNNNNN.pt.)
    stop = False
    log_every = max(1, cfg.log_every)
    # Pool-BT: upper-triangle candidate pair indices (fixed k = cfg.pool_k).
    ii, jj = torch.triu_indices(max(1, cfg.pool_k), max(1, cfg.pool_k), 1)
    # D2 listnet logging acc: upper-triangle candidate pair indices (subsampled to the
    # first 16 valid candidates' worth of pairs at k>16; all C(16,2)=120 at k=16).
    pi, pj = torch.triu_indices(max(1, min(cfg.pool_k, 16)),
                                max(1, min(cfg.pool_k, 16)), 1)

    ckpt_path = run_dir / "checkpoint.pt"
    adapter_path = run_dir / "adapters.pt"  # portable artifact: adapters only, no optimizer
    # Preview dump target: the first N images the trainer feeds the towers, saved
    # exactly as fed (post-crop, post-jitter, [0,1]). Default ON -- see TrainConfig.
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(exist_ok=True)
    preview_left = max(0, cfg.preview_images)
    if resume_ck is not None:
        step, _, _ = restore_from_checkpoint(resume_ck, model, opt, device)
        # steps_per_epoch counts OPTIMIZER steps, but the loader yields BATCHES. With
        # grad_accum > 1 those differ, so skipping `step % steps_per_epoch` batches would
        # under-skip by exactly grad_accum and re-train data.
        if pool_ds is None and len(train_loader) % max(1, cfg.grad_accum) != 0:
            print(f"[resume] WARNING: {len(train_loader)} batches is not divisible by "
                  f"grad_accum={cfg.grad_accum}; mid-epoch resume will not be exact.")
        skip_batches = (step % max(1, steps_per_epoch)) * max(1, cfg.grad_accum)
        start_epoch = step // max(1, steps_per_epoch)
        micro = 0  # restart the accumulation window so the first resumed step is whole
        print(f"[resume] continuing at epoch {start_epoch}, skipping {skip_batches} batches "
              f"({step} optimizer steps already done)")
        if cfg.resume_reshuffle:
            # Re-shuffle and consume from batch 0. The skip computed above is tied to the OLD
            # permutation (it is derived from the step count, not from the data), so it MUST
            # be zeroed here: under a new permutation those first 331 batches are different
            # triplets, and skipping them would drop unseen data while re-showing data the
            # previous segment already trained on -- the exact failure this option exists to
            # avoid.
            skip_batches = 0
            rs = cfg.reshuffle_seed if cfg.reshuffle_seed is not None else cfg.seed + 7919
            print(f"[resume] RESHUFFLE ON: fresh permutation (seed {rs}) and starting at "
                  f"batch 0 of epoch {start_epoch}. The {step} steps already done are NOT "
                  f"fast-forwarded, so this segment draws a new sample from all "
                  f"{len(train_loader.dataset)} training triplets.")
        elif skip_batches and pool_ds is None:
            # SAMPLER-LEVEL SKIP -- this is the fix for the NVMe thrash.
            #
            # The epoch loop below also has a `continue`-based skip, and it is a trap: a
            # `continue` still FETCHES AND DECODES the batch from the DataLoader before
            # discarding it. Resuming at step 2900 means skipping 293 batches = ~14,000 images
            # decoded and thrown away. Measured on the real machine, that pinned the dataset
            # NVMe at 100% active time / 125 MB/s with the CPU at 100% and the GPU idling at
            # 7% -- minutes of wall clock per resume, spent entirely on images nobody uses.
            #
            # Building the SAME permutation and slicing off the consumed prefix means those
            # files are never opened. The ordering is identical to what the `continue` path
            # produced, because both come from RandomSampler seeded with `seed + start_epoch`.
            from torch.utils.data import RandomSampler
            gen = torch.Generator().manual_seed(cfg.seed + start_epoch)
            order = list(RandomSampler(train_ds, generator=gen))
            cut = skip_batches * cfg.batch_size
            resume_indices = order[cut:]
            print(f"[resume] sampler-level skip: the first {skip_batches} batches "
                  f"({cut} samples) are REMOVED FROM THE SAMPLER, not decoded and discarded. "
                  f"{len(resume_indices)} samples remain in epoch {start_epoch}.")
    # Where THIS segment's LR schedule begins. 0 for a fresh run; the resume step for a
    # continuation, so warmup and the cosine are measured from here rather than from 0.
    segment_start = step
    if cfg.lr_absolute_schedule:
        # Continue the ORIGINAL curve across an interruption instead of warm-restarting.
        segment_start = 0
        print(f"[resume] lr_absolute_schedule ON: the LR curve is measured from step 0, so "
              f"this continues the original schedule rather than restarting warmup at "
              f"step {step}.")

    try:
        # --- VRAM hygiene (sinkosaur recipe, 2026-09-26) ---------------------
        # Ordering per vn_MemoryCleanup (kerok's ComfyUI node) as ported from
        # sinkosaur3-vae-studio: synchronize FIRST so no async kernel holds
        # tensor refs, then gc so wrappers die, then empty_cache LAST. Syncless
        # empty_cache releases nothing (async-held blocks stay committed).
        def cuda_settle():
            # (2026-09-26 forensic reversal: empty_cache() removed here too -- on WDDM it
            # frees nothing measurable while triggering driver-side host-commit churn.
            # synchronize + gc still release Python-side refs, which is the useful part.)
            if device == "cuda":
                torch.cuda.synchronize()
            gc.collect()

        # Anti-creep trim (sinkosaur loop.py): every N optimizer steps, return
        # cached-but-unallocated segments to the driver ONLY when the slack
        # exceeds 2 GiB, so the cudaMalloc re-tax is worth paying.
        anti_creep_every = 25

        def anti_creep():
            if device != "cuda":
                return
            if (torch.cuda.memory_reserved()
                    - torch.cuda.max_memory_allocated()) > 2 * 2**30:
                torch.cuda.empty_cache()

        def _epoch_loader_for(e: int, skip: int):
            """Build the per-epoch loader for epoch `e` (pool/aspect paths), else None.

            Extracted from the epoch loop so _PrefetchedIterator can build epoch e+1 while
            e is still training. `skip` only ever applies to the first epoch of a resumed
            segment; prefetched epochs always pass 0.
            """
            if pool_ds is not None:
                from dreamsim_oft.data import PoolSampler
                return make_loader(pool_ds, cfg.batch_size, num_workers=cfg.num_workers,
                                   pin_memory=cfg.pin_memory,
                                   persistent_workers=cfg.persistent_workers,
                                   prefetch_factor=(cfg.prefetch_factor or None),
                                   collate_fn=pool_collate,
                                   batch_sampler=PoolSampler(pool_ds, cfg.batch_size,
                                                             seed=cfg.seed, epoch=e,
                                                             skip_batches=skip))
            if aspect and not cfg.synthetic:
                from dreamsim_oft.data import AspectBatchSampler
                return make_loader(train_ds, cfg.batch_size, num_workers=cfg.num_workers,
                                   pin_memory=cfg.pin_memory,
                                   persistent_workers=cfg.persistent_workers,
                                   prefetch_factor=(cfg.prefetch_factor or None),
                                   batch_sampler=AspectBatchSampler(train_ds, cfg.batch_size,
                                                                    seed=cfg.seed, epoch=e,
                                                                    skip_batches=skip))
            return None

        # Carried across epoch iterations: the background build of the NEXT epoch's loader.
        prefetch: _PrefetchedIterator | None = None
        prefetch_epoch: int | None = None

        for epoch in range(start_epoch, cfg.epochs):
            # EPOCH-BOUNDARY HYGIENE: the previous epoch's loader (workers +
            # pinned buffers) dies while the fresh one spins up; settle the
            # heap BEFORE the new loader exists so the WDDM wedge cannot form.
            active_loader = None
            if not cfg.synthetic:
                cuda_settle()
            if prefetch is not None and prefetch_epoch == epoch:
                # Primed in the background during the previous epoch: the ~16 s worker
                # respawn already happened, so the boundary costs ~nothing. Fall back to
                # a synchronous build if the background thread failed.
                try:
                    active_loader = prefetch.take()
                except Exception as pe:
                    print(f"[prefetch] epoch {epoch}: background build failed "
                          f"({type(pe).__name__}: {pe}) -- building synchronously")
                    active_loader = _epoch_loader_for(epoch, 0)
                prefetch = None
                prefetch_epoch = None
            elif pool_ds is not None:
                # Pool mode: a FRESH deterministic loader per epoch -- the PoolSampler
                # plan (bucket assignment + pool order) is seeded by seed+epoch, so a
                # resumed run reproduces it exactly. skip_batches slices the consumed
                # prefix at the sampler level (the NVMe-thrash lesson).
                from dreamsim_oft.data import PoolSampler
                skip = skip_batches if epoch == start_epoch else 0
                skip_batches = 0
                active_loader = make_loader(pool_ds, cfg.batch_size,
                                            num_workers=cfg.num_workers,
                                            pin_memory=cfg.pin_memory,
                                            persistent_workers=cfg.persistent_workers,
                                            prefetch_factor=(cfg.prefetch_factor or None),
                                            collate_fn=pool_collate,
                                            batch_sampler=PoolSampler(
                                                pool_ds, cfg.batch_size, seed=cfg.seed,
                                                epoch=epoch, skip_batches=skip))
            elif aspect and not cfg.synthetic:
                # Aspect mode: a FRESH deterministic loader per epoch -- the sampler's
                # plan (bucket assignment + batch grouping) is seeded by seed+epoch, so
                # every epoch redraws crops AND a resumed run reproduces them exactly.
                # skip_batches slices the consumed prefix at the sampler level (the
                # NVMe-thrash lesson) on the first epoch of a resume only.
                from dreamsim_oft.data import AspectBatchSampler
                skip = skip_batches if epoch == start_epoch else 0
                # Consumed at the SAMPLER level; zero it so the inner `continue`
                # fallback cannot double-skip (it would fetch-and-discard).
                skip_batches = 0
                active_loader = make_loader(train_ds, cfg.batch_size,
                                            num_workers=cfg.num_workers,
                                            pin_memory=cfg.pin_memory,
                                            persistent_workers=cfg.persistent_workers,
                                            prefetch_factor=(cfg.prefetch_factor or None),
                                            batch_sampler=AspectBatchSampler(
                                                train_ds, cfg.batch_size, seed=cfg.seed,
                                                epoch=epoch, skip_batches=skip))
            else:
                # Deterministic per-epoch shuffle so a resumed run sees the same order.
                if train_loader.generator is not None:
                    data_seed = cfg.seed + epoch
                    # With reshuffle on, the FIRST epoch of this segment uses a different seed, so
                    # the permutation is not the one the previous segment was part-way through.
                    # Later epochs of the segment fall back to the normal per-epoch seed.
                    if cfg.resume_reshuffle and epoch == start_epoch:
                        data_seed = (cfg.reshuffle_seed if cfg.reshuffle_seed is not None
                                     else cfg.seed + 7919)
                    train_loader.generator.manual_seed(data_seed)

                # First epoch of a resume: iterate the PRE-SLICED ordering, so the batches the
                # previous segment already consumed are never fetched from disk at all. Falling
                # back to train_loader on every other epoch keeps the normal shuffle.
                if epoch == start_epoch and resume_indices is not None:
                    active_loader = make_loader(train_ds, cfg.batch_size,
                                                num_workers=cfg.num_workers,
                                                indices=resume_indices,
                                                pin_memory=cfg.pin_memory)
                    print(f"[resume] epoch {epoch}: iterating {len(active_loader)} pre-sliced "
                          f"batches (consumed prefix never decoded)")
                else:
                    active_loader = train_loader

            for batch_i, batch in enumerate(active_loader):
                if (epoch == start_epoch and resume_indices is None
                        and batch_i < skip_batches):
                    continue  # only when no sampler-level slice was built

                # ---- background-build the NEXT epoch's loader --------------------------
                # ORDERING IS LOAD-BEARING: this must happen only AFTER this epoch's own
                # `iter(active_loader)` has completed, i.e. after its workers exist. The
                # worker processes snapshot the parent dataset's geometry AT SPAWN, so
                # assigning epoch e+1's geometry while epoch e was still spawning made the
                # batches be grouped by one geometry and resized by another -> collate_pools
                # "stack expects each equal size" (observed 2026-09-26). By the time the
                # first batch has been yielded, epoch e's plan is materialised and its
                # workers hold their own dataset copy, so the parent is free to move on.
                if (batch_i == 0 and prefetch is None and (epoch + 1) < cfg.epochs
                        and (pool_ds is not None or (aspect and not cfg.synthetic))):
                    prefetch = _PrefetchedIterator(
                        lambda e=epoch + 1: _epoch_loader_for(e, 0), epoch + 1)
                    prefetch_epoch = epoch + 1

                if pool_ds is not None:
                    # ---- D1/D2 pool forward ----
                    # batch: images_ref (B,3,H,W), images_cand (B,k,3,H,W),
                    #        votes (B,k,k), mask (B,k,k), scores (B,k) score_norm,
                    #        cand_valid (B,k) bool, vote_w (B,k);
                    # votes[i,j]=1 <=> j preferred
                    ref_i, cand_i, votes_i, mask_i, scores_i, candv_i, votew_i = batch
                    if isinstance(cand_i, (list, tuple)):
                        # default_collate (synthetic path): a k-list of (B,3,H,W)
                        cand_i = torch.stack(list(cand_i), dim=1)
                    ref_i = ref_i.to(device, non_blocking=True)
                    cand_i = cand_i.to(device, non_blocking=True)
                    votes_i = votes_i.to(device)
                    mask_i = mask_i.to(device)
                    scores_i = scores_i.to(device)
                    candv_i = candv_i.to(device)
                    votew_i = votew_i.to(device)
                    Bn, kn = mask_i.shape[:2]
                    # PREVIEW (pool mode): dump the FIRST pool of the batch -- the ref plus
                    # every VALID candidate -- one pool per step, spanning steps. The
                    # score_norm (the ListNet target signal) goes INTO THE FILENAME, and the
                    # pool's argmax candidate is suffixed _TOP, so the pictures double as a
                    # direct visualization of the target distribution the loss chases.
                    if preview_left > 0:
                        from torchvision.utils import save_image
                        h, w = ref_i.shape[-2], ref_i.shape[-1]
                        pool = 0
                        sc = scores_i[pool]
                        vv = candv_i[pool]
                        if preview_left > 0:
                            save_image(ref_i[pool].clamp(0, 1).cpu(),
                                       str(preview_dir / f"step{step + 1:04d}_p00_ref_{h}x{w}.png"))
                            preview_left -= 1
                        top = int(torch.argmax(sc.masked_fill(~vv, -float("inf"))).item())
                        for ci in range(kn):
                            if not bool(vv[ci]) or preview_left <= 0:
                                continue
                            tag = "_TOP" if ci == top else ""
                            save_image(cand_i[pool, ci].clamp(0, 1).cpu(),
                                       str(preview_dir / f"step{step + 1:04d}_p00_c{ci:02d}"
                                           f"_s{float(sc[ci]):.3f}{tag}_{h}x{w}.png"))
                            preview_left -= 1
                    with amp_context(cfg.amp, device):
                        # ONE forward through the UNCHANGED model.embed(): ref + all
                        # candidates flattened into a single image stack.
                        emb = model.embed(torch.cat([ref_i, cand_i.flatten(0, 1)], dim=0))
                    # embed() returns BLOCK layout [refs(B), cands(B*k)] -- a plain
                    # .view(B, k+1, D) interleaves wrongly (pool b's candidate slots
                    # would span other pools' refs/cands; forensic audit 2026-09-26,
                    # scratch\20260925\audit_d2\FORENSIC_REPORT.md). Reshape the
                    # candidate block per pool and re-attach each pool's ref at slot 0.
                    emb = emb.float()
                    emb = torch.cat([emb[:Bn].unsqueeze(1),
                                     emb[Bn:].view(Bn, kn, -1)], dim=1)
                    d = 1 - torch.nn.functional.cosine_similarity(
                        emb[:, :1], emb[:, 1:], dim=-1)          # (B, k) dist to ref
                    if cfg.pool_loss == "listnet":
                        # ---- D2 ListNet: CE against a score_norm target distribution ----
                        if cfg.listnet_t > 0:
                            # explicit temperature (future ablation only)
                            z = -d / float(cfg.listnet_t)
                        else:
                            # per-pool standardization over VALID candidates
                            cnt = candv_i.sum(-1, keepdim=True).clamp(min=1).float()
                            mu = d.masked_fill(~candv_i, 0.0).sum(-1, keepdim=True) / cnt
                            var = ((d - mu).masked_fill(~candv_i, 0.0) ** 2).sum(
                                -1, keepdim=True) / cnt
                            # closer (smaller d) must get HIGHER probability mass to
                            # match the score_norm target direction; sign was inverted
                            # before the 2026-09-26 forensic audit (explicit-temp
                            # branch z = -d/t was already correct).
                            z = -(d - mu) / (var.sqrt() + 1e-6)
                        z = z.masked_fill(~candv_i, -float("inf"))
                        # target: softmax(score_norm / tau_t) over valid, optionally
                        # vote-weighted, renormalized; padded slots get exactly 0.
                        t = torch.softmax(
                            scores_i.masked_fill(~candv_i, -float("inf"))
                            / float(cfg.listnet_tau_t), dim=-1)
                        if cfg.listnet_vote_weight:
                            t = t * votew_i
                            t = t / t.sum(-1, keepdim=True).clamp(min=1e-8)
                        t = t.masked_fill(~candv_i, 0.0)
                        logp = torch.log_softmax(z, dim=-1)
                        # 0 * -inf would be NaN; zero the padded log-probs explicitly so
                        # padded slots contribute exactly 0 to the loss.
                        logp = torch.where(candv_i, logp, torch.zeros_like(logp))
                        L = -(t * logp).sum(-1)                   # (B,) per-pool CE
                        loss_sum = L.sum().float()
                        if not torch.isfinite(loss_sum):
                            raise FloatingPointError(
                                f"non-finite pool loss at step {step} (epoch {epoch}). "
                                f"The last save_every checkpoint is still usable.")
                        with torch.no_grad():
                            # logging acc: pairwise sign agreement between distance and
                            # score over VALID candidates, near-ties skipped.
                            pred = d[:, pi] < d[:, pj]
                            want = scores_i[:, pi] > scores_i[:, pj]
                            pv = candv_i[:, pi] & candv_i[:, pj] & \
                                ((scores_i[:, pi] - scores_i[:, pj]).abs() > 1e-6)
                            correct = ((pred == want) & pv).sum()
                            scored = pv.sum()
                        loss = loss_sum / Bn / max(1, cfg.grad_accum)
                        loss.backward()
                        running_px += Bn * (kn + 1) * ref_i.shape[-2] * ref_i.shape[-1]
                        # listnet logs per-POOL loss, not per-pair: the shared tail's
                        # running_loss denominator is running_loss_n (below), not scored.
                        running_loss_n += Bn
                    else:
                        # ---- D1 pool-BT: every supervised candidate pair is one logit
                        # (statements unchanged; re-indented under this else only).
                        z = (d[:, ii] - d[:, jj]) / float(cfg.bt_tau)  # (B, P) upper triangle
                        pmask = mask_i[:, ii, jj]
                        zv = z[pmask]
                        yv = votes_i[:, ii, jj][pmask]
                        if zv.numel() == 0:
                            continue  # degenerate pool batch: nothing supervised here
                        loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
                            zv, yv, reduction="sum")
                        loss_sum = loss_sum.float()
                        if not torch.isfinite(loss_sum):
                            raise FloatingPointError(
                                f"non-finite pool loss at step {step} (epoch {epoch}). "
                                f"The last save_every checkpoint is still usable.")
                        with torch.no_grad():
                            pred_i_better = d[:, ii] < d[:, jj]
                            want_j_better = votes_i[:, ii, jj] > 0.5
                            correct = (((pred_i_better != want_j_better) & pmask).sum())
                            scored = pmask.sum()
                        loss = loss_sum / scored / max(1, cfg.grad_accum)
                        loss.backward()
                        running_px += Bn * (kn + 1) * ref_i.shape[-2] * ref_i.shape[-1]
                        # BT logs per-PAIR loss; running_loss_n tracks its denominator.
                        running_loss_n += scored.item()
                else:
                    # Datasets return (ref, left, right, target, id) -- slice off the
                    # first FOUR so a non-aspect loader that carries the id (or any
                    # future extra field) cannot crash the unpack here.
                    ref, left, right, target = (t.to(device, non_blocking=True)
                                                for t in batch[:4])
                    # PREVIEW: dump ONE TRIPLET (ref/left/right of item 0) per batch,
                    # spanning successive batches until N images are written. A batch is
                    # bucket-uniform by construction, so dumping within one batch would
                    # show a single resolution; spanning batches shows the real aspect
                    # variety, and the shared crop is visible across each triplet.
                    if preview_left > 0:
                        from torchvision.utils import save_image
                        h, w = ref.shape[-2], ref.shape[-1]
                        # Dump the WHOLE batch (every item, all three slots), not just item 0:
                        # at small step counts the per-batch-one-triplet scheme could not reach
                        # a large preview_images budget (batch 10 x 10 steps = 30 images, not 100).
                        for bi in range(ref.shape[0]):
                            for nm, t in (("ref", ref), ("left", left), ("right", right)):
                                if preview_left <= 0:
                                    break
                                save_image(t[bi].clamp(0, 1).cpu(),
                                           str(preview_dir / f"step{step + 1:04d}_b{bi:02d}_{nm}_{h}x{w}.png"))
                                preview_left -= 1
                            if preview_left <= 0:
                                break
                    with amp_context(cfg.amp, device):
                        d0 = model(ref, left)
                        d1 = model(ref, right)
                        loss_sum = crit(d0 - d1, target)
                    # NOTE: keep the raw SUM computed under autocast (it is used for reporting
                    # and for the finite-check), but do the per-triplet mean and backward in fp32
                    # so the scale of the update never depends on the autocast dtype.
                    loss_sum = loss_sum.float()
                    # NON-FINITE GUARD. A single NaN/Inf pixel propagates into every parameter
                    # and stays broken for the rest of the run, SILENTLY -- the loss keeps
                    # printing and just stops meaning anything. Abort loudly instead, and name
                    # which input is at fault so the offending sample can be found.
                    if not torch.isfinite(loss_sum):
                        bad = [nm for nm, t in (("ref", ref), ("left", left), ("right", right))
                               if not torch.isfinite(t).all()]
                        raise FloatingPointError(
                            f"non-finite loss at step {step} (epoch {epoch}). Non-finite input "
                            f"tensors: {bad if bad else 'NONE -> the adapter weights are already '
                            'non-finite'}. The last save_every checkpoint is still usable.")
                    loss = loss_sum / target.shape[0] / max(1, cfg.grad_accum)
                    loss.backward()
                    c_scored, s_scored = two_afc_scored(d0, d1, target)  # ties excluded
                    correct, scored = c_scored, s_scored
                    # Actual pixels pushed through BOTH towers this batch (all 3 images per
                    # triplet, true h x w whatever the bucket drew).
                    running_px += ref.shape[0] * 3 * ref.shape[2] * ref.shape[3]
                running_loss += loss_sum.item()
                # PAIR-BRANCH DENOMINATOR (bug found 2026-09-25 evening): the pool
                # refactor introduced running_loss_n as the logger's denominator and
                # incremented it ONLY in the pool branches, so pair runs logged the raw
                # BATCH-SUM (x batch_size; found at b60 = 60x). This line restores the
                # per-triplet mean. Value identical to the pre-refactor `running_n`.
                # (2026-09-26: `target` exists only in the pair branch -- guard so pool
                # mode does not NameError; pool branches increment this themselves.)
                if pool_ds is None:
                    running_loss_n += target.shape[0]
                running_correct += correct.item()
                running_n += scored.item()
                micro += 1

                if micro % max(1, cfg.grad_accum) != 0:
                    continue

                # anti-creep trim between optimizer steps (sinkosaur recipe;
                # see the cuda_settle definition at the epoch loop head)
                if step % anti_creep_every == 0:
                    anti_creep()

                lr = lr_at(step, total_steps, cfg, segment_start=segment_start)
                for g in opt.param_groups:
                    # oft_lr_scale carries the SOFT compensation; lr_scale is the generic
                    # per-group multiplier (DOFT's magnitude group uses it to set its own
                    # dose). Both absent on ordinary groups.
                    g["lr"] = lr * g.get("lr_scale", g.get("oft_lr_scale", 1.0))
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0 or step == 1:
                    vram = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
                    reserved = (torch.cuda.memory_reserved() / 2**20) if device == "cuda" else 0.0
                    health = adapter_health(model)
                    dt_log = max(1e-6, time.time() - t_log)
                    steps_per_s = (step - last_log_step) / dt_log
                    px_per_s = running_px / dt_log
                    # ETA (kerok 2026-09-26: "can you please put the ETA in it").
                    # Rate comes from THIS logging interval, not the segment average, so a
                    # stall or a slow eval-adjacent stretch shows up in the ETA immediately.
                    remaining = max(0, total_steps - step)
                    eta_s = (remaining / steps_per_s) if steps_per_s > 1e-6 and remaining else None
                    rec = {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "step": step, "epoch": epoch,
                        "loss": round(running_loss / max(1, running_loss_n), 6),
                        "acc": round(running_correct / max(1, running_n), 4),
                        "lr": lr,
                        "steps_per_s": round(steps_per_s, 4),
                        "px_per_s": round(px_per_s, 1),
                        **{k: round(v, 10) for k, v in health.items()},
                        "vram_peak_mib": round(vram, 1),
                        # Forensic §6.6 (2026-09-26): host-side pressure beside torch's
                        # own peak, so the next wedge is a lookup instead of a session.
                        "cuda_reserved_mib": round(reserved, 1),
                        **host_mem_sample(),
                        "elapsed_s": round(time.time() - t0, 1),
                        "eta_s": (round(eta_s, 1) if eta_s is not None else None),
                    }
                    steps_fh.write(json.dumps(rec) + "\n")
                    steps_fh.flush()
                    hs = " ".join(f"{k.replace('oft_', '').replace('dora_', '')}={v:.2e}"
                                  for k, v in health.items() if "tensors" not in k)
                    if eta_s is None:
                        eta_str = "--"
                    elif eta_s >= 3600:
                        eta_str = f"{int(eta_s // 3600)}h{int(eta_s % 3600 // 60):02d}m"
                    else:
                        eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                    print(f"  step {step:>5d} | loss {rec['loss']:.4f} | train_acc {rec['acc']:.3f} "
                          f"| {hs} | lr {lr:.2e} | {steps_per_s:.2f} st/s | "
                          f"{px_per_s/1e6:.2f} Mpx/s | peak {vram:.0f} MiB | resv {reserved:.0f} "
                          f"| host {rec.get('host_commit_pct', 0):.0f}% | {rec['elapsed_s']:.0f}s "
                          f"| eta {eta_str} ({remaining} left)")
                    running_loss = running_correct = 0.0
                    running_n = running_loss_n = 0
                    running_px = 0
                    t_log = time.time()
                    last_log_step = step

                # (2026-09-26, kerok: NO evals during training -- mid-run and final
                # eval blocks REMOVED from this loop. train_acc/loss in steps.jsonl
                # are the live signal; ckpt/stepNNNNNN.pt preserves every candidate;
                # scripts/eval_sweep.py scores them all after finish/stop under one
                # protocol. The in-process eval machinery was the wedge trigger --
                # probe A: two subprocess evals clean, its single in-process final
                # eval wedged at batch 7/28.)
                if cfg.save_every and step % cfg.save_every == 0:
                    save_checkpoint(ckpt_path, model, opt, step, epoch, cfg)
                    print(f"  [ckpt @ {step}] {ckpt_path.name} "
                          f"({ckpt_path.stat().st_size / 2**20:.1f} MiB)")
                    if cfg.versioned_checkpoints:
                        # NO-OVERWRITE HISTORY: keep a copy whose NAME carries the step,
                        # so every 50-step checkpoint stays testable forever (checkpoint.pt
                        # alone is overwritten and a mid-run val curve is unrecoverable).
                        vdir = run_dir / "ckpt"
                        vdir.mkdir(exist_ok=True)
                        vpath = vdir / f"step{step:06d}.pt"
                        if not vpath.exists():
                            shutil.copy2(ckpt_path, vpath)
                            print(f"  [ckpt @ {step}] {vpath.name} (versioned copy)")

                # PAUSE FLAG -- checked every step, acted on at the END of the step so the
                # saved checkpoint is a clean step boundary (kerok 2026-09-26).
                _pflag = _pause_flag_path(cfg, run_dir)
                if _pflag is not None:
                    raise _PauseRequested(str(_pflag))

                if step >= total_steps:
                    stop = True
                    break
            if stop:
                break

        # Write the PORTABLE ARTIFACT. (The old final eval used to sit AFTER this save
        # and wedge the run at the finish line -- probe A, 2026-09-26. There is no
        # in-process eval anymore: score adapters.pt / ckpt/stepNNNNNN.pt afterwards
        # with scripts/eval_sweep.py, in its own process.)
        save_checkpoint(adapter_path, model, opt, step, epoch, cfg, lean=True)
        print(f"  [artifact] adapters.pt written at step {step} "
              f"({adapter_path.stat().st_size / 2**20:.1f} MiB, adapters only)")
        print(f"  [done] training finished at step {step}. No in-run evals by design --")
        print(f"         score the checkpoints now:  python scripts/eval_sweep.py --run-dir {run_dir}")
    except _PauseRequested as _p:
        # Graceful pause: full checkpoint + portable artifact + a NAME-carrying versioned
        # copy, so a paused run is immediately scoreable and resumable. Exit code 0: this
        # is a request, not a failure. The flag is retired to "<name>.done" so that
        # resuming does not pause again on the same file (delete that to re-arm).
        _flag = Path(_p.args[0]) if _p.args else None
        print(f"\n[pause] flag {_flag} found -- saving at step {step} and exiting gracefully")
        try:
            save_checkpoint(ckpt_path, model, opt, step, epoch, cfg)
            save_checkpoint(adapter_path, model, opt, step, epoch, cfg, lean=True)
            print(f"[pause] checkpoint.pt + adapters.pt saved at step {step}")
            if cfg.versioned_checkpoints:
                vdir = run_dir / "ckpt"
                vdir.mkdir(exist_ok=True)
                vpath = vdir / f"step{step:06d}.pt"
                if not vpath.exists():
                    shutil.copy2(ckpt_path, vpath)
                    print(f"[pause] named copy {vpath.name}")
            print(f"[pause] resume with:  --resume {ckpt_path}")
        except Exception as se:
            print(f"[pause] WARNING: save failed: {type(se).__name__}: {se}")
        if _flag is not None:
            try:
                done = _flag.parent / (_flag.name + ".done")
                if done.exists():
                    done.unlink()
                _flag.rename(done)
                print(f"[pause] flag retired -> {done}  (delete it to re-arm)")
            except Exception as fe:
                print(f"[pause] WARNING: could not retire the flag ({fe}); "
                      f"it will pause again immediately on resume")
        return run_dir
    except (KeyboardInterrupt, SystemExit):
        # Best-effort save so an interactive stop is not a lost run. A hard kill
        # (taskkill on Windows) never reaches here -- that is what save_every is for.
        print(f"\n[interrupt] saving checkpoint at step {step} ...")
        try:
            save_checkpoint(ckpt_path, model, opt, step, epoch, cfg)
            print(f"[interrupt] resumable via --resume {ckpt_path}")
        except Exception as se:
            # Must not mask the interrupt with a save error.
            print(f"[interrupt] WARNING: checkpoint save failed: {type(se).__name__}: {se}")
        steps_fh.close()
        raise
    except BaseException as e:
        # Any other failure -- most plausibly a data/disk error mid-run -- must
        # still leave something usable rather than throwing away hours of compute.
        print(f"\n[error] run failed at step {step}: {type(e).__name__}: {e}")
        try:
            save_checkpoint(ckpt_path, model, opt, step, epoch, cfg)
            save_checkpoint(adapter_path, model, opt, step, epoch, cfg, lean=True)
            print(f"[error] recovered checkpoint.pt + adapters.pt at step {step} "
                  f"-- resume with --resume {ckpt_path}")
        except Exception as se:
            print(f"[error] recovery save ALSO failed: {type(se).__name__}: {se}")
        raise
    finally:
        steps_fh.close()

    # Final re-save marked final=True (metrics key stays in the schema as None: evals
    # no longer happen in-run, so there is nothing to enrich with). Best-effort: the
    # artifacts already exist.
    try:
        save_checkpoint(ckpt_path, model, opt, step, epoch, cfg, metrics=None, final=True)
        save_checkpoint(adapter_path, model, opt, step, epoch, cfg, metrics=None,
                        final=True, lean=True)
        if cfg.versioned_checkpoints:
            vdir = run_dir / "ckpt"
            vdir.mkdir(exist_ok=True)
            vfinal = vdir / f"step{step:06d}.pt"
            if not vfinal.exists():
                shutil.copy2(ckpt_path, vfinal)
        print(f"saved   : {adapter_path}  ({adapter_path.stat().st_size / 2**20:.1f} MiB)")
        print(f"saved   : {ckpt_path}  (resumable)")
    except Exception as e:
        print(f"[warn] final metadata save failed ({type(e).__name__}: {e}); the "
              f"adapters.pt and checkpoint.pt written earlier are intact.")
    return run_dir


def _field_types() -> dict:
    """Resolve TrainConfig annotations, unwrapping Optional[X] to X.

    Needed because the dataclass defaults for grad_clip / nesterov_coef are None, so a
    type cannot be inferred from the default -- and without it argparse hands back a
    STRING. `--nesterov-coef 0.5` then reaches Tensor.lerp_(buf, '0.5') and raises at the
    very first optimizer step.
    """
    import typing

    out: dict = {}
    for name, ann in typing.get_type_hints(TrainConfig).items():
        if ann in (int, float, str, bool):
            out[name] = ann
            continue
        args = [a for a in getattr(ann, "__args__", ()) if a is not type(None)]
        if len(args) == 1 and args[0] in (int, float, str, bool):
            out[name] = args[0]
    return out


def parse_args(argv=None, base: dict | None = None) -> TrainConfig:
    """CLI parser generated from TrainConfig, with an optional YAML/JSON overlay.

    Config resolution order, lowest priority first:
        TrainConfig defaults  <  --config-file  <  explicit CLI flags
    """
    d = asdict(TrainConfig())
    if base:
        unknown = set(base) - set(d)
        if unknown:
            raise SystemExit(f"unknown key(s) in config file: {sorted(unknown)}")
        d.update(base)

    types = _field_types()
    ap = argparse.ArgumentParser(description="Train DreamSim-style metric with OFTv2 + SinkSGD_adv")
    ap.add_argument("--config-file", default=None,
                    help="YAML or JSON file of TrainConfig keys, overridden by CLI flags")
    for f, v in d.items():
        flag = f"--{f.replace('_', '-')}"
        if f == "backbones":
            ap.add_argument(flag, nargs="*", default=list(v))
        elif f == "extra":
            continue
        elif f == "readout_layers":
            # tuple-typed field; take a comma/space-separated STRING ("4,8,12") so
            # argparse's type conversion cannot shred it into characters. The default
            # honours a config-file value (list/tuple) by stringifying it.
            ap.add_argument(flag, type=str,
                            default=",".join(str(x) for x in v) if v else "",
                            help="readout layer indices, e.g. 4,8,12; empty = single")
        elif isinstance(v, bool):
            # BooleanOptionalAction so a default-True flag can still be turned OFF from
            # the CLI (--no-oft-scaled). Plain store_true made A/B testing impossible
            # without editing the YAML.
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        elif v is None:
            ap.add_argument(flag, type=types.get(f, str), default=None)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    a = ap.parse_args(argv)
    kw = vars(a)
    kw.pop("config_file", None)
    if not kw.get("backbones"):
        print(f"[cli] --backbones was empty; using the default {list(TrainConfig.backbones)}")
        kw["backbones"] = list(TrainConfig.backbones)
    kw["backbones"] = tuple(kw["backbones"])
    # readout_layers arrives from the CLI as a string ("4,8,12") and from JSON as a
    # list; TrainConfig wants a tuple of ints. () keeps the legacy single-readout head.
    rl = kw.get("readout_layers")
    if isinstance(rl, str):
        kw["readout_layers"] = tuple(int(x) for x in rl.replace(",", " ").split())
    elif isinstance(rl, list):
        kw["readout_layers"] = tuple(int(x) for x in rl)
    return TrainConfig(**kw)


def load_config_file(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise SystemExit("PyYAML is required for .yaml config files; use JSON instead") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    base = {}
    # Pre-scan for --config-file so its values can act as parser defaults.
    for i, a in enumerate(argv):
        if a == "--config-file" and i + 1 < len(argv):
            base = load_config_file(argv[i + 1])
            print(f"[config] loaded {argv[i + 1]}: {len(base)} key(s)")
        elif a.startswith("--config-file="):
            base = load_config_file(a.split("=", 1)[1])
            print(f"[config] loaded {a.split('=', 1)[1]}: {len(base)} key(s)")
    cfg = parse_args(argv, base=base)
    train(cfg)


if __name__ == "__main__":
    from .logutil import run as _run

    raise SystemExit(_run("train", main, sys.argv[1:]))
