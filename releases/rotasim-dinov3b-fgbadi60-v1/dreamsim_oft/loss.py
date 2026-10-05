"""
[Begin Work Zone]
Hinge / triplet ranking loss, ported from DreamSim's util/train_utils.py.

The task is 2AFC: given a reference x and two distortions (x0, x1), a human said one
is more similar to the reference. The loss pushes the human-preferred distortion to be
closer. With d0 = D(x, x0), d1 = D(x, x1) and x = d0 - d1, the target maps
{0, 1} -> {-1, +1} and

    L = max(0, margin - x * y_transformed)

which is a triplet loss with margin `margin` on the distance difference. It is a
RANKING loss: only the sign/ordering of distances is supervised, never their absolute
scale. That matters when pairing it with an unusual optimizer -- the loss is flat
whenever the margin is already satisfied, and an entire batch can sit at exactly zero
gradient.

DreamSim's original sums over the batch and then divides by batch size in the training
step. We keep that behaviour for parity, and expose `reduction` for mean/sum.

usage:
    from dreamsim_oft.loss import HingeLoss
    crit = HingeLoss(margin=0.05, device="cuda")
    loss = crit(dist0 - dist1, target)
[End Work Zone]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self, margin: float = 0.05, device: str | torch.device = "cuda",
                 reduction: str = "sum"):
        super().__init__()
        self.margin = margin
        self.device = device
        self.reduction = reduction

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logit = d0 - d1 ; target in [0,1] where >=0.5 means "x1 is more similar"."""
        y_rounded = torch.round(target)
        y_transformed = -1 * (1 - 2 * y_rounded)          # {0,1} -> {-1,+1}
        raw = torch.clamp(self.margin - logit * y_transformed, min=0.0)
        if self.reduction == "sum":
            return raw.sum()
        if self.reduction == "mean":
            return raw.mean()
        return raw


def two_afc_accuracy(dist0: torch.Tensor, dist1: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Count of triplets where the closer distortion matches the human vote.

    Mirrors DreamSim: `decisions = dist1 < dist0`, correct when target >= 0.5.
    TIES (target == 0.5) are EXCLUDED from both numerator and denominator: a tie
    expresses no preference, so neither answer can be "correct". FGResQ carries
    ~1.8% ties; NIGHTS has none, so this is a no-op there.
    Returned as a tensor so the caller can accumulate across batches.
    """
    decisions = torch.lt(dist1, dist0)
    mask = target != 0.5
    if mask.all():
        return ((target >= 0.5) == decisions).sum()
    if not mask.any():
        return torch.zeros((), device=dist0.device)
    return (((target >= 0.5) == decisions) & mask).sum()


def two_afc_scored(dist0: torch.Tensor, dist1: torch.Tensor,
                   target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(correct, scored) with ties EXCLUDED from both. Use this for honest accuracies:
    the caller divides correct by scored, so tie rows neither help nor dilute."""
    decisions = torch.lt(dist1, dist0)
    mask = target != 0.5
    correct = (((target >= 0.5) == decisions) & mask).sum()
    return correct, mask.sum()


class BradleyTerryLoss(nn.Module):
    """Logistic / Bradley-Terry loss on the 2AFC preference (the modern replacement).

        L = sum[ -y*log s(z) - (1-y)*log(1-s(z)) ],   z = (d0 - d1) / tau

    Why this beats the hinge:
      * The hinge is a RANKING loss with a FLAT region -- once the margin is satisfied
        its gradient is EXACTLY zero. At ~91% train accuracy roughly 91% of pairs
        contribute nothing and the objective goes silent. BT keeps a gradient on every
        pair, which is what you want when the model is still underfitting.
      * No magic margin. The margin's job (setting the scale of z at which a comparison
        counts as settled) is taken over by tau, and BT is the standard objective for
        human-preference pairs (it is what reward modelling and DPO are built on).

    tau is FIXED, deliberately NOT learned. SinkSGD_adv normalises update magnitude to
    unit RMS via Sinkhorn iterations, so the optimizer discards a global loss scale --
    exactly the quantity a learned tau most directly controls. A learned tau would be
    near-unidentifiable, and would add optimizer state that has to be checkpointed and
    locked across resumes, for no measurable gain. A fixed tau gives the same relative
    per-example reweighting for free.

    Note on soft labels: this loss ACCEPTS y in [0,1], but HingeLoss above does
    `torch.round(target)`, so any soft target handed to the hinge is silently collapsed
    to 0/1. Soft targets only mean anything on the BT path.
    """

    def __init__(self, tau: float = 0.05, device: str | torch.device = "cuda",
                 reduction: str = "sum", confidence_weight: bool = False):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau
        self.device = device
        self.reduction = reduction
        # VOTE-CONFIDENCE WEIGHTING (D2 post-mortem arm A): per-triplet weight
        # w = 2*|y - 0.5| in [0,1]. A decisive human label (y in {0,1}) keeps full
        # weight; a TIE (y = 0.5, FGResQ only) contributes EXACTLY ZERO loss, so it
        # no longer pushes d0 == d1. On binary-target data (NIGHTS) this is a strict
        # no-op (every w == 1). Rationale: the audited signal analysis says usable
        # supervision concentrates in confident pairs; ties are "no preference", and
        # a 2AFC metric should not spend gradient equalising them.
        self.confidence_weight = confidence_weight

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logit = d0 - d1 ; target in [0,1], >=0.5 meaning "x1 is more similar"."""
        z = logit / self.tau
        y = target
        # Numerically stable BCE-with-logits: max(z,0) - z*y + log1p(exp(-|z|)).
        raw = torch.clamp(z, min=0.0) - z * y + torch.log1p(torch.exp(-z.abs()))
        if self.confidence_weight:
            raw = raw * (2.0 * (y - 0.5).abs())
        if self.reduction == "sum":
            return raw.sum()
        if self.reduction == "mean":
            return raw.mean()
        return raw


class BCProbitLoss(nn.Module):
    """Thurstone (probit) pair model matched to soft labels by the Bhattacharyya coefficient.

    This is the objective used by AFINE (CVPR 2025) for the DiffIQA 3AFC/2AFC split, verified
    in their source on 2026-10-03:

        TrainAFINE/basicsr/models/afine_stage2_model.py
            arg12   = (u2ref - u1ref) / sqrt(2)      # Thurstone Case V
            p_hat12 = Normal(0,1).cdf(arg12)
        TrainAFINE/basicsr/losses/basic_loss.py  (class FidelityLoss)
            loss = 1 - (sqrt(p*g + eps) + sqrt((1-p)*(1-g) + eps)),   g in {0, 0.5, 1}

    We apply the same two formulas to OUR margin, z = (d0 - d1) / tau, i.e. we substitute
    "quality difference over sqrt(2)" with "distance difference over tau". The two are the
    same one-parameter sigmoid family:

        p = Phi(z)  vs  BT's  p = sigma(z)

    CALIBRATION CAVEAT (measured, scratch\\sophie\\bench_champs\\loss_link_calibration.py):
    their sqrt(2) is only meaningful because their u is a BOUNDED, normalised quality axis
    (u in ~[-1,1] => p in [0.08, 0.92]); on our uncalibrated cosine distance the scale must
    be FIRED, and the MLE-optimal scale is corpus-dependent (0.0902 FGResQ, 0.0319 NIGHTS
    against our fixed tau = 0.05). So `tau` here is the same knob as BT's, and an arm that
    changes the loss at a FIXED tau isolates the loss FORM, not the scale.

    Chance value: at z = 0, p = 0.5 and L = 1 - sqrt(0.5) = 0.2929 (vs BCE's ln2 = 0.6931),
    so a BC-probit log is NOT comparable to a BT log -- read the first block accordingly.
    """

    def __init__(self, tau: float = 0.05, device: str | torch.device = "cuda",
                 reduction: str = "sum", confidence_weight: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau
        self.device = device
        self.reduction = reduction
        self.confidence_weight = confidence_weight
        self.eps = eps

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logit = d0 - d1 ; target in [0,1], >=0.5 meaning "x1 is more similar"."""
        import math
        z = logit / self.tau
        p = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        y = target
        bc = (torch.sqrt(p * y + self.eps)
              + torch.sqrt((1.0 - p) * (1.0 - y) + self.eps))
        raw = 1.0 - bc
        if self.confidence_weight:
            raw = raw * (2.0 * (y - 0.5).abs())
        if self.reduction == "sum":
            return raw.sum()
        if self.reduction == "mean":
            return raw.mean()
        return raw


def build_criterion(kind: str, device, reduction: str = "sum", margin: float = 0.05,
                    tau: float = 0.05, confidence_weight: bool = False) -> nn.Module:
    """Select the ranking objective. Kept as one place so train and eval cannot diverge."""
    kind = (kind or "hinge").lower()
    if kind == "hinge":
        return HingeLoss(margin=margin, device=device, reduction=reduction)
    if kind in ("bt", "bradley_terry", "logistic"):
        return BradleyTerryLoss(tau=tau, device=device, reduction=reduction,
                                confidence_weight=confidence_weight)
    if kind in ("bc", "bc_probit", "probit", "thurstone", "fidelity"):
        return BCProbitLoss(tau=tau, device=device, reduction=reduction,
                            confidence_weight=confidence_weight)
    raise ValueError(f"unknown loss_type {kind!r}; expected 'hinge', 'bt' or 'bc_probit'")
