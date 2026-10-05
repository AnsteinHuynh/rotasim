"""Block-diagonal orthogonal rotation adapters (OFT) for frozen towers.

Implements the OFT adapter family after the papers:

  * OFT -- "Orthogonal Finetuning: A General Post-Training Method for
    Language and Vision Models", NeurIPS 2023 (arXiv:2306.07280).
  * OFTv2 -- "Controlling Text-to-Image Diffusion by Orthogonal Finetuning",
    CVPR 2024 (arXiv:2312.09266): block-diagonal rotations + the Neumann-series
    Cayley approximation that makes long contexts affordable.

Each adapted nn.Linear keeps its frozen weight W and learns a skew parameter
vector theta packing the strict upper triangle of a block-diagonal
skew-symmetric matrix Q. The Cayley transform

    R = (I + Q)^-1 (I - Q)

maps Q to an orthogonal R, so the layer computes y = W (R x): a pure rotation
of the input that cannot change any weight norm. theta is zero-initialised and
Cayley(0) = I, so an untrained adapter is an exact identity.

The SOFT scaling variant ("Scaled OFT", design credited to Koratahiu's
OneTrainer PR #1315) divides theta by 2*sqrt(block_size - 1) before the
transform, which keeps the effective rotation magnitude in a block-size
independent range.

Checklist of the load-bearing contracts in here -- do not break casually:

  * state_dict shape: the parameter is a flat (r, n_elements) bag named
    `oft_R.weight` (n_elements = block_size * (block_size - 1) / 2); a
    persistent `scaled_oft` marker buffer appears iff oft_scaled. Checkpoints
    on disk depend on these exact names.
  * numerics run in an fp32 island with autocast suspended: the inverse inside
    the Cayley transform and the rotation einsum are the two places where a
    silent bf16 downcast destabilises training.
  * adv_optm compatibility: parameter names ending in `oft_R.weight` /
    `dora_log_multiplier` are what tagging.py keys off (see that file), so the
    child module and parameter names here are API, not style.
  * the rotation cache: R depends only on the weights, which are frozen between
    optimizer steps, but forward() runs once per image. See OFTRotationModule.

usage:
    from dreamsim_oft.oft import inject_oft
    stats = inject_oft(model, suffixes=("q_proj", "k_proj", "v_proj"), block_size=32)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cayley-transform primitives (free functions so the stats probes can reuse
# them without instantiating a module).
# ---------------------------------------------------------------------------

def _packed_skew(theta: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                 width: int) -> torch.Tensor:
    """Expand packed strict-upper-triangle coordinates into skew-symmetric Q.

    theta has shape (batch, n_elements); the result is (batch, width, width)
    with Q[:, i, j] = theta[k] at every upper-triangle coordinate and the
    lower triangle the negated mirror, so Q + Q^T = 0.
    """
    batch = theta.shape[0]
    out = torch.zeros(batch, width, width, device=theta.device, dtype=theta.dtype)
    # A plain advanced-indexing assignment triggers a pytorch indexing bug
    # (github.com/pytorch/pytorch/issues/169179); index_put with explicit
    # batch indices sidesteps it.
    batch_idx = torch.arange(batch, device=theta.device).unsqueeze(1)
    out = out.index_put((batch_idx, rows, cols), theta)
    return out - out.transpose(-2, -1)


def _neumann_series_alphas(terms: int) -> list[float]:
    """Coefficients of the powers Q^1..Q^m kept by the truncated Neumann form.

    (I + Q)^-1 (I - Q) expands to I - 2Q + 2Q^2 - 2Q^3 + ..., so the truncated
    approximation is I plus 2*Q^k for the middle powers and a single
    coefficient-1 closing term. The exact set of powers retained for a given
    `terms` is kept bit-compatible with the historical implementation because
    trained checkpoints were produced under it.
    """
    if terms <= 1:
        return []
    if terms == 2:
        return [2.0]
    # 2*Q, 2*Q^2, then 2*Q^3 .. 2*Q^(terms-2), closing Q^(terms-1)
    return [2.0, 2.0] + [2.0] * max(0, terms - 4) + [1.0]


def _neumann_cayley(q: torch.Tensor, terms: int) -> torch.Tensor:
    """Cayley transform via the truncated Neumann approximation of (I+Q)^-1."""
    batch, width = q.shape[0], q.shape[-1]
    r = torch.eye(width, device=q.device, dtype=q.dtype).repeat(batch, 1, 1)
    power = q
    for i, alpha in enumerate(_neumann_series_alphas(terms)):
        if i > 0:
            power = torch.bmm(power, q)
        r.add_(power, alpha=alpha)
    return r


def _exact_cayley(q: torch.Tensor) -> torch.Tensor:
    """Cayley transform via a direct linear solve (slower, reference-grade)."""
    batch, width = q.shape[0], q.shape[-1]
    ident = torch.eye(width, device=q.device).unsqueeze(0).expand(batch, width, width)
    return torch.linalg.solve(ident + q, ident - q, left=False)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class MultiplicativeDropoutLayer(nn.Module):
    """Whole-block identity dropout for the rotation tensors (default off).

    While training, each block in the batched rotation is independently swapped
    for the identity matrix with probability p; at evaluation (or p == 0) the
    input passes through untouched. Applied to R, not to activations.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.p > 0.0):
            return blocks
        n, h, w = blocks.shape
        if h != w:
            raise ValueError(f"block-dropout expects square blocks, got {h}x{w}")
        if n == 1:
            return blocks
        keep = torch.empty(n, 1, 1, device=blocks.device, dtype=blocks.dtype)
        keep.bernoulli_(1.0 - self.p)
        ident = torch.eye(h, device=blocks.device, dtype=blocks.dtype).repeat(n, 1, 1)
        return keep * blocks + (1.0 - keep) * ident


class OFTRotationModule(nn.Module):
    """Learns `r` independent block-diagonal rotations of size `block_size`.

    Owns the flat skew parameter, the Cayley pipeline, the (optional) SOFT
    pre-scaling and the rotation cache. Injected as the `oft_R` child of
    OFTLinear below.
    """

    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        block_share: bool = False,
        oft_scaled: bool = False,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(r, n_elements))
        self.block_share = block_share
        if oft_scaled:
            # Persistent marker so inference tools can detect the scaled
            # variant from the state_dict alone.
            self.register_buffer("scaled_oft", torch.tensor(True))
        self.oft_scaled = oft_scaled
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms

        # Coordinates of the strict upper triangle, used to scatter the packed
        # parameter into Q. Non-persistent: derivable from block_size.
        upper = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", upper[0], persistent=False)
        self.register_buffer("cols", upper[1], persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)
        # ROTATION CACHE (2026-10-04, launch-latency fix): R depends only on the
        # weights, which are constant between optimizer steps -- but forward() is
        # called once per image per chunked Qwen batch (~80x/step), each call
        # rebuilding the whole Cayley pipeline for all 108 modules (~10k redundant
        # small-kernel chains/step = the GPU-idle signature). The cache is valid
        # until invalidate_rotation_caches() is called by the trainer AFTER each
        # backward; the weight._version check is a belt-and-braces second
        # invalidation. Gradients are exact: every cached forward references the
        # SAME graph node, so one backward call fans the accumulated gradient back
        # through the rotation once -- the linear sum that N separate rebuilds
        # would produce. Stochastic OFT-dropout (p>0 while training) is never
        # cached (it must resample per forward).
        self._R_cache: torch.Tensor | None = None
        self._R_cache_ver = -1

    # -- rotation construction ------------------------------------------------

    def soft_scale(self) -> float:
        """Pre-Cayley divisor applied to theta under the scaled (SOFT) variant."""
        return 2.0 * math.sqrt(self.block_size - 1) if self.oft_scaled else 1.0

    def orthogonal_from(self, theta: torch.Tensor) -> torch.Tensor:
        """Rotation blocks for packed skew parameters `theta` (no SOFT scaling,
        no dropout) under this module's transform settings.

        Exposed for the drift stats in this file; the forward path goes through
        `_rotation`, which adds the scaling, dropout and the cache.
        """
        q = _packed_skew(theta, self.rows, self.cols, self.block_size)
        if self.use_cayley_neumann:
            return _neumann_cayley(q, self.num_cayley_neumann_terms)
        return _exact_cayley(q)

    def _rotation(self) -> torch.Tensor:
        """Current rotation tensor (SOFT-scaled, dropout-applied, cached).

        The cache is what keeps chunked multi-image forwards cheap -- see the
        ROTATION CACHE note in __init__.
        """
        trainable = self.weight.requires_grad and torch.is_grad_enabled()
        cache_ok = (self._R_cache is not None
                    # a cache stored under no_grad is DETACHED -- reusing it in a
                    # grad-enabled forward would silently drop the rotation gradient
                    and (not trainable or self._R_cache.requires_grad)
                    and self._R_cache_ver == self.weight._version
                    and not (self.dropout.training and self.dropout.p > 0))
        if cache_ok:
            return self._R_cache
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            theta = self.weight.float() / self.soft_scale()
            r = self.orthogonal_from(theta)
            r = self.dropout(r)
        if trainable and not (self.dropout.training and self.dropout.p > 0):
            self._R_cache = r
            self._R_cache_ver = self.weight._version
        return r

    # -- forward ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        incoming_dtype = x.dtype
        if incoming_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        shape_in = x.shape
        # fp32 island: suspending autocast is what actually pins the Cayley solve
        # and the rotation einsum to fp32 under bf16 training -- casting the
        # tensors alone is not enough, the autocast policy would downcast the
        # matmuls right back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            blocks = _cached_rotation(self)
            n_blocks = self.in_features // self.block_size if self.block_share else self.r
            grouped = xf.reshape(*xf.shape[:-1], n_blocks, self.block_size)
            if self.block_share:
                blocks = blocks.repeat(n_blocks, 1, 1)
            rotated = torch.einsum("...bk,bkc->...bc", grouped, blocks)
        return rotated.reshape(*shape_in).to(incoming_dtype)


@torch._dynamo.disable(recursive=True)
def _cached_rotation(m: "OFTRotationModule") -> torch.Tensor:
    """Fetch R through the cache, OUTSIDE any torch.compile graph. Inlining this
    branch into dynamo would either bake the stale path in or force a recompile
    every step; a disabled region keeps the graph guards stable and costs one
    python call per module per forward (microseconds)."""
    return m._rotation()


def precompute_rotations(root: nn.Module) -> None:
    """Build every rotation ONCE, OUTSIDE the checkpointed tower blocks. If the
    first Cayley pipeline ran INSIDE a torch.utils.checkpoint region, its
    original forward and the warm-cache recompute would save a different number
    of tensors and the backward would abort with CheckpointError (caught live
    2026-10-04: 'saved 62 vs recomputed 34'). The trainer calls this before each
    forward group; backward then traverses the shared graph exactly once."""
    for m in root.modules():
        if isinstance(m, OFTRotationModule):
            _cached_rotation(m)


def invalidate_rotation_caches(root: nn.Module) -> None:
    """Drop every OFT rotation cache (trainer calls this after EACH backward; see
    OFTRotationModule._rotation). Without it, a second backward through the shared
    cached graph would raise 'trying to backward a second time'."""
    for m in root.modules():
        if isinstance(m, OFTRotationModule):
            m._R_cache = None
            m._R_cache_ver = -1


def _snap_block_size(in_features: int, requested: int) -> int:
    """Snap `requested` to the nearest divisor of in_features (ties -> smaller).

    The rotation blocks must tile the input width exactly, so a non-divisor
    request falls back to the closest size that divides cleanly.
    """
    if in_features % requested == 0 and requested <= in_features:
        return requested
    if requested >= in_features:
        return in_features
    up = next(d for d in range(requested, in_features + 1) if in_features % d == 0)
    down = next(d for d in range(requested, 0, -1) if in_features % d == 0)
    return down if (requested - down) <= (up - requested) else up


class OFTLinear(nn.Module):
    """Wraps a frozen nn.Linear with an OFT input rotation.

    y = W (R x): rotating the INPUT is mathematically the same as rotating W's
    rows but keeps the frozen weight untouched, which is the whole point -- the
    released artifact is the small `oft_R` bag plus the frozen tower.

    The child is deliberately named `oft_R` (state-dict and adv_optm-tagging
    contract; see the module docstring).
    """

    def __init__(self, orig_module: nn.Linear, block_size: int = 32, block_share: bool = False,
                 oft_scaled: bool = False, dropout_probability: float = 0.0,
                 use_cayley_neumann: bool = True, num_cayley_neumann_terms: int = 5,
                 adjustment_log: list | None = None):
        super().__init__()
        if not isinstance(orig_module, nn.Linear):
            raise NotImplementedError(f"OFTLinear only supports nn.Linear, got {type(orig_module).__name__}")
        if block_size <= 1:
            raise ValueError(
                f"block_size must be >= 2, got {block_size}. At block_size=1 n_elements = "
                f"b(b-1)/2 = 0, so the adapter would have ZERO parameters and its forward "
                f"would be an exact no-op -- while still passing inject_oft's "
                f"'wrapped N layers' guard and the 'some param requires grad' check. "
                f"Silent no-op training is the worst possible failure here.")

        in_features = orig_module.in_features
        self.orig_module = orig_module
        for p in self.orig_module.parameters():
            p.requires_grad_(False)

        adj = _snap_block_size(in_features, block_size)
        if adj != block_size and adjustment_log is not None:
            adjustment_log.append({"in_features": in_features, "requested": block_size, "used": adj})

        self.block_size = adj
        self.block_share = block_share
        self.oft_scaled = oft_scaled
        self.rank = 1 if block_share else in_features // adj
        n_elements = adj * (adj - 1) // 2

        self.oft_R = OFTRotationModule(
            r=self.rank,
            n_elements=n_elements,
            block_size=adj,
            in_features=in_features,
            block_share=block_share,
            oft_scaled=oft_scaled,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
            dropout_probability=dropout_probability,
        )
        # Untrained = exact identity: Cayley(0) = I, so theta starts at zero.
        nn.init.zeros_(self.oft_R.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig_module(self.oft_R(x))

    @property
    def trainable_parameters(self) -> int:
        return self.oft_R.weight.numel()


class DoRAOFTLinear(OFTLinear):
    """OFT rotation plus a learned per-output-channel magnitude.

    A DoRA-style weight decomposition specialised to orthogonal adapters
    (design after Koratahiu's DOFT proposal, OneTrainer PR #1335; reimplemented
    here from the formula). Because R is orthogonal it preserves every row norm
    of the frozen W, so DoRA's renormalisation collapses to a diagonal rescale:

        y = m * (W (R x) + b) + b * (1 - m),     m = exp(dora_log_multiplier)

    Two choices are load-bearing:
      * ZERO-INIT: m = exp(0) = 1, so step 0 is an exact identity.
      * LOG-PARAMETERISED: the multiplier learns ratios with a uniform step
        size; a linear parameterisation moves at the wrong rate relative to
        the rotation blocks. Do not switch it back.
    """

    def __init__(self, orig_module: nn.Linear, **kwargs):
        super().__init__(orig_module, **kwargs)
        # Match the rotation's dtype (oft_weight_dtype pins that to fp32).
        self.dora_log_multiplier = nn.Parameter(
            torch.zeros(orig_module.out_features, dtype=self.oft_R.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)                       # W (R x) + b
        m = torch.exp(self.dora_log_multiplier).to(out.dtype)
        if self.orig_module.bias is not None:
            b = self.orig_module.bias
            return out * m + b.to(out.dtype) * (1.0 - m)
        return out * m

    @property
    def trainable_parameters(self) -> int:
        return self.oft_R.weight.numel() + self.dora_log_multiplier.numel()


def _get_parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def inject_oft(model: nn.Module, suffixes=("q_proj", "k_proj", "v_proj"), block_size: int = 32,
               block_share: bool = False, oft_scaled: bool = False, dropout_probability: float = 0.0,
               use_cayley_neumann: bool = True, num_cayley_neumann_terms: int = 5,
               include_prefix: str | None = None, verbose: bool = True,
               dora_oft: bool = False) -> dict:
    """Replace every nn.Linear whose name ends in one of `suffixes` with an OFTLinear.

    Returns a stats dict: how many were wrapped, where, and the trainable cost.
    Refuses to return silently having wrapped nothing -- that failure mode would
    look like training a completely frozen model.
    """
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not name.split(".")[-1] in suffixes:
            continue
        if include_prefix and include_prefix not in name:
            continue
        targets.append(name)

    if not targets:
        raise RuntimeError(
            f"inject_oft found NO nn.Linear matching suffixes={suffixes}"
            + (f" under prefix {include_prefix!r}" if include_prefix else "")
            + ". Refusing to continue, since this would train an entirely frozen model."
        )

    adjustments: list = []
    cls = DoRAOFTLinear if dora_oft else OFTLinear
    for name in targets:
        parent, leaf = _get_parent(model, name)
        orig = getattr(parent, leaf)
        setattr(parent, leaf, cls(
            orig, block_size=block_size, block_share=block_share, oft_scaled=oft_scaled,
            dropout_probability=dropout_probability, use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms, adjustment_log=adjustments))

    n_train = sum(p.numel() for n, p in model.named_parameters() if n.endswith("oft_R.weight"))
    n_mag = sum(p.numel() for n, p in model.named_parameters()
                if n.endswith("dora_log_multiplier"))
    stats = {
        "wrapped": len(targets),
        "suffixes": list(suffixes),
        "block_size_requested": block_size,
        "block_size_adjustments": adjustments,
        "oft_trainable_params": n_train,
        "doft_magnitude_params": n_mag,
        "example_names": targets[:3],
    }
    if verbose:
        print(f"[oft] wrapped {len(targets)} Linear layers with "
              f"{'DoRA-OFT (DOFT)' if dora_oft else 'OFTv2'} "
              f"(block_size={block_size}, block_share={block_share}, scaled={oft_scaled}, "
              f"dropout={dropout_probability})")
        if adjustments:
            print(f"[oft] NOTE: block_size auto-adjusted on {len(adjustments)} layer(s): "
                  f"{adjustments[:4]}{' ...' if len(adjustments) > 4 else ''}")
        print(f"[oft] trainable OFT params: {n_train:,}")
        if dora_oft:
            print(f"[doft] magnitude params: {n_mag:,} (zero-init dora_log_multiplier, "
                  f"exp(0)=1 => identity at step 0)")
        for n in targets[:3]:
            print(f"        e.g. {n}")
    return stats


def oft_rotation_stats(model: nn.Module) -> dict:
    """Drift of the rotation that is ACTUALLY APPLIED in forward, plus the raw weight drift.

    Why this is not simply "max |R - I| on the raw weights": with `oft_scaled` (SOFT) the
    forward divides the skew weights by 2*sqrt(block_size-1) BEFORE the Cayley transform
    (see OFTRotationModule.soft_scale). Measuring Cayley(raw weight) therefore overstates
    the applied rotation by exactly that factor -- 11.1355x at block_size 32. Since this
    number is the run's primary "are the adapters moving" signal, an 11x overstatement
    would make a barely-moving model look healthy. Reported both, but `applied` is the
    one to watch.

    Returns {"applied": float, "raw": float, "scaled_layers": int, "scale_factor": float}
    """
    applied = 0.0
    raw = 0.0
    orth_err = 0.0
    n_scaled = 0
    scale_factor = 1.0

    with torch.no_grad():
        for _, mod in model.named_modules():
            if not isinstance(mod, OFTLinear):
                continue
            w = mod.oft_R.weight
            bs = mod.oft_R.block_size
            eye = torch.eye(bs, device=w.device, dtype=w.dtype)

            r_raw = mod.oft_R.orthogonal_from(w)
            raw = max(raw, (r_raw - eye.unsqueeze(0)).abs().max().item())

            if mod.oft_R.oft_scaled:
                scale_factor = 2 * math.sqrt(bs - 1)
                r_app = mod.oft_R.orthogonal_from(w / scale_factor)
                n_scaled += 1
            else:
                r_app = r_raw

            applied = max(applied, (r_app - eye.unsqueeze(0)).abs().max().item())
            # Orthogonality budget: ||R^T R - I||. The truncated Neumann series is only
            # 4th-order accurate, so R stops being meaningfully orthogonal as ||Q|| grows.
            # Worth watching across 5,214 steps -- it should stay near 0.
            rt_r = torch.bmm(r_app.transpose(1, 2), r_app)
            orth_err = max(orth_err, (rt_r - eye.unsqueeze(0)).abs().max().item())

    return {"applied": applied, "raw": raw, "orthogonality_err": orth_err,
            "scaled_layers": n_scaled, "scale_factor": scale_factor}


def oft_identity_error(model: nn.Module) -> float:
    """Max |R - I| for the APPLIED rotation. ~0 before any optimizer step."""
    return oft_rotation_stats(model)["applied"]
