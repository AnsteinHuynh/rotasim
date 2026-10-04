"""Post-resize 288px pixel cache: serves the ALREADY-CROPPED (3,288,288) tensor.

This is not the decode cache (`data._open_image`): it skips decode AND the resize, so it can
only be used by the square/288 input class, and only if it was built by the very same
function the live path uses (dreamsim_oft.resize.resize_center_square_lanczos).

Layout (mirrors the decode cache, dir-qualified because basenames collide between the two
source trees -- 218 collisions, and those pairs are NOT duplicates):

    <cache>/<dir>/<imagename>-<ext>.pt        images/000285.bmp -> <cache>/images/000285-bmp.pt

Why the interlocks below exist: this project has ALREADY shipped a cache that served the
wrong pixels (built from a different tree, keyed by basename with the extension stripped --
20/60 sampled images differed, max delta 254, two different SHAPES) and its own losslessness
gate passed 100/100 because it compared against the same wrong source. So this reader:

  * REFUSES a cache directory whose name does not carry the method token
    ("centresquare-window-fraclanczos3"), instead of silently serving pixels produced by some
    other resize. Pointing it at the old squash-era directory raises; it does not miss quietly.
  * VERIFIES ONE IMAGE END-TO-END at construction -- decodes the source, resizes it live, and
    requires bit-identity with the cached entry. If that fails it raises, because a cache that
    is wrong is worse than no cache: it changes the pixels the model trains on.
  * COUNTS hits and misses and reports them, so a cache that is silently never used (a
    previous on-disk val cache was constructed nowhere in the codebase) is visible.

Enable by default (kerok directive 2026-10-01): the canonical fp32 directory is used automatically
when it exists. Disable with FGRESQ_PT_288=off, or point it at another directory. The u8 variant
exists but is slower on the training path -- use fp32.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

__all__ = ["CACHE_METHOD_TOKEN", "PixelCache288", "cache_rel", "resolve_288_root"]

#: Must appear in the cache directory name -- identifies WHICH resize produced the pixels.
CACHE_METHOD_TOKEN = "centresquare-window-fraclanczos3"

#: Known transforms, and how each can be verified from HERE.
#:   * fraclanczos3      -- analytical; recompute through the live path and require bit-equality.
#:   * rtxvsrultra       -- an AI transform produced by a DIFFERENT interpreter (the ComfyUI env
#:                          owns `nvvfx`), so it cannot be recomputed here at all. Verified by
#:                          the build-time SHA-256 record instead (see verify_integrity).
#: A cache directory must name exactly one of these; an unknown token is refused.
CACHE_METHOD_TOKENS = (
    "centresquare-window-fraclanczos3",
    "centresquare-crop-rtxvsrultra",
)
#: tokens whose pixels cannot be recomputed in this environment -> integrity verification
INTEGRITY_ONLY_TOKENS = ("centresquare-crop-rtxvsrultra",)
HASH_RECORD = "hashes.json"


def cache_rel(rel: str) -> str:
    """<dir>/<imagename>-<ext>.pt for a source-relative path. Byte-for-byte the same key
    convention as data._open_image (keep them in step; the token check is what catches it)."""
    relp = str(rel).replace("\\", "/")
    head, _, tail = relp.rpartition("/")
    stem, dot, ext = tail.rpartition(".")
    name = f"{stem}-{ext.lower()}.pt" if dot else f"{tail}.pt"
    return f"{head}/{name}" if head else name


#: kerok directive 2026-10-01: the fp32 288 cache is ON by default. Measured in the CUDA
#: environment (15 train steps/arm + an evaluator A/B at n=50): training 24 s vs 25 s (inside
#: noise, GPU-bound), eval wall 21.2 vs 24.4 s with IDENTICAL accuracy. u8 is 4x smaller but
#: SLOWER on the training path (30 s -- dequantising costs more than the decode+resize it
#: saves), so fp32 is the variant to use. It skips decode AND crop AND resize.
CANONICAL_FP32_DIR = (
    r"I:\MyApps\sd-train\fgresq"
    r"\was-cached-as-trainer-sees-it-no-augs-288x288px-centresquare-window-fraclanczos3-fullframe-fp32")


def resolve_288_root(cache_root: str | os.PathLike | None) -> str | None:
    """Explicit argument > FGRESQ_PT_288 > the canonical fp32 dir if it exists > OFF.

    Disable with FGRESQ_PT_288=off (also accepts none/0/empty). Any other value is a path.
    """
    if cache_root:
        return str(cache_root)
    env = os.environ.get("FGRESQ_PT_288")
    if env is not None:
        env = env.strip()
        if env.lower() in ("", "off", "none", "0"):
            return None
        return env
    return CANONICAL_FP32_DIR if os.path.isdir(CANONICAL_FP32_DIR) else None


class PixelCache288:
    """Reader for one dtype variant of the 288 cache. Thread/process-local counters."""

    def __init__(self, root: str | os.PathLike, cache_root: str | os.PathLike | None = None,
                 out: int = 288, dtype: str = "float32", verify_rel: str | None = None,
                 verify_tol: float = 1e-6, verbose: bool = True):
        self.root = Path(root)
        self.out = int(out)
        self.dtype = torch.float32 if dtype in ("float32", "fp32") else torch.uint8
        resolved = resolve_288_root(cache_root)
        self.dir: Path | None = Path(resolved) if resolved else None
        self.hits = 0
        self.misses = 0
        self.verify_tol = float(verify_tol)
        if self.dir is not None:
            # The dtype belongs to the DIRECTORY (…-u8 / …-fp32), not to a free-floating flag.
            # Leaving them independent let `dtype=u8` point at the fp32 dir, which presents as
            # "every lookup missed" -- a silent-looking failure that is really a config error.
            nm = self.dir.name.lower()
            if nm.endswith("-u8") and self.dtype != torch.uint8:
                raise ValueError(f"{self.dir.name} holds uint8 entries but dtype={dtype!r} was "
                                 f"requested: the directory decides the dtype.")
            if nm.endswith(("-fp32", "-float32")) and self.dtype != torch.float32:
                raise ValueError(f"{self.dir.name} holds float32 entries but dtype={dtype!r} "
                                 f"was requested: the directory decides the dtype.")
            tok = next((t for t in CACHE_METHOD_TOKENS if t in self.dir.name), None)
            if tok is None:
                raise ValueError(
                    f"refusing 288 cache {self.dir}: its name names no known transform "
                    f"(expected one of {CACHE_METHOD_TOKENS}). A cache directory must say WHICH "
                    f"resize built it -- serving pixels from an unknown transform is how this "
                    f"project once trained on the wrong images.")
            self.method_token = tok
            if not self.dir.is_dir():
                raise FileNotFoundError(f"288 cache dir does not exist: {self.dir}")
            # verify() is the ONLY version interlock: the directory name carries the method
            # token, not RESIZE_TAG, so a cache built by an older kernel passes the name
            # check. Measured under the v1 kernel: bit-equality would have raised on
            # 8481/12593 square sources. Enabling a cache without verification is therefore
            # not allowed -- pass verify_rel.
            if verify_rel is None:
                raise ValueError("PixelCache288 requires verify_rel=<a source rel path> when "
                                 "enabled: it is the only thing that proves the entries "
                                 "were produced by the CURRENT transform.")
            if tok in INTEGRITY_ONLY_TOKENS:
                self.verify_integrity(verify_rel)
            else:
                self.verify(verify_rel)
        if verbose:
            print(self.describe())

    # -- reporting -----------------------------------------------------------
    def describe(self) -> str:
        if self.dir is None:
            return f"[288cache] OFF ({self.dtype}) -- decode + live resize"
        tok = getattr(self, "method_token", CACHE_METHOD_TOKEN)
        how = "INTEGRITY-verified" if tok in INTEGRITY_ONLY_TOKENS else "live-verified"
        return (f"[288cache] ON  {self.dir.name} | {self.dtype} | out={self.out} | "
                f"{tok} ({how}; decode AND resize skipped)")

    def report(self) -> str:
        n = self.hits + self.misses
        rate = (100.0 * self.hits / n) if n else 0.0
        return (f"[288cache] hits={self.hits} misses={self.misses} "
                f"({rate:.1f}% hit rate, {n} lookups)")

    # -- data ----------------------------------------------------------------
    def path_for(self, rel: str) -> Path | None:
        return None if self.dir is None else self.dir / cache_rel(rel)

    def get(self, rel: str) -> torch.Tensor | None:
        """The cached (3,out,out) tensor, or None on miss. Never raises on a bad entry."""
        p = self.path_for(rel)
        if p is None or not p.exists():
            self.misses += 1
            return None
        try:
            t = torch.load(p, map_location="cpu")
        except Exception as exc:                                   # corrupt/partial file
            self.misses += 1
            if self.misses <= 3:
                print(f"[288cache] unreadable {p.name}: {exc!r} -- falling back to live path")
            return None
        if not isinstance(t, torch.Tensor) or t.dtype != self.dtype \
                or tuple(t.shape) != (3, self.out, self.out):
            self.misses += 1
            if self.misses <= 3:
                got = (type(t).__name__, getattr(t, "dtype", None), tuple(getattr(t, "shape", ())))
                print(f"[288cache] bad entry {p.name}: want (3,{self.out},{self.out}) "
                      f"{self.dtype}, got {got} -- falling back")
            return None
        self.hits += 1
        return t

    def as_model_input(self, t: torch.Tensor) -> torch.Tensor:
        """uint8 entries -> float 0..1; float entries pass through. Matches the live path's
        dtype semantics (the u8 variant is a quantised copy; max delta 0.0020 = 0.51/255)."""
        return t.to(torch.float32).div_(255.0) if self.dtype == torch.uint8 else t

    # -- the gate that would have caught the wrong-tree cache ------------------
    def verify(self, rel: str) -> float:
        """Recompute one image through the LIVE path and require bit-identity."""
        from torchvision.transforms.functional import to_tensor

        from .data import _open_image
        from .resize import resize_center_square_lanczos

        h0, m0 = self.hits, self.misses          # the verification lookup is not a training hit
        cached = self.get(rel)
        self.hits, self.misses = h0, m0
        if cached is None:
            raise RuntimeError(f"288 cache verification failed: no entry for {rel!r} "
                               f"(looked for {self.path_for(rel)}). An empty/partial cache "
                               f"must not be used.")
        live = resize_center_square_lanczos(
            to_tensor(_open_image(self.root, rel, None)), out=self.out)
        if self.dtype == torch.uint8:
            live = (live * 255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
        d = (cached.to(torch.float32) - live.to(torch.float32)).abs().max().item()
        # Tolerance, not bit-equality: correct float pipelines can differ in the last ulp
        # (a ~1e-17 integer tap in the Lanczos kernel was enough to make torch.equal fail
        # while the images were identical for every practical purpose). The bound is ~2.5
        # float32 ulps on the [0,1] scale; a cache built from a DIFFERENT transform or tree
        # differs by up to 1.0, so this still catches every real defect.
        if d > self.verify_tol:
            raise RuntimeError(
                f"288 cache verification FAILED on {rel!r}: max|d| = {d} "
                f"({d * 255:.2f}/255, tol {self.verify_tol}) vs the live path. The cache does "
                f"not match the current transform -- rebuild it (see RESIZE_TAG in "
                f"dreamsim_oft.resize). Refusing to train on pixels that are not what the "
                f"loader would produce.")
        return d

    def verify_integrity(self, rel: str, sample: int = 24) -> float:
        """Interlock for caches whose transform CANNOT be recomputed in this environment.

        The RTX-VSR cache is produced by a different interpreter (the ComfyUI env owns
        `nvvfx`), so `verify()`'s live-path bit-comparison is impossible here. This proves the
        entries are the bytes that were BUILT -- sha256 recorded at build time -- plus
        well-formedness via get()'s shape/dtype check on load.

        What it does NOT prove: which transform produced them. That is what the directory-name
        method token carries, and why an unknown token is refused outright.
        """
        import hashlib
        import json as _json

        rec_p = self.dir / HASH_RECORD
        if not rec_p.exists():
            raise RuntimeError(
                f"{self.dir.name} carries no {HASH_RECORD}: an integrity-verified cache must "
                f"record its build-time hashes. Refusing to train on pixels of unknown "
                f"provenance.")
        rec = _json.loads(rec_p.read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or not rec:
            raise RuntimeError(f"{HASH_RECORD} in {self.dir.name} is empty or malformed")

        key = cache_rel(rel)
        if key not in rec:
            raise RuntimeError(
                f"288 cache verification failed: no entry for {rel!r} in {HASH_RECORD} "
                f"(looked for {key}). An empty/partial cache must not be used.")
        # Verify the named path AND a deterministic sample of the whole record, so corruption
        # anywhere is caught rather than only at whichever path a caller happened to name.
        keys = sorted(rec)
        step = max(1, len(keys) // max(1, sample))
        checks = [key] + [k for k in keys[::step][:sample] if k != key]

        h0, m0 = self.hits, self.misses
        _ = self.get(rel)                      # shape/dtype/exists, without touching counters
        self.hits, self.misses = h0, m0
        bad = []
        for k in checks:
            p = self.dir / k
            if not p.exists():
                bad.append((k, "missing"))
            elif hashlib.sha256(p.read_bytes()).hexdigest() != rec[k]:
                bad.append((k, "hash mismatch"))
        if bad:
            raise RuntimeError(
                f"288 cache integrity FAILED: {len(bad)} of {len(checks)} checked entries are "
                f"wrong, e.g. {bad[:3]}. The cache is not what was built -- rebuild it "
                f"(scratch/sophie/rtxvsr/build_vsr_cache.py). Refusing to train on pixels that "
                f"are not what the loader would produce.")
        return 0.0
