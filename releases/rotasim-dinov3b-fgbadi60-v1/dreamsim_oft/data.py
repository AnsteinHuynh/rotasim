"""
[Begin Work Zone]
NIGHTS 2AFC dataset, plus a synthetic stand-in so the pipeline can be smoke-tested
without the 58 GB download.

REAL SCHEMA (confirmed against I:\\MyApps\\sd-train\\dreamsim-nights\\data.csv; the
upstream docs never documented it, so this was read off the actual file):

    id, left_vote, right_vote, votes, ref_path, left_path, right_path,
    split, is_imagenet, prompt

  * `left_vote` / `right_vote` are COMPLEMENTARY BINARY WINNERS, not judge counts:
    exactly one is 1, and left_vote + right_vote == 1 for all 20,019 rows. So the
    target is `right_vote`, meaning 1 <=> the RIGHT image is the more similar one.
    This agrees with DreamSim's own loader, which took positional column 2 --
    and column 2 is right_vote. Independent confirmation of the convention.
  * `votes` is the number of unanimous human judgments (1..11). It is NOT the label.
    Note this carefully: resolving the target by fuzzy name-matching would happily
    land on `votes` (values 6..11) and produce a loss that runs but is meaningless.
  * `right_path` does NOT always point at the `_1.png` file -- the CSV swaps which
    physical distortion sits on which side (e.g. id=6 has right_path=..._006_0.png).
    So paths must be used exactly as given, never reconstructed from the id.

Filtering: rows with votes < 6 are dropped, matching both the paper and DreamSim's
loader. On this copy that is 20,019 -> 17,444 triplets.

No pandas: this venv is a deliberately tuned Python 3.14 stack and pandas is not in
it. The CSV is 20k rows of small fields, so the stdlib `csv` module is plainly
sufficient and avoids perturbing that environment.

The dataset returns [0,1] tensors at 224x224 and does NOT normalize: each backbone
wants different statistics (SigLIP 0.5/0.5, CLIP OpenAI stats, DINOv3 ImageNet), and
that normalization belongs to the branch, not the data.

usage:
    from dreamsim_oft.data import TwoAFCDataset, SyntheticTwoAFCDataset
    ds = TwoAFCDataset(r"I:\\MyApps\\sd-train\\dreamsim-nights", split="train")
    ds = SyntheticTwoAFCDataset(n=64)          # no files needed
[End Work Zone]
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

IMG_SIZE = 224
MIN_VOTES = 6
NIGHTS_NATIVE = 768          # every NIGHTS source is 768x768 (sampled, MEMORY)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ValImageCache:
    """Persistent disk cache of POST-RESIZE uint8 CHW tensors for VAL-side images.

    Stores exactly what the aspect/square decode produces BEFORE normalization and
    jitter (val never jitters anyway). Key = (split_version, geometry seed, (h, w),
    relative image path); the key is PART OF THE FILE NAME, so a key mismatch or a
    missing file is a miss -> decode fresh and store. Stale pixels can never be
    served because a different geometry/split simply addresses a different file.

    Layout (under <project>/cache/fgresq_val/):
        sv{split_version}_g{geom_seed}/{h}x{w}_{md5(rel)[:16]}.bin   raw uint8 CHW
        index.json    one manifest of the whole dir, rebuilt by write_index()
    Concurrency-safe with DataLoader workers: each key maps to one file, written
    once (write-if-missing); the index is only a human-readable manifest, never
    consulted on lookup.
    """

    def __init__(self, subdir: str = "fgresq_val", split_version: int = 2,
                 geom_seed: int = 0):
        self.dir = PROJECT_ROOT / "cache" / subdir / f"sv{int(split_version)}_g{int(geom_seed)}"
        self.split_version = int(split_version)
        self.geom_seed = int(geom_seed)
        self.hits = 0
        self.misses = 0

    def _path(self, rel: str, h: int, w: int) -> Path:
        import hashlib
        key = f"{self.split_version}|{self.geom_seed}|{h}x{w}|{rel}"
        tag = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.dir / f"{h}x{w}_{tag}.bin"

    def get(self, rel: str, h: int, w: int):
        """Return a fresh float CHW tensor in [0,1], or None on any miss."""
        p = self._path(rel, h, w)
        if not p.exists():
            self.misses += 1
            return None
        try:
            raw = p.read_bytes()
            expected = 3 * h * w
            if len(raw) != expected:
                self.misses += 1
                return None
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8,
                                 ).reshape(3, h, w).float().div_(255.0)
            self.hits += 1
            return t
        except Exception:
            # Corrupt/truncated entry = a miss, never an error: decode and overwrite.
            self.misses += 1
            return None

    def put(self, rel: str, h: int, w: int, t: torch.Tensor) -> None:
        u8 = (t.detach().float().clamp(0, 1) * 255.0).to(torch.uint8).contiguous()
        p = self._path(rel, h, w)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            # UNIQUE tmp name: the tmp path was previously hash-derived, so two DataLoader
            # workers decoding the same (missing) entry at once both wrote the SAME .tmp
            # and one crashed on os.replace with WinError 32 (found launching seed 4321
            # against a fresh geometry dir). pid+uuid makes each writer independent.
            import os as _os, uuid as _uuid
            tmp = p.with_suffix(f".{_os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_bytes(u8.numpy().tobytes())
                tmp.replace(p)
            except OSError:
                # Another worker won the race / file locked on Windows: their bytes are
                # the same pixels, so losing is fine. Never fail a training batch here.
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def write_index(self) -> dict:
        """Manifest the whole cache dir into ONE index.json (human-readable only)."""
        import json
        files = sorted(self.dir.glob("*.bin")) if self.dir.exists() else []
        idx = {
            "cache": str(self.dir),
            "split_version": self.split_version,
            "geom_seed": self.geom_seed,
            "files": len(files),
            "total_mib": round(sum(f.stat().st_size for f in files) / 2**20, 2),
            "names": [f.name for f in files],
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "index.json").write_text(json.dumps(idx, indent=1), encoding="utf-8")
        return idx



# Aspect buckets -- THE FIVE, user-fixed 2026-09-24. Replaces the old 11-bucket set
# entirely (user directive: "I want only these 5 buckets"). All dims %16 (patch size),
# all fit a 768x768 NIGHTS source with NO upscaling. Areas 295,936..308,224 px.
# Ratios: 1:1, 4:3, 3:4, 1.536 (~19:12), 0.651.
ASPECT_BUCKETS_544: tuple[tuple[int, int], ...] = (
    (448, 688), (480, 640), (544, 544), (640, 480), (688, 448),
)
# (432,720)/(720,432) added 2026-09-24 on user request: EXACT 3:5 / 5:3 aspect
# (704x416 is 1.692, 1.5% off 5:3; 720x432 is exactly 1.6667). Both %16, fit a
# 768x768 NIGHTS source with no upscale, area 311,040 px = +5.1% over the 295,936
# nominal budget -- the largest deviation in the set, accepted deliberately.

# Explicit, ordered candidates. `target` deliberately does NOT list bare "vote"/"votes":
# on this dataset that would select the judge count instead of the label.
_CANDIDATES: dict[str, list[str]] = {
    "id": ["id", "index", "idx", "triplet_id"],
    "target": ["right_vote", "target", "label", "choice", "y"],
    "ref": ["ref_path", "ref", "reference", "img_ref"],
    "left": ["left_path", "left", "img_0", "x0", "path_0"],
    "right": ["right_path", "right", "img_1", "x1", "path_1"],
    "split": ["split", "set", "fold"],
    "votes": ["votes", "num_votes", "n_votes"],
    "is_imagenet": ["is_imagenet", "imagenet"],
}


def _resolve_columns(fieldnames: list[str], verbose: bool = True) -> dict[str, str | None]:
    lower = {c.lower().strip(): c for c in fieldnames}
    resolved: dict[str, str | None] = {}
    for field, names in _CANDIDATES.items():
        resolved[field] = next((lower[n] for n in names if n in lower), None)

    if verbose:
        print(f"[data] csv columns: {list(fieldnames)}")
        for field, col in resolved.items():
            print(f"[data]   {field:<12s} -> {col}")

    for req in ("target", "ref", "left", "right"):
        if not resolved.get(req):
            raise ValueError(
                f"Could not resolve required column {req!r} from {list(fieldnames)}. "
                f"Candidates tried: {_CANDIDATES[req]}")
    return resolved


def _falsy(v) -> bool:
    return str(v).strip().lower() in {"false", "0", "no", "none", ""}


_PT_CACHE_ANNOUNCED: set[str] = set()


def _pt_cache_root(root, override: str | None = None) -> str | None:
    """Directory of pre-decoded uint8 CHW `.pt` tensors for this dataset, or None.

    DEFAULT OFF (env `FGRESQ_PT_CACHE=<dir>` only). It was briefly default-ON and that was
    wrong: the cache is keyed by BASENAME, while the fgresq tree contains same-basename
    files with different content (e.g. ref/000068.bmp decodes 288x288 but the cache entry
    000068.pt is 384x512). Measured 2026-09-26 on a 60-image sample: 20 mismatches, max
    pixel delta 254, two of them different SHAPES -- i.e. it silently serves the WRONG
    IMAGE for roughly a third of files. See scratch/sophie/pt_cache_ab.py. Do not re-enable
    by default until the cache is rebuilt keyed by RELATIVE PATH (and re-verified lossless);
    until then this is an explicit, opt-in performance experiment only.
    """
    env = os.environ.get("FGRESQ_PT_CACHE")
    resolved: str | None
    if env is None:
        resolved = None
    elif env.strip() == "" or env.strip().lower() in ("none", "off", "0", "false", "no"):
        resolved = None
    else:
        resolved = env.strip()
    key = f"{root}|{resolved}"
    if key not in _PT_CACHE_ANNOUNCED:
        _PT_CACHE_ANNOUNCED.add(key)
        if resolved:
            print(f"[data] pt cache: {resolved} (DECODE SKIPPED -- basename-keyed, "
                  f"UNSOUND for ~1/3 of images until rebuilt; opt-in only)")
        elif os.environ.get("FGRESQ_PT_CACHE_QUIET") is None:
            print("[data] pt cache: OFF -- decoding PNG/BMP source images")
    return resolved


def _open_image(root, rel: str, cache_dir: str | None):
    """PIL RGB image for `rel`, preferring the pre-decoded `.pt` cache (skips DECODE only).

    The cache (scratch/20260925/build_images_pt.py) holds one uint8 CHW tensor per source
    image, lossless-verified against the PNG/BMP decode. Every resize / aspect / jitter step
    downstream still runs on those same pixels, so model inputs are BIT-IDENTICAL: this
    changes speed and nothing else. Returns a plain image (no file handle to close).
    """
    from PIL import Image

    if cache_dir:
        # LAYOUT (kerok directive 2026-09-26): <cache>/<dir>/<imagename>-<ext>.pt
        #   ref/000068.bmp    -> <cache>/ref/000068-bmp.pt
        #   images/000000.png -> <cache>/images/000000-png.pt
        # The directory carries which tree the image came from and the name carries stem AND
        # extension, so nothing can collide. Earlier keys (basename only, extension stripped)
        # silently served the WRONG IMAGE for ~1/3 of files -- see _pt_cache_root.
        relp = rel.replace("\\", "/")
        head, _, tail = relp.rpartition("/")
        stem, dot, ext = tail.rpartition(".")
        name = f"{stem}-{ext.lower()}.pt" if dot else f"{tail}.pt"
        p = os.path.join(cache_dir, head, name) if head else os.path.join(cache_dir, name)
        if os.path.exists(p):
            t = torch.load(p)
            return Image.fromarray(t.permute(1, 2, 0).contiguous().numpy())
    with Image.open(Path(root) / rel) as im:
        return im.convert("RGB")


def _hflip_sample(imgs, p: float, seed, idx: int, split: str):
    """Mirror a WHOLE triplet/pool horizontally with probability p (kerok 2026-10-01 morning).

    WHY: a perceptual similarity metric should be flip-symmetric -- a mirrored image is
    equally similar -- and NIGHTS/FGResQ ships no flipped examples, so the prior has to come
    from augmentation.

    THREE PROPERTIES THAT MATTER:
      * ALL images of one sample are flipped together. Flipping only some of ref/left/right
        would silently corrupt the 2AFC label rather than augment it.
      * Train split only. val/test are the measuring instrument and must never be augmented
        (the same rule as color_jitter: "val never jitters anyway").
      * Deterministic in (geometry seed, index) -- the same generator family as the geometry
        plan -- so the augmentation stays a pure function of (seed, epoch), a resume replays
        it exactly, and prefetched worker sets agree with the parent. The pool sampler calls
        assign_epoch_geometry(seed + epoch) every epoch, so the draw VARIES BY EPOCH: the same
        triplet is seen mirrored in one epoch and not in another, which is what actually
        teaches orientation invariance. (A fixed per-index flip would only add data variety.)

    p <= 0 returns the list UNCHANGED -- bit-identical to the pre-flip pipeline.
    """
    if split != "train" or p <= 0.0:
        return imgs
    g = torch.Generator().manual_seed((int(seed) if seed is not None else 0) * 1000033 + int(idx))
    if torch.rand(1, generator=g).item() >= p:
        return imgs
    return [torch.flip(t, dims=[-1]) for t in imgs]


class TwoAFCDataset(Dataset):
    """Real NIGHTS triplets."""

    def __init__(self, root_dir: str | os.PathLike, split: str = "train",
                 verbose: bool = True, validate_files: bool = True, max_validate: int = 64,
                 image_size: int = IMG_SIZE,
                 aspect_buckets: tuple[tuple[int, int], ...] | None = None,
                 color_jitter: float = 0.0, hflip: float = 0.0,
                 shape_seed: int | None = None):
        self.root = Path(root_dir)
        csv_path = self.root / "data.csv"
        # Decode fast path (default ON when <root>/images-pt exists; see _pt_cache_root).
        self._pt_cache = _pt_cache_root(self.root)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Expected the NIGHTS root containing data.csv, "
                f"ref/ and distort/.")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = _resolve_columns(list(reader.fieldnames or []), verbose=verbose)
            rows = list(reader)
        c = self.columns

        n_all = len(rows)
        if c["votes"]:
            rows = [r for r in rows if int(r[c["votes"]]) >= MIN_VOTES]
            if verbose:
                print(f"[data] votes >= {MIN_VOTES}: {n_all} -> {len(rows)} triplets")
        elif verbose:
            print("[data] no votes column; skipping the unanimous-vote filter")

        if split != "all":
            if not c["split"]:
                raise ValueError(f"split={split!r} requested but no split column in csv")
            rows = [r for r in rows if r[c["split"]] == split]

        self.rows = rows
        self.split = split
        # The resolution the towers will actually be fed. NIGHTS is 768x768 natively, so
        # this is a real downsample (or, at 544, a much smaller one). It MUST match the
        # model's image_size: the position tables are sized for one specific grid.
        self.image_size = image_size
        self._tf = None  # built lazily so importing this module needs no torchvision
        # ---- random-aspect mode (the 544 class) ----
        # aspect_buckets non-None turns on native-resolution random cropping: each
        # triplet gets ONE (h,w) bucket + ONE crop rect, shared by ref/left/right.
        # Batching is handled by AspectBatchSampler (same bucket per batch), because
        # both towers need a uniform sequence length within a batch.
        self.aspect_buckets = tuple(aspect_buckets) if aspect_buckets else None
        self.color_jitter = float(color_jitter)
        self.hflip = float(hflip)
        # geometry per index: idx -> (h, w, x0, y0). Assigned per epoch by
        # assign_epoch_geometry(); deterministic given the seed, so a resume rebuilds
        # the exact same crops.
        self._geom: dict[int, tuple[int, int, int, int]] = {}
        self._geom_seed: int | None = None
        # For val/test: fix the geometry ONCE, deterministically, so every eval of this
        # split sees the same crops and accuracies are comparable across the run.
        if self.aspect_buckets and split != "train":
            self.assign_epoch_geometry(shape_seed if shape_seed is not None else 0)
        if verbose and self.aspect_buckets:
            areas = [h * w for h, w in self.aspect_buckets]
            print(f"[data] random-aspect ON: {len(self.aspect_buckets)} buckets, "
                  f"area {min(areas)}..{max(areas)} px (budget {544*544}), "
                  f"jitter +/-{self.color_jitter * 100:.0f}%"
                  + ("" if split == "train" else f", geometry FROZEN (seed "
                                                    f"{shape_seed if shape_seed is not None else 0})"))

        if verbose:
            tgt = [float(r[c["target"]]) for r in rows]
            pos = sum(1 for t in tgt if t >= 0.5)
            print(f"[data] split={split!r}: {len(rows)} triplets "
                  f"(target=1 on {pos}, target=0 on {len(tgt) - pos})")
            if tgt and not set(tgt) <= {0.0, 1.0}:
                raise ValueError(
                    f"target column {c['target']!r} has values outside {{0,1}} "
                    f"(e.g. {sorted(set(tgt))[:5]}). Did column resolution pick the "
                    f"vote COUNT instead of the vote LABEL?")
            if not pos or pos == len(tgt):
                raise ValueError(
                    f"target column {c['target']!r} is constant ({tgt[0]}); refusing to "
                    f"train on a degenerate label.")

        if validate_files and rows:
            missing = self._check_files(max_validate)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of the first {min(max_validate, len(rows))} triplets "
                    f"reference missing images. First few: {missing[:4]}")

    # -- helpers --------------------------------------------------------------
    @property
    def transform(self):
        """Square / fixed-aspect class input transform.

        DELIBERATE CROP (kerok directive): the 288 square class uses ONE full-frame resize
        on the exact fractional sampling grid -- a centre-square Lanczos3 crop with
        full-frame support (see dreamsim_oft.resize.SquareCrop, RESIZE_TAG). It replaces
        Resize((n, n), BICUBIC), which did not crop at all but SQUASHED the aspect ratio.

        SQUARE CLASS ONLY. The aspect-bucket branch deliberately keeps "RESIZE, NOT CROP"
        (user directive 2026-09-24): when geometry is drawn per pool, squashing the whole
        image keeps every pixel and never loses content asymmetrically inside a triplet.
        Do not "unify" the two policies -- they answer different questions.
        """
        if self._tf is None:
            from .resize import SquareCrop

            self._tf = SquareCrop(self.image_size)
        return self._tf

    @property
    def pixel288(self):
        """Optional post-resize 288 cache (env FGRESQ_PT_288), OFF unless set.

        Skips DECODE AND RESIZE. Only valid for the square class: it serves an already
        cropped (3,288,288) tensor, so the aspect branch must never call it. The reader
        proves at construction that its pixels match the live transform bit-for-bit, and
        refuses a directory that does not name the resampling method.
        """
        if getattr(self, "_pc288", None) is None:
            import os as _os

            from .pixel_cache import PixelCache288, resolve_288_root

            if self.image_size != 288 or resolve_288_root(None) is None:
                # The canonical cache is a FIXED 288x288 square tensor. Never auto-use it for
                # any other square size: a 544-square run must decode+resize exactly as before.
                self._pc288 = False
            else:
                rel = None
                try:
                    if self.rows:
                        rel = str(self.rows[0][self.columns["ref"]])
                except Exception:
                    rel = None
                if rel is None:
                    print("[288cache] dataset has no rows -- cache disabled")
                    self._pc288 = False
                else:
                    self._pc288 = PixelCache288(
                        self.root, out=self.image_size,
                        dtype=_os.environ.get("FGRESQ_PT_288_DTYPE", "float32"),
                        verify_rel=rel)
        return self._pc288 or None

    def _rotasim_eval_get(self, rel):
        """ROTASIM_EVAL_SHARDS=<rotasim root>: serve eval pixels from the shard
        cache instead of decoding sources (corpus may live on slow media after
        the I:->X: move). Pixels are protocol-IDENTICAL by construction -- the
        fgresq-eval shards were produced by this class's own SquareCrop
        transform (fraclanczos3-288-centresquare-fp32, 5,148 files, audited
        max|d|=0.0 vs the live path). OFF unless the env var is set."""
        src = getattr(self, "_rotasim_src", None)
        if src is None:
            import os as _os
            root = _os.environ.get("ROTASIM_EVAL_SHARDS")
            if not root:
                self._rotasim_src = False
                return None
            from .shardio import RotasimShardSource
            src = RotasimShardSource(root, "fgresq-eval", "eval")
            print(f"[rotasim-eval] ON  {len(src)} cached eval pixels from {root} "
                  f"(fraclanczos3-288-fp32; decode+resize skipped)")
            self._rotasim_src = src
        t = src.get(rel) if src else None
        return t

    def _pixel288_get(self, rel):
        t = self._rotasim_eval_get(rel)
        if t is not None:
            return t
        c = self.pixel288
        if c is None:
            return None
        t = c.get(str(rel))
        return None if t is None else c.as_model_input(t)

    def _paths(self, row) -> tuple[str, str, str]:
        c = self.columns
        return str(row[c["ref"]]), str(row[c["left"]]), str(row[c["right"]])

    def _check_files(self, n: int) -> list[str]:
        bad = []
        for i in range(min(n, len(self.rows))):
            for rel in self._paths(self.rows[i]):
                if not (self.root / rel).exists():
                    bad.append(rel)
        return bad

    def label_balance(self) -> float:
        """Fraction of triplets whose label is 1 (right more similar). 0.5 is balanced."""
        t = [float(r[self.columns["target"]]) for r in self.rows]
        return sum(1 for x in t if x >= 0.5) / max(1, len(t))

    # -- random-aspect geometry ------------------------------------------------
    def assign_epoch_geometry(self, seed: int) -> None:
        """Draw (bucket, crop rect) for EVERY index, deterministically from `seed`.

        Called by AspectBatchSampler.__iter__ before workers are spawned (they pickle
        the dataset at iterator creation, so the assignment they see is this one).
        Deterministic in `seed`, which train.py sets to cfg.seed + epoch: that makes a
        mid-epoch resume rebuild the exact same crops by replaying the same seed.
        """
        if not self.aspect_buckets:
            return
        g = torch.Generator().manual_seed(int(seed))
        n_b = len(self.aspect_buckets)
        rects = torch.randint(0, NIGHTS_NATIVE + 1, (len(self.rows), 4), generator=g)
        self._geom = {}
        for i in range(len(self.rows)):
            h, w = self.aspect_buckets[int(rects[i, 0]) % n_b]
            x0 = min(int(rects[i, 1]), NIGHTS_NATIVE - w)
            y0 = min(int(rects[i, 2]), NIGHTS_NATIVE - h)
            self._geom[i] = (h, w, x0, y0)
        self._geom_seed = int(seed)

    def bucket_of(self, idx: int) -> tuple[int, int]:
        """The (h,w) this index currently holds -- used by the batch sampler to group."""
        if not self._geom:
            raise RuntimeError("bucket_of() called before assign_epoch_geometry()")
        return self._geom[idx][:2]

    def _draw_jitter(self, g: torch.Generator) -> dict:
        """Draw ONE set of +/-color_jitter factors, deterministic in the generator.

        Drawn ONCE PER TRIPLET and applied identically to all three images (user
        directive 2026-09-24): the triplet's color IDENTITY is preserved -- the same
        exposure shift happens to ref, left and right -- while the run as a whole
        still sees varied colors. (The earlier per-image scheme is retired.)
        """
        j = self.color_jitter
        u = lambda: (torch.rand(1, generator=g).item() * 2 - 1) * j  # uniform in [-j, +j]
        return {"brightness": 1.0 + u(), "contrast": 1.0 + u(),
                "saturation": 1.0 + u(), "hue": u()}  # hue spans [-0.5,0.5]; j of that

    @staticmethod
    def _jitter(x: torch.Tensor, f: dict) -> torch.Tensor:
        """Apply a pre-drawn factor set (shared triplet-wide) in [0,1] space."""
        from torchvision.transforms import functional as TF
        x = TF.adjust_brightness(x, f["brightness"])
        x = TF.adjust_contrast(x, f["contrast"])
        x = TF.adjust_saturation(x, f["saturation"])
        x = TF.adjust_hue(x, f["hue"])
        return x.clamp(0.0, 1.0)

    # -- Dataset --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.rows[idx]
        c = self.columns
        ref, left, right = self._paths(row)
        imgs = []
        if self.aspect_buckets:
            if idx not in self._geom:
                raise RuntimeError(
                    f"index {idx} has no geometry; assign_epoch_geometry() must run "
                    f"before iteration (AspectBatchSampler does this).")
            h, w, _x0, _y0 = self._geom[idx]
            # ONE jitter factor set PER TRIPLET, deterministic in (geometry seed, idx):
            # ref/left/right all receive the SAME +/-3% shift.
            jf = None
            if self.split == "train" and self.color_jitter > 0:
                jg = torch.Generator().manual_seed((self._geom_seed or 0) * 1000003 + idx)
                jf = self._draw_jitter(jg)
            for slot, rel in enumerate((ref, left, right)):
                im = _open_image(self.root, rel, self._pt_cache)
                if im.size != (NIGHTS_NATIVE, NIGHTS_NATIVE):
                    # Defensive: everything sampled was 768x768, but do not crash
                    # a 4-hour run on a rare outlier -- normalize first.
                    im = im.resize((NIGHTS_NATIVE, NIGHTS_NATIVE), Image.BILINEAR)
                # RESIZE, NOT CROP (user directive 2026-09-24): a shared crop rect
                # still loses content ASYMMETRICALLY across a triplet -- the monkey
                # on the right of ref falls outside the window while the monkey on
                # the left of img1 survives, and the comparison silently loses its
                # object. Squashing the FULL 768x768 down to (w,h) keeps every
                # pixel of every image; the aspect change becomes the augmentation.
                # BILINEAR for speed per user directive.
                im = im.resize((w, h), Image.BILINEAR)
                from torchvision.transforms.functional import to_tensor
                t = to_tensor(im)
                if jf is not None:
                    t = self._jitter(t, jf)
                imgs.append(t)
        else:
            for rel in (ref, left, right):
                t = self._pixel288_get(rel)
                imgs.append(t if t is not None
                            else self.transform(_open_image(self.root, rel, self._pt_cache)))
        imgs = _hflip_sample(imgs, self.hflip, self._geom_seed, idx, self.split)
        target = torch.tensor(float(row[c["target"]]), dtype=torch.float32)
        tid = int(row[c["id"]]) if c["id"] else idx
        return imgs[0], imgs[1], imgs[2], target, tid


class FGResQDataset(TwoAFCDataset):
    """FGRestore (REAL photographs) pairwise-preference dataset -- the first non-SD,
    non-square data this pipeline has ever trained on.

    Source layout (I:\\MyApps\\sd-train\\fgresq):
        annotations/IR_train.json      24,817 train PAIRS in exact 2AFC shape
        annotations/test_pairs/*.json  6,068 official preference-test pairs
        images/  (distorted/restored candidates), ref/  (references)

    IR_train.json record fields: ref_image, image_nameA, image_nameB,
    human_preference_for_A (1 | 0 | 0.5 tie), score_normA/B, task, scene_id.

    CONVENTIONS (deliberate, documented):
      * A -> left, B -> right, and target = 1 - human_preference_for_A, so the
        project-wide invariant holds: target=1 <=> the RIGHT image is more similar.
        A tie (0.5) stays 0.5 -- the BT loss accepts it natively (it pushes d0==d1),
        and two_afc_accuracy EXCLUDES ties from the count.
      * scene_id 3/4 have NO pristine reference (ref_image is the pre-restoration
        image there). exclude_scene34=True drops them from train/val (~2.1k rows).
        The official test split is scored WITH them unless the scorer says otherwise.
      * There is no split column. The train/val partition is a deterministic hash of
        the row's own path triple, so two dataset instances built separately agree on
        the partition without a shared side file. Same caveat as NIGHTS: images of a
        scene recur across rows, so val tests preference generalisation, not new
        subjects.
      * RESIZE, NOT CROP (the live-class policy): real photos arrive at many sizes
        (288x288 SR ... ~1800x1200), so the whole image is resized to the bucket
        (w,h). Aspect-native 544-class behaviour otherwise matches TwoAFCDataset.

    Returns the same tuple as TwoAFCDataset: (ref, left, right, target, id).
    """

    def __init__(self, root_dir, split: str = "train", verbose: bool = True,
                 validate_files: bool = True, max_validate: int = 64,
                 image_size: int = IMG_SIZE,
                 aspect_buckets=None, color_jitter: float = 0.0, hflip: float = 0.0,
                 shape_seed: int | None = None,
                 exclude_scene34: bool = True,
                 split_version: int = 1,
                 val_fraction: float = 0.08, val_seed: int = 7919,
                 val_cache: "ValImageCache | None" = None):
        import hashlib
        import json as _json

        self.root = Path(root_dir)
        self.split = split
        self._pt_cache = _pt_cache_root(self.root)
        self.image_size = image_size
        self._tf = None
        self.aspect_buckets = tuple(aspect_buckets) if aspect_buckets else None
        self.color_jitter = float(color_jitter)
        self.hflip = float(hflip)
        self._geom = {}
        self._geom_seed = None
        # Persistent VAL decode cache (post-resize uint8). TRAIN must stay None: the
        # train side jitters live and must never serve cached pixels.
        self.val_cache = val_cache if split != "train" else None
        # Columns map so the inherited label_balance() / helpers keep working.
        self.columns = {"id": "id", "target": "target", "ref": "ref",
                        "left": "left", "right": "right", "split": None,
                        "votes": None, "is_imagenet": None}

        if split in ("train", "val"):
            jpath = self.root / "annotations" / "IR_train.json"
            recs = _json.loads(jpath.read_text(encoding="utf-8"))
            n_all = len(recs)
            if exclude_scene34:
                recs = [r for r in recs if int(r["scene_id"]) not in (3, 4)]
                if verbose:
                    print(f"[fgresq] scene_id 3/4 excluded (no pristine ref): "
                          f"{n_all} -> {len(recs)} pairs")
            # Deterministic train/val carve. split_version 1 (LEGACY): hash of the ROW's
            # own path triple -> POOL-LEAKY (the ~8% val rows come from pools that also
            # contribute train rows, so the keeper's 77.91% was measured on a leaky
            # carve). split_version 2: hash of the POOL key (ref_image) alone, so a pool
            # lives entirely on one side and val measures generalisation to unseen
            # pools. Same md5/10000 scheme, same val_fraction, deterministic.
            def _row_is_val(r):
                key = f"{r['ref_image']}|{r['image_nameA']}|{r['image_nameB']}|{val_seed}"
                return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000 \
                    < int(val_fraction * 10000)

            def _pool_is_val(r):
                key = f"{r['ref_image']}|{val_seed}"
                return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000 \
                    < int(val_fraction * 10000)

            self.split_version = int(split_version)
            _is_val = _pool_is_val if self.split_version >= 2 else _row_is_val
            want_val = split == "val"
            recs = [r for r in recs if _is_val(r) == want_val]
            if verbose:
                n_pools = len({r["ref_image"] for r in recs})
                print(f"[fgresq] IR_train.json: split={split!r} -> {len(recs)} rows in "
                      f"{n_pools} pools (split_version={self.split_version}, "
                      f"val_fraction={val_fraction})")
        elif split == "test":
            d = self.root / "annotations" / "test_pairs"
            recs = []
            for jp in sorted(d.glob("*.json")):
                part = _json.loads(jp.read_text(encoding="utf-8"))
                recs.extend(part)
                if verbose:
                    print(f"[fgresq] test_pairs/{jp.name}: {len(part)} pairs")
        else:
            raise ValueError(f"FGResQDataset split must be train|val|test, got {split!r}")

        self.rows = []
        for i, r in enumerate(recs):
            self.rows.append({
                "ref": str(r["ref_image"]),
                "left": str(r["image_nameA"]),       # A -> left
                "right": str(r["image_nameB"]),      # B -> right
                # target = 1 <=> RIGHT more similar (see class docstring)
                "target": 1.0 - float(r["human_preference_for_A"]),
                "id": i,
                "task": r.get("task", "?"),
                "scene_id": r.get("scene_id", -1),
            })

        # Label sanity: targets must be in {0, 0.5, 1} and non-constant.
        tgt = [r["target"] for r in self.rows]
        bad = [t for t in tgt if t not in (0.0, 0.5, 1.0)]
        if bad:
            raise ValueError(f"[fgresq] targets outside {{0,0.5,1}}: e.g. {bad[:5]}")
        n_tie = sum(1 for t in tgt if t == 0.5)
        pos = sum(1 for t in tgt if t > 0.5)
        if verbose:
            print(f"[fgresq] split={split!r}: {len(tgt)} pairs | pref-right {pos} "
                  f"| pref-left {len(tgt) - pos - n_tie} | TIES {n_tie}")
        if tgt and pos in (0, len(tgt)) and n_tie == 0:
            raise ValueError("[fgresq] target is constant; refusing a degenerate label.")

        if self.aspect_buckets and split != "train":
            self.assign_epoch_geometry(shape_seed if shape_seed is not None else 0)
        if verbose and self.aspect_buckets:
            areas = [h * w for h, w in self.aspect_buckets]
            print(f"[fgresq] random-aspect ON: {len(self.aspect_buckets)} buckets, "
                  f"area {min(areas)}..{max(areas)} px, jitter +/-{self.color_jitter*100:.0f}%"
                  + ("" if split == "train" else
                     f", geometry FROZEN (seed {shape_seed if shape_seed is not None else 0})"))

        if validate_files and self.rows:
            bad = []
            for r in self.rows[:max_validate]:
                for rel in (r["ref"], r["left"], r["right"]):
                    if not (self.root / rel).exists():
                        bad.append(rel)
            if bad:
                raise FileNotFoundError(
                    f"[fgresq] {len(bad)} of the first {min(max_validate, len(self.rows))} "
                    f"pairs reference missing images. First few: {bad[:4]}")

    def label_balance(self) -> float:
        t = [r["target"] for r in self.rows]
        return sum(1 for x in t if x > 0.5) / max(1, len(t))

    def __getitem__(self, idx: int):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        row = self.rows[idx]
        imgs = []
        if self.aspect_buckets:
            if idx not in self._geom:
                raise RuntimeError(
                    f"index {idx} has no geometry; assign_epoch_geometry() must run "
                    f"before iteration (AspectBatchSampler does this).")
            h, w, _x0, _y0 = self._geom[idx]
            jf = None
            if self.split == "train" and self.color_jitter > 0:
                jg = torch.Generator().manual_seed((self._geom_seed or 0) * 1000003 + idx)
                jf = self._draw_jitter(jg)
            for rel in (row["ref"], row["left"], row["right"]):
                t = None
                if self.val_cache is not None and jf is None:
                    t = self.val_cache.get(rel, h, w)
                if t is None:
                    # RESIZE, NOT CROP: real photos at arbitrary native sizes go
                    # whole-image to the bucket, same policy as the live 544 class.
                    im = _open_image(self.root, rel, self._pt_cache)
                    t = to_tensor(im.resize((w, h), Image.BILINEAR))
                    if self.val_cache is not None and jf is None:
                        self.val_cache.put(rel, h, w, t)
                if jf is not None:
                    t = self._jitter(t, jf)
                imgs.append(t)
        else:
            for rel in (row["ref"], row["left"], row["right"]):
                t = self._pixel288_get(rel)
                if t is None and self.val_cache is not None:
                    t = self.val_cache.get(rel, self.image_size, self.image_size)
                if t is None:
                    # WARNING: an on-disk cache keyed only by (rel, h, w) cannot tell one
                    # resize from another, so changing dreamsim_oft.resize.RESIZE_TAG
                    # silently invalidates these entries. Stale pixels here are the exact
                    # bug class we already hit once (a cache that served the wrong tree):
                    # rebuild, never reuse across a resize change.
                    t = self.transform(_open_image(self.root, rel, self._pt_cache))
                    if self.val_cache is not None:
                        self.val_cache.put(rel, self.image_size, self.image_size, t)
                imgs.append(t)
        # This dataset IS the training side -- it has no `split` attribute at all (the
        # evaluator uses the pairwise path), so the split is passed literally rather than
        # read from self, which would be an AttributeError on the first batch.
        imgs = _hflip_sample(imgs, self.hflip, self._geom_seed, idx, "train")
        return imgs[0], imgs[1], imgs[2], \
            torch.tensor(row["target"], dtype=torch.float32), row["id"]


class SyntheticTwoAFCDataset(Dataset):
    """Random-noise triplets: exercises the full pipeline with zero dataset on disk.

    Not a learning signal -- but the label convention must still be right, because
    getting it backwards makes accuracy sit at exactly 0.0 rather than ~chance, and
    that is a much more confusing symptom than an obvious failure.
    """

    def __init__(self, n: int = 64, seed: int = 0, size: int = IMG_SIZE,
                 shape: tuple[int, int] | None = None):
        self.n = n
        self.size = size
        self.shape = shape      # non-square (h,w) for the 544-class smoke test
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        h, w = self.shape if self.shape else (self.size, self.size)
        ref = torch.rand(3, h, w, generator=g)
        # Convention (matches DreamSim: `decisions = dist1 < dist0`, correct when
        # target >= 0.5):  target == 1  <=>  the RIGHT image is the more similar one.
        right = torch.clamp(ref + 0.05 * torch.randn(ref.shape, generator=g), 0, 1)
        left = torch.rand(3, h, w, generator=g)
        target = torch.tensor(1.0)
        return ref, left, right, target, idx


class AspectBatchSampler:
    """Batch sampler that groups triplets by their CURRENT aspect bucket.

    Both towers need a uniform token count within a batch, and with a per-triplet
    random (h,w) the only honest alternatives are bucketing or per-sample forwards
    with accumulation (strictly worse: forfeits batched attention). This sampler:

      * draws per-index geometry deterministically from seed (+epoch) via
        ds.assign_epoch_geometry() -- at __iter__ time, BEFORE DataLoader workers
        are spawned, so every worker pickles the same assignment;
      * groups indices by bucket, chunks into batches, and shuffles BATCH order;
      * supports skip_batches for mid-epoch resume: the plan is deterministic, so
        slicing off the consumed prefix reproduces the original continuation
        WITHOUT decoding anything (the NVMe-thrash lesson, at the sampler level).
    """

    def __init__(self, ds: "TwoAFCDataset", batch_size: int, seed: int = 1234,
                 epoch: int = 0, shuffle: bool = True, drop_last: bool = True,
                 skip_batches: int = 0):
        if not getattr(ds, "aspect_buckets", None):
            raise ValueError("AspectBatchSampler needs a dataset built with aspect_buckets")
        self.ds = ds
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.skip_batches = skip_batches

    def _plan(self) -> list[list[int]]:
        self.ds.assign_epoch_geometry(self.seed + self.epoch)
        g = torch.Generator().manual_seed((self.seed + self.epoch) * 7919 + 13)
        order = torch.randperm(len(self.ds), generator=g).tolist()
        groups: dict[tuple[int, int], list[int]] = {}
        for i in order:
            groups.setdefault(self.ds.bucket_of(i), []).append(i)
        batches: list[list[int]] = []
        keys = list(groups)
        if self.shuffle:
            perms = torch.randperm(len(keys), generator=g).tolist()
            keys = [keys[p] for p in perms]
        for k in keys:
            items = groups[k]
            for s in range(0, len(items), self.batch_size):
                chunk = items[s:s + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle:
            perms = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[p] for p in perms]
        return batches

    def __iter__(self):
        return iter(self._plan()[self.skip_batches:])

    def __len__(self) -> int:
        return max(0, len(self._plan()) - self.skip_batches)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4,
                seed: int = 1234, indices=None, batch_sampler=None, pin_memory: bool = True,
                collate_fn=None, persistent_workers: bool = False, prefetch_factor=None):
    """DataLoader over `ds`.

    `indices` (optional): an explicit ordering to iterate instead of a shuffle. Used to resume
    MID-EPOCH without decoding the batches that were already consumed -- see the long note in
    train.py. Passing indices disables shuffling; the caller owns the ordering.

    `batch_sampler` (optional): supersedes batch_size/shuffle/indices -- used for the
    aspect-bucketed 544-class loaders.

    `pin_memory`: page-locked host staging so non_blocking H2D copies can overlap compute.
    Measured 9->13 GB/s transfer and, more importantly, non_blocking silently degrades to
    synchronous on pageable memory. Kept as a flag so it can be A/B'd end to end.

    `persistent_workers` / `prefetch_factor`: WDDM stall mitigation (forensic report
    2026-09-26, scratch\\20260926\\VRAM_WEDGE_TIMELINE.md). After an eval the loader's
    prefetched queue is torn down while the GPU context churns; with the default
    prefetch_factor=2 x 8 workers that is 16 batches of host commit to re-establish, and
    the producer side wedged ~30 s per batch for the rest of the run. Keeping workers
    alive (persistent) and the queue shallow (prefetch 1 => 4 batches at nw4) bounds the
    teardown/re-priming cost. Defaults preserve legacy behaviour exactly.
    """
    from torch.utils.data import DataLoader

    extra = {}
    if num_workers and num_workers > 0:
        if persistent_workers:
            extra["persistent_workers"] = True
        if prefetch_factor is not None:
            extra["prefetch_factor"] = int(prefetch_factor)

    if batch_sampler is not None:
        return DataLoader(ds, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=collate_fn, **extra)
    if indices is not None:
        return DataLoader(ds, batch_size=batch_size, sampler=list(indices),
                          num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
                          **extra)

    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g, drop_last=False, pin_memory=pin_memory, **extra)


# ============================ D1 dense pool batching ============================
# FGResQ pools are 1 ref + K candidates with a full preference table over candidate
# PAIRS. Pool training draws whole pools, scores every candidate against the ref in
# one forward, and turns every supervised candidate pair into one BT logit.

def build_fgresq_pools(root, exclude_scene34: bool = True, val_fraction: float = 0.08,
                       val_seed: int = 7919, split_version: int = 2, verbose: bool = True,
                       soft_gap_ref: float = 0.0):
    """Pools from IR_train.json for the D1 pool-BT path.

    Returns (pools, pair_target):
      pools      list of {"ref": path, "cands": [(path, score_norm), ...]} -- TRAIN-side
                 pools only (scene 3/4 out; pools landing in the val carve under the
                 given split_version are excluded; the official test_pairs untouched).
      pair_target dict {(path_a, path_b): y} with y = P(b preferred over a), for every
                 row that exists; ties (0.5) are kept here so the COLLATE can mask them.
    Deterministic; the carve predicate mirrors FGResQDataset exactly (split_version
    1 = row-hash carve, 2 = pool-disjoint hash of ref_image).
    """
    import hashlib
    import json as _json

    recs = _json.loads((Path(root) / "annotations" / "IR_train.json").read_text(encoding="utf-8"))
    n_all = len(recs)
    if exclude_scene34:
        recs = [r for r in recs if int(r["scene_id"]) not in (3, 4)]

    thresh = int(val_fraction * 10000)

    def _pool_is_val(r):
        key = f"{r['ref_image']}|{val_seed}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000 < thresh

    def _row_is_val(r):
        key = f"{r['ref_image']}|{r['image_nameA']}|{r['image_nameB']}|{val_seed}"
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10000 < thresh

    pair_target: dict[tuple[str, str], float] = {}
    nvotes: dict[str, int] = {}          # D2: candidate path -> rows it appears in
    pools: dict[str, dict] = {}
    for r in recs:
        ref, a, b = str(r["ref_image"]), str(r["image_nameA"]), str(r["image_nameB"])
        nvotes[a] = nvotes.get(a, 0) + 1
        nvotes[b] = nvotes.get(b, 0) + 1
        p = float(r["human_preference_for_A"])          # 1 = A preferred, 0.5 tie
        y_ab = 1.0 - p                                  # P(b preferred over a)
        y_ba = p
        # CONFIDENCE-SOFTENED LABEL (2026-10-01, `soft_gap_ref` > 0 only; default 0 = OFF
        # and bit-identical to the old hard label). THE BUG THIS FIXES: the target was
        # hard 0/1, so a 99/1 row and a 51/49 row produced the SAME gradient, even though
        # |score_normA - score_normB| -- the human confidence -- is the strongest measured
        # predictor of this metric's own error, and 13.7% of test rows sit at |gap| < 0.02
        # where the model scores 55.7% carrying 23% of all errors. `bt_confidence_weight`
        # could NOT see this: it computes 2|y - 0.5| from the BINARISED y, so it is 1.0 on
        # 98.6% of rows (structurally inert). Softening HERE is what makes the confidence
        # visible to the loss: y = 0.5 + 0.5*sign*gap/ref, so a decisive pair is unchanged
        # (gap >= ref -> 1.0/0.0) and a near-tie is pulled toward 0.5, where the BT/BCE
        # gradient stops pushing a direction the humans barely expressed.
        # The gap-0 case stays exactly 0.5 on purpose: __getitem__ masks `t == 0.5`, so a
        # true human tie remains UNSUPERVISED rather than becoming a target.
        # MEASURED SIDE EFFECT, gated: 8 of 45,402 train pairs have a DECISIVE preference
        # but score_normA == score_normB, so they soften to exactly 0.5 and are therefore
        # MASKED as ties (0.018% of pairs). That is deliberate -- the two signals contradict
        # each other on those rows, so there is no defensible target -- but it is a mask
        # change, so it is asserted in the gate rather than left to be discovered.
        if soft_gap_ref > 0.0 and y_ab != 0.5:
            gap = abs(float(r.get("score_normA", 0.0)) - float(r.get("score_normB", 0.0)))
            c = min(gap / float(soft_gap_ref), 1.0)
            s = 1.0 if y_ab > 0.5 else -1.0
            y_ab = 0.5 + 0.5 * s * c
            y_ba = 1.0 - y_ab
        pair_target[(a, b)] = y_ab
        pair_target[(b, a)] = y_ba
        pool = pools.setdefault(ref, {"ref": ref, "cands": {}})
        pool["cands"][a] = float(r.get("score_normA", 0.0))
        pool["cands"][b] = float(r.get("score_normB", 0.0))

    if int(split_version) >= 2:
        val_refs = {ref for ref in pools if _pool_is_val({"ref_image": ref})}
    else:
        val_refs = {str(r["ref_image"]) for r in recs if _row_is_val(r)}
    keep = [pools[ref] for ref in sorted(pools) if ref not in val_refs]
    for pool in keep:
        pool["cands"] = sorted(pool["cands"].items())   # deterministic candidate order
        pool["nvotes"] = nvotes                         # shared global count map (D2)
    if verbose:
        print(f"[fgresq-pool] {n_all} -> {len(recs)} rows (scene34 out) -> "
              f"{len(keep)} TRAIN pools ({len(val_refs)} val-side pools excluded), "
              f"{len(pair_target)} directed pair targets, split_version={split_version}")
    return keep, pair_target


def stratified_k(cands: list, k: int, gen: torch.Generator) -> list[str]:
    """Pick k paths from [(path, score_norm)], one per score_norm QUANTILE stratum.

    Sort by score_norm, cut into k equal-count strata, draw one member per stratum
    with `gen` -- deterministic given the generator, so a resume replays the exact
    same draw. Pools with <= k candidates contribute everything (clamp+mask policy).
    """
    n = len(cands)
    if n <= k:
        return [p for p, _ in cands]
    srt = sorted(cands, key=lambda t: (t[1], t[0]))
    strata: list[list] = [[] for _ in range(k)]
    for i, c in enumerate(srt):
        strata[i * k // n].append(c)
    out = []
    for st in strata:
        pick = int(torch.randint(0, len(st), (1,), generator=gen).item())
        out.append(st[pick][0])
    return out


class FGResQPoolDataset(Dataset):
    """One ITEM = one pool: ref image + its candidates + the pairwise vote matrices.

    __getitem__ returns (ref_img, [cand_imgs...], votes (m,m), mask (m,m)) where
    votes[i, j] = 1.0 <=> the human preferred candidate j over candidate i (0.0 the
    other way; ties and unsupervised pairs are masked off). This orientation matches
    the training logit z = (d_i - d_j)/tau: target 1 pushes d_i > d_j, i.e. j closer
    to the ref, i.e. j preferred. One aspect bucket PER POOL PER EPOCH so ref and all
    candidates share one (h, w) and a batch is bucket-uniform.
    """

    def __init__(self, root, image_size: int = IMG_SIZE, aspect_buckets=None,
                 color_jitter: float = 0.0, hflip: float = 0.0, k: int = 6, exclude_scene34: bool = True,
                 split_version: int = 2, val_fraction: float = 0.08,
                 val_seed: int = 7919, verbose: bool = True, soft_gap_ref: float = 0.0):
        self.pools, self.pair_target = build_fgresq_pools(
            root, exclude_scene34=exclude_scene34, val_fraction=val_fraction,
            val_seed=val_seed, split_version=split_version, verbose=verbose,
            soft_gap_ref=soft_gap_ref)
        self.root = Path(root)
        self._pt_cache = _pt_cache_root(self.root)
        self.image_size = image_size
        self.aspect_buckets = tuple(aspect_buckets) if aspect_buckets else None
        self.color_jitter = float(color_jitter)
        self.hflip = float(hflip)
        self.k = int(k)
        # ---- 288 PIXEL CACHE, NOW WIRED INTO **TRAINING** (FIX 2026-09-28) ----------------
        # THE DEFECT: PixelCache288 was constructed ONLY in TwoAFCDataset (the EVAL class), so
        # every sq288 pool arm TRAINED on `im.resize((w,h), BILINEAR)` -- which SQUASHES the
        # aspect ratio to 288x288 -- while being EVALUATED on SquareCrop fractional Lanczos3.
        # The two agree on the 67% natively-square sources and MISMATCH on the rest, so the
        # "retired" squash defect was still live in TRAINING, and the cache dir name
        # (was-cached-as-trainer-sees-it-no-augs-...) was false: it was what the EVALUATOR saw.
        # Proven by a launch with FGRESQ_PT_288 pointed at another cache logging a step-1 loss
        # BIT-IDENTICAL to the incumbent's (0.435482) -- identical config+seed reproduces step 1
        # exactly, so identical step 1 means identical pixels.
        # GATE: fixed-square geometry ONLY (`aspect_buckets is None`). An aspect-bucket run has
        # h != w and must keep decoding+resizing exactly as before; the 544 class is untouched.
        # A MISS falls back to the live path (see PixelCache288.get), so a partial cache cannot
        # silently truncate a run.
        self._pc288 = None
        if self.aspect_buckets is None and int(image_size) == 288 and self.pools:
            from .pixel_cache import PixelCache288          # local, as in TwoAFCDataset
            pc = PixelCache288(
                self.root, out=image_size,
                dtype=os.environ.get("FGRESQ_PT_288_DTYPE", "float32"),
                verify_rel=self.pools[0]["ref"], verbose=False)
            # dir is None => the cache is OFF; keep _pc288 None so the fast path is skipped
            # entirely rather than counting a miss per lookup.
            self._pc288 = pc if pc.dir is not None else None
            print(f"[fgresq-pool] {pc.describe()}")
        self._geom = {}
        self._geom_seed = None

    def __len__(self) -> int:
        return len(self.pools)

    def assign_epoch_geometry(self, seed: int) -> None:
        if not self._geom or self._geom_seed != int(seed):
            n_b = len(self.aspect_buckets) if self.aspect_buckets else 1
            g = torch.Generator().manual_seed(int(seed))
            draws = torch.randint(0, max(1, n_b), (len(self.pools),), generator=g)
            fallback = (self.image_size, self.image_size)
            self._geom = {
                i: (self.aspect_buckets[int(draws[i]) % n_b] if self.aspect_buckets
                    else fallback)
                for i in range(len(self.pools))}
            self._geom_seed = int(seed)

    def bucket_of(self, idx: int) -> tuple[int, int]:
        return self._geom[idx]

    def _load(self, rel: str, h: int, w: int, jf: dict | None) -> torch.Tensor:
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        # FIX 2026-09-28: on the fixed-square geometry, serve the SAME 288 cache the evaluator
        # uses, so TRAIN and EVAL see the same pixels. The cached tensor is pre-jitter and
        # pre-flip, so jitter/hflip below compose exactly as they did before.
        if self._pc288 is not None and h == w == self.image_size:
            t = self._pc288.get(str(rel))
            if t is not None:
                t = self._pc288.as_model_input(t)
                if jf is not None:
                    t = TwoAFCDataset._jitter(t, jf)
                return t
        # Fast path: pre-decoded uint8 CHW .pt tensor via _open_image (see _pt_cache_root).
        # Skips only the DECODE -- the resize/jitter below run on identical pixels.
        im = _open_image(self.root, rel, self._pt_cache)
        t = to_tensor(im.resize((w, h), Image.BILINEAR))
        if jf is not None:
            t = TwoAFCDataset._jitter(t, jf)
        return t

    def __getitem__(self, idx: int):
        pool = self.pools[idx]
        h, w = self._geom[idx]
        g = torch.Generator().manual_seed((self._geom_seed or 0) * 1000003 + idx)
        drawn = stratified_k(pool["cands"], self.k, g)
        jf = None
        if self.color_jitter > 0:
            jg = torch.Generator().manual_seed((self._geom_seed or 0) * 1000003 + idx + 7)
            jf = TwoAFCDataset._draw_jitter(self, jg)
        ref_img = self._load(pool["ref"], h, w, jf)
        cand_imgs = [self._load(p, h, w, jf) for p in drawn]
        # HORIZONTAL FLIP of the WHOLE POOL -- ref AND every candidate, one decision per pool.
        # This is the class the training path actually uses (pool_batch=True), and it has NO
        # `split` attribute because it is train-only, so "train" is passed literally. Mirroring
        # ref and candidates together preserves every pairwise vote; mirroring only some would
        # silently corrupt the labels instead of augmenting them. Deterministic in (geometry
        # seed, idx), and the pool sampler advances that seed per epoch, so a given pool is seen
        # mirrored in some epochs and not others.
        if self.hflip > 0.0:
            pool_imgs = _hflip_sample([ref_img] + cand_imgs, self.hflip,
                                      self._geom_seed, idx, "train")
            ref_img, cand_imgs = pool_imgs[0], pool_imgs[1:]
        m = len(drawn)
        # D2 listnet inputs: per-candidate score_norm and vote-share weight.
        sc = dict(pool["cands"])
        scores = torch.tensor([float(sc.get(p, 0.0)) for p in drawn])
        nv = pool.get("nvotes") or {}
        counts = [max(1, int(nv.get(p, 1))) for p in drawn]
        cmax = max(counts)
        vote_w = torch.tensor([c / cmax for c in counts])
        votes = torch.zeros(m, m)
        mask = torch.zeros(m, m, dtype=torch.bool)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                t = self.pair_target.get((drawn[i], drawn[j]))
                if t is not None and t != 0.5:
                    votes[i, j] = t
                    mask[i, j] = True
        return ref_img, cand_imgs, votes, mask, scores, vote_w


def collate_pools(batch, k: int):
    """Pad each pool's candidate block to exactly k (clamp+mask policy: pools with
    < k candidates get their extra slots MASKED OFF, never dropped silently).

    D2: also emits `scores` (B,k) score_norm (0-padded), `cand_valid` (B,k) bool
    (padded slots False) and `vote_w` (B,k) vote-share weights (padded 0.0).
    """
    B = len(batch)
    refs = torch.stack([b[0] for b in batch])
    cands = torch.zeros(B, k, *refs.shape[1:])
    votes = torch.zeros(B, k, k)
    mask = torch.zeros(B, k, k, dtype=torch.bool)
    scores = torch.zeros(B, k)
    cand_valid = torch.zeros(B, k, dtype=torch.bool)
    vote_w = torch.zeros(B, k)
    for b, item in enumerate(batch):
        _ref, cs, v, mk = item[0], item[1], item[2], item[3]
        sc = item[4] if len(item) > 4 else torch.ones(len(cs))
        vw = item[5] if len(item) > 5 else torch.ones(len(cs))
        m = min(k, len(cs))
        for j in range(m):
            cands[b, j] = cs[j]
        votes[b, :m, :m] = v
        mask[b, :m, :m] = mk
        scores[b, :m] = sc[:m]
        vote_w[b, :m] = vw[:m]
        cand_valid[b, :m] = True
    return refs, cands, votes, mask, scores, cand_valid, vote_w


class PoolSampler:
    """Batch sampler for pool mode: groups POOLS by their per-epoch aspect bucket and
    yields lists of pool indices. Same discipline as AspectBatchSampler -- the plan is
    deterministic in seed+epoch (a resume replays it) and skip_batches slices off the
    consumed prefix at the sampler level so nothing is decoded and discarded."""

    def __init__(self, ds, batch_size: int, seed: int = 1234, epoch: int = 0,
                 skip_batches: int = 0, shuffle: bool = True):
        self.ds = ds
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.skip_batches = skip_batches
        self.shuffle = shuffle

    def _plan(self) -> list[list[int]]:
        self.ds.assign_epoch_geometry(self.seed + self.epoch)
        g = torch.Generator().manual_seed((self.seed + self.epoch) * 7919 + 13)
        order = torch.randperm(len(self.ds), generator=g).tolist()
        groups: dict[tuple[int, int], list[int]] = {}
        for i in order:
            groups.setdefault(self.ds.bucket_of(i), []).append(i)
        batches: list[list[int]] = []
        keys = list(groups)
        if self.shuffle:
            keys = [keys[p] for p in torch.randperm(len(keys), generator=g).tolist()]
        for kk in keys:
            items = groups[kk]
            for s in range(0, len(items), self.batch_size):
                batches.append(items[s:s + self.batch_size])
        if self.shuffle:
            batches = [batches[p] for p in torch.randperm(len(batches), generator=g).tolist()]
        return batches

    def __iter__(self):
        return iter(self._plan()[self.skip_batches:])

    def __len__(self) -> int:
        return max(0, len(self._plan()) - self.skip_batches)


class SyntheticPoolDataset(Dataset):
    """Random pools with a learnable preference structure, for CPU/GPU smoke tests.

    Candidate j = ref + (1 - score_j) * 0.15 * noise, so a HIGHER score really is
    closer to the ref and the vote table is consistent with the geometry -- a model
    that learns anything should beat chance, while a random one wobbles around 0.5
    without landing on exactly 0.0 or 0.5."""

    def __init__(self, n_pools: int = 32, k: int = 6, seed: int = 0,
                 shape: tuple[int, int] | None = None, size: int = IMG_SIZE):
        self.n, self.k, self.seed = n_pools, k, seed
        self.shape = shape or (size, size)
        g = torch.Generator().manual_seed(seed)
        self.scores = torch.rand(n_pools, k, generator=g)

    def __len__(self) -> int:
        return self.n

    def assign_epoch_geometry(self, seed: int) -> None:
        self._geom_seed = int(seed)

    def bucket_of(self, idx: int) -> tuple[int, int]:
        return tuple(self.shape)

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        h, w = self.shape
        ref = torch.rand(3, h, w, generator=g)
        s = self.scores[idx]
        cands = [torch.clamp(ref + (1.0 - float(sj)) * 0.15
                             * torch.randn(ref.shape, generator=g), 0, 1) for sj in s]
        m = self.k
        votes = torch.zeros(m, m)
        mask = torch.zeros(m, m, dtype=torch.bool)
        for i in range(m):
            for j in range(m):
                if i != j and s[j] != s[i]:
                    votes[i, j] = 1.0 if s[j] > s[i] else 0.0
                    mask[i, j] = True
        # D2: scores = the ground-truth score vector; vote_w all ones (no vote counts
        # exist synthetically).
        return ref, cands, votes, mask, s.clone(), torch.ones(m)
