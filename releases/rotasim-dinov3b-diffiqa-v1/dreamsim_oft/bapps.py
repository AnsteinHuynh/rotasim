"""[Begin Work Zone]
BAPPS 2afc as a TRAINING corpus, at our 288 square class.

WHY THIS FILE EXISTS: every trained arm in this repo so far used FGResQ (a restoration-quality
corpus with ~a few thousand rows). BAPPS train is 151,400 human 2AFC judgments over three
tracks -- an order of magnitude more rows, a DIFFERENT question (which of two distortions on
ONE reference looks better), and the corpus the LPIPS/ST-LPIPS reference metrics were trained
on. Multi-corpus training ("train on N corpora") starts here.

TARGET CONVENTION -- verified, do not "fix" it: BAPPS `judge.npy` holds P(humans preferred p1)
(with 0.5 = a tie). `loss.py` defines `target >= 0.5` as "x1 (the RIGHT candidate) is more
similar", so `target = judge` drops in with NO relabeling. The evaluator's rule is the mirror
image of this: `ok = ((target >= 0.5) == (d1 < d0))`.

PREPROCESS: BAPPS images are 256x256 square RGB/RGBA, so the 288 class's SquareCrop is a clean
upsample with NO crop -- and it is the SAME transform scripts and probes use on BAPPS val, so a
model trained here is evaluated on exactly the pixels it trained on.

INDEX CACHE: reading 151,400 tiny judge .npy files + hashing 151,400 ref PNGs costs ~1-2 min,
and on Windows every spawned DataLoader worker re-runs __init__. So the index is built once and
cached at <split>/index_bapps_v1.json (delete it to rebuild).
[End Work Zone]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import IMG_SIZE, _hflip_sample

INDEX_NAME = "index_bapps_v1"          # real cache file: index_bapps_v1-<tracks>.json
DEFAULT_TRACKS = ("cnn", "mix", "traditional")
ROLES = ("ref", "p0", "p1")


def _cache_path(split_dir: Path, tracks) -> Path:
    """Cache PER TRACK SET.

    A single shared filename looked harmless and was a trap: opening the split with
    tracks=("cnn",) first wrote a cnn-only index, and the TRAINING config (all three tracks)
    then hit the tracks guard and died at startup. Keying the filename by the track set makes
    every request self-consistent while the guard below stays as a second line of defence.
    """
    key = "+".join(tracks) or "ALL"
    return split_dir / f"{INDEX_NAME}-{key}.json"


def _build_index(split_dir: Path, tracks, image_size: int, verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    for track in tracks:
        tdir = split_dir / track
        if not tdir.is_dir():
            print(f"[bapps] track {track!r}: NOT PRESENT ({tdir}) -- skipped")
            continue
        ids = sorted(p.stem for p in (tdir / "p0").glob("*.png"))
        kept = 0
        for tid in ids:
            if not all((tdir / r / f"{tid}.png").is_file() for r in ROLES):
                continue
            jp = tdir / "judge" / f"{tid}.npy"
            if not jp.is_file():
                continue
            judge = float(np.load(jp).reshape(-1)[0])
            cluster = hashlib.sha1((tdir / "ref" / f"{tid}.png").read_bytes()).hexdigest()[:16]
            rows.append({"track": track, "id": tid, "judge": judge, "cluster": cluster})
            kept += 1
        print(f"[bapps] track {track}: {kept}/{len(ids)} usable rows")
    return rows


class BAPPSDataset(Dataset):
    """One item = one human 2AFC judgment: (ref, p0, p1, judge, idx)."""

    def __init__(self, root, split: str = "train", tracks=DEFAULT_TRACKS,
                 image_size: int = IMG_SIZE, hflip: float = 0.0, limit: int | None = None,
                 verbose: bool = True):
        self.root = Path(root)
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"{self.split_dir} not found (expected <root>/train etc.)")
        self.split = split
        self.tracks = tuple(tracks)
        self.image_size = int(image_size)
        self.hflip = float(hflip)
        self._geom_seed = None
        self.aspect_buckets = None  # square class only: no per-epoch geometry
        self._tf = None

        cache = _cache_path(self.split_dir, self.tracks)
        tracks_key = ",".join(self.tracks)
        if cache.is_file():
            blob = json.loads(cache.read_text())
            if blob.get("tracks") != tracks_key:
                raise ValueError(
                    f"{cache} was built for tracks={blob.get('tracks')!r} but "
                    f"{tracks_key!r} was requested; delete it or use the cached tracks")
            rows = blob["rows"]
        else:
            rows = _build_index(self.split_dir, self.tracks, self.image_size, verbose=verbose)
            cache.write_text(json.dumps({"tracks": tracks_key, "rows": rows}))
            if verbose:
                print(f"[bapps] index written: {cache} ({len(rows)} rows)")

        if limit:
            rows = rows[:limit]
        self.rows = rows
        if verbose:
            j = np.array([r["judge"] for r in rows], dtype=np.float64)
            u, c = np.unique(j, return_counts=True)
            hist = ", ".join(f"{v:g}x{n}" for v, n in zip(u[:6], c[:6]))
            print(f"[bapps] split={split!r} tracks={tracks_key or 'ALL'} rows={len(rows)} "
                  f"refs={len({r['cluster'] for r in rows})} judge[{hist}]")

    # -- interface used by the trainer (mirrors TwoAFCDataset) -----------------
    @property
    def transform(self):
        if self._tf is None:
            from .resize import SquareCrop

            self._tf = SquareCrop(self.image_size)
        return self._tf

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from PIL import Image

        r = self.rows[idx]
        tdir = self.split_dir / r["track"]
        imgs = []
        for role in ROLES:
            with Image.open(tdir / role / f"{r['id']}.png") as im:
                t = self.transform(im.convert("RGB"))
            imgs.append(t)
        imgs = _hflip_sample(imgs, self.hflip, self._geom_seed, idx, self.split)
        target = torch.tensor(float(r["judge"]), dtype=torch.float32)
        return imgs[0], imgs[1], imgs[2], target, idx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"I:\MyApps\sd-train\BAPPS")
    p.add_argument("--split", default="train")
    p.add_argument("--tracks", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--image-size", type=int, default=288)
    a = p.parse_args()
    tracks = tuple(t for t in a.tracks.split(",") if t) or DEFAULT_TRACKS
    ds = BAPPSDataset(a.root, split=a.split, tracks=tracks, image_size=a.image_size)
    print(f"[selftest] len={len(ds)}")
    for i in (0, len(ds) // 2, len(ds) - 1):
        ref, p0, p1, tgt, tid = ds[i]
        assert ref.shape == (3, a.image_size, a.image_size), ref.shape
        assert ref.dtype == torch.float32
        same = torch.equal(p0, p1)
        print(f"  [{i}] id={ds.rows[i]['id']} track={ds.rows[i]['track']} target={tgt.item():g} "
              f"shape={tuple(ref.shape)} range=[{ref.min():.3f},{ref.max():.3f}] "
              f"p0==p1: {same} (expected False)")
        assert not same
    from .resize import RESIZE_TAG

    print(f"[selftest] OK resize_tag={RESIZE_TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
