"""DiffIQA dataset reader (dreamsim_oft).

DiffIQA (A-FINE, CVPR 2025): reference-conditioned pairwise quality preferences over
512x512 crops enhanced by a PASD-style SD-backbone diffusion enhancer. Full triage:
scratch/sophie/diffiqa/DIFFIQA_TRIAGE.md.

Layout after the 2026-09-30 extraction (Google-Drive-named archives under
<root>/{Train,Test,Validation}, each extracted to {zipname}-ext/):

    <root>/Train/TrainImage-006-ext/images/{01..06,Original}/*.png
    <root>/Train/trainlabel-ext/[trainlabel/]TripletEachType/{NYY,PNY,PSY,PYY,SNY,SSY,SYY}.txt
    <root>/Test/TestImage-011-ext/images/...        + testlabel-ext/...
    <root>/Validation/ValidationImage-007-ext/images/... + validationlabel-ext/...

Label semantics (verified 350,664/350,664 rows in the triage): each row is
img1,img2,ref,gt[,l1,l2]; gt=1 means the FIRST-listed candidate is better, so our
convention (target >= 0.5 == "p1 = RIGHT candidate more similar") needs

    p0 = img1, p1 = img2, ref = ref, target = 1 - gt

WHY CANDIDATE-VS-CANDIDATE ONLY (measured 2026-09-30, full Test, n=70,132):
the *YY types (50.6% of rows) put the REFERENCE itself in the candidate slot.
d(ref, ref) = 0 is the global minimum of a reference-distance metric, so the ref
always wins: PYY (variant judged BETTER than its ref) scores exactly 0% for ANY
metric of this family -- frozen DINOv3-B 52.42% overall / 47.13% cc-nontie, and the
BAPPS-trained arm 52.52 / 47.38, i.e. the corpus's quality axis is orthogonal to
reference-closeness (PSY, "better vs similar", is INVERTED at 16.8%: enhancements
that improve on the ref are perceptually FAR from it). We therefore train only on
the four cc types PNY/PSY/SNY/SSY (ties included -- SSY is the tie type, target 0.5
pulls the logit to no-preference, same role as BAPPS ties).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import IMG_SIZE, _hflip_sample

DEFAULT_SPLIT = "train"
CC_TYPES = ("PNY", "PSY", "SNY", "SSY")   # candidate-vs-candidate only; *YY excluded
INDEX_NAME = "index_diffiqa_v1"           # real cache file: index_diffiqa_v1-<split>-<types>.json
SPLIT_DIRS = {
    "train": ("Train", "TrainImage-006-ext", "trainlabel-ext"),
    "val": ("Validation", "ValidationImage-007-ext", "validationlabel-ext"),
    "test": ("Test", "TestImage-011-ext", "testlabel-ext"),
}
ROLES = ("ref", "p0", "p1")


def _resolve(images_dir: Path, token: str) -> Path:
    token = token.replace("\\", "/").strip()
    stem = token[:-4] if token.endswith(".png") else token
    if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
        variant = stem.rsplit("_", 1)[1]
    else:
        variant = "Original"
    if "/" in token:
        cand = images_dir / token
        if cand.is_file():
            return cand
    return images_dir / variant / f"{stem}.png"


def _build_index(root: Path, split: str, types=CC_TYPES, image_size: int = IMG_SIZE,
                 verbose: bool = True) -> list[dict]:
    top, img_dirname, lab_dirname = SPLIT_DIRS[split]
    images_dir = root / top / img_dirname / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found (expected the {img_dirname}-ext tree)")
    lab_root = root / top / lab_dirname
    tet = sorted(lab_root.rglob("TripletEachType"))
    if not tet:
        raise FileNotFoundError(f"TripletEachType/ not found under {lab_root}")
    tet = tet[0]

    rows: list[dict] = []
    targets: list[float] = []
    for t in types:
        f = tet / f"{t}.txt"
        n0 = len(rows)
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = line.split(",")
            ref = _resolve(images_dir, r[2])
            p0 = _resolve(images_dir, r[0])
            p1 = _resolve(images_dir, r[1])
            if not (ref.is_file() and p0.is_file() and p1.is_file()):
                continue
            target = 1.0 - float(r[3])
            rows.append({"type": t, "ref": str(ref), "p0": str(p0), "p1": str(p1),
                         "target": target,
                         "cluster": ref.stem.rsplit("_", 1)[0]})
            targets.append(target)
        if verbose:
            print(f"[diffiqa] type {t}: {len(rows) - n0} usable rows")
    return rows


def _cache_path(root: Path, split: str, types) -> Path:
    key = "+".join(types) or "ALL"
    top, _, lab_dirname = SPLIT_DIRS[split]
    # cache lives inside the split's Drive-named label folder (Train/Test/Validation)
    return root / top / lab_dirname / f"{INDEX_NAME}-{key}.json"


class DiffIQADataset(Dataset):
    """One item = one cc-type judgment: (ref, p0, p1, target=1-gt, idx)."""

    def __init__(self, root, split: str = DEFAULT_SPLIT, types=CC_TYPES,
                 image_size: int = IMG_SIZE, hflip: float = 0.0, limit: int | None = None,
                 verbose: bool = True):
        self.root = Path(root)
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(SPLIT_DIRS)}")
        self.split = split
        self.types = tuple(types)
        self.image_size = int(image_size)
        self.hflip = float(hflip)
        self._geom_seed = None
        self.aspect_buckets = None  # square class only
        self._tf = None

        cache = _cache_path(self.root, split, self.types)
        cache.parent.mkdir(parents=True, exist_ok=True)
        types_key = ",".join(self.types)
        if cache.is_file():
            blob = json.loads(cache.read_text())
            if blob.get("types") != types_key or blob.get("split") != split:
                raise ValueError(
                    f"{cache} was built for split={blob.get('split')!r} types={blob.get('types')!r} "
                    f"but split={split!r} types={types_key!r} was requested; delete it to rebuild")
            rows = blob["rows"]
            # 2026-10-01 corpus relocation: index caches built at the old I: home carry
            # I:-absolute row paths. Remap them onto THIS dataset's root so the corpus is
            # relocatable (the move to X:\unique\datasets otherwise crashes __getitem__).
            old_home = r"I:\MyApps\sd-train\DiffIQA"
            remapped = 0
            for _r in rows:
                for role in ROLES:
                    p = _r[role]
                    if isinstance(p, str) and p.startswith(old_home):
                        _r[role] = str(self.root.joinpath(p[len(old_home) + 1:]))
                        remapped += 1
            if remapped and verbose:
                print(f"[diffiqa] remapped {remapped} row paths: {old_home} -> {self.root}")
        else:
            rows = _build_index(self.root, split, self.types, self.image_size, verbose=verbose)
            cache.write_text(json.dumps({"split": split, "types": types_key, "rows": rows}))
            if verbose:
                print(f"[diffiqa] index written: {cache} ({len(rows)} rows)")

        if limit:
            rows = rows[:limit]
        self.rows = rows
        if verbose:
            t = np.array([r["target"] for r in rows], dtype=np.float64)
            u, c = np.unique(t, return_counts=True)
            hist = ", ".join(f"{v:g}x{n}" for v, n in zip(u[:6], c[:6]))
            print(f"[diffiqa] split={split!r} types={types_key} rows={len(rows)} "
                  f"crops={len({r['cluster'] for r in rows})} target[{hist}]")

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
        imgs = []
        for role in ROLES:
            with Image.open(r[role]) as im:
                t = self.transform(im.convert("RGB"))
            imgs.append(t)
        ref, p0, p1 = imgs
        # hflip: applied to the WHOLE triplet, so the 2AFC label survives (train split only)
        if self.hflip > 0.0 and self.split != "test":
            ref = _hflip_sample(ref, self.hflip, self._geom_seed, idx)
            p0 = _hflip_sample(p0, self.hflip, self._geom_seed, idx)
            p1 = _hflip_sample(p1, self.hflip, self._geom_seed, idx)
        # target as a 0-dim tensor: FGResQ/BAPPS return that shape, and a mixed
        # mix3 batch collates only if ALL corpora agree (a float here killed the
        # first mix3 smoke with 'expected Tensor as element N').
        return ref, p0, p1, torch.tensor(r["target"], dtype=torch.float32), idx


def _selftest(root: str, split: str, limit: int | None) -> None:
    ds = DiffIQADataset(root, split=split, limit=limit)
    print(f"[selftest] len={len(ds)}")
    for i in (0, len(ds) // 2, len(ds) - 1):
        ref, p0, p1, tgt, idx = ds[i]
        r = ds.rows[i]
        same_ref = (p0 == ref).all()
        print(f"  [{i}] type={r['type']} target={tgt:g} shapes={tuple(ref.shape)} "
              f"range=[{ref.min():.3f},{ref.max():.3f}] p0==ref: {same_ref} (expect False)")
        assert tuple(ref.shape) == (3, ds.image_size, ds.image_size)
        assert not same_ref
    from .resize import RESIZE_TAG
    print(f"[selftest] OK resize_tag={RESIZE_TAG}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--root", default=r"I:\MyApps\sd-train\DiffIQA")
    a.add_argument("--split", default=DEFAULT_SPLIT, choices=list(SPLIT_DIRS))
    a.add_argument("--limit", type=int, default=None)
    raise SystemExit(_selftest(a.parse_args().root, a.parse_args().split, a.parse_args().limit))
