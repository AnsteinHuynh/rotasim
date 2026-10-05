"""Minimal reader for the rotasim sharded pixel cache (format v2, 'RTSM').

Used by the EVAL path (data.py::_rotasim_eval_get) when ROTASIM_EVAL_SHARDS points
at the rotasim root -- serves protocol-identical fraclanczos3-288 tensors from
NVMe shards instead of decoding source images (which may live on slow media after
the corpus move). Writer/builder lives in scratch\\sophie\\rotasim (rotasim_shard.py);
this module is the frozen read-only subset.

Shard format (little-endian): b'RTSM' | u16 version=2 | u32 n |
n x (32B sha256 + u64 offset + u32 length) | blobs (torch-serialized tensors).
"""
from __future__ import annotations

import csv
import io
import os
import struct
from pathlib import Path

MAGIC = b"RTSM"
VERSION = 2
_HEADER = 4 + 2 + 4
_REC = 32 + 8 + 4


def read_shard_index(path: Path):
    with open(path, "rb") as f:
        head = f.read(_HEADER)
        if head[:4] != MAGIC:
            raise ValueError(f"{path}: bad magic")
        ver, n = struct.unpack("<HI", head[4:])
        if ver != VERSION:
            raise ValueError(f"{path}: version {ver}")
        raw = f.read(n * _REC)
    for i in range(n):
        b = i * _REC
        yield raw[b:b + 32], *struct.unpack("<QI", raw[b + 32:b + 44])


def read_record(path: Path, offset: int, length: int):
    import torch
    with open(path, "rb") as f:
        f.seek(offset)
        return torch.load(io.BytesIO(f.read(length)), weights_only=False)


class RotasimShardSource:
    """orig_relpath -> tensor, via a manifest + shard files. Read-only."""

    def __init__(self, rotasim_root: str | Path, dataset: str, split: str):
        import torch  # noqa: F401  (records are torch-serialized)
        root = Path(rotasim_root)
        man = root / f"manifest_{dataset}-{split}.csv"
        self.data_dir = root / dataset / split
        self.loc: dict[str, tuple[Path, int, int]] = {}
        with man.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.loc[r["orig_relpath"]] = (self.data_dir / r["shard"],
                                               int(r["offset"]), int(r["length"]))
        self._fds: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.loc)

    def get(self, rel: str):
        """Returns the tensor for a corpus-relative path, or None if absent."""
        hit = self.loc.get(str(rel).replace("\\", "/"))
        if hit is None:
            return None
        path, off, ln = hit
        fd = self._fds.get(path.name)
        if fd is None:
            if len(self._fds) > 8:
                for f in self._fds.values():
                    f.close()
                self._fds.clear()
            fd = open(path, "rb")
            self._fds[path.name] = fd
        fd.seek(off)
        return _load(fd, ln)

    def close(self) -> None:
        for f in self._fds.values():
            f.close()
        self._fds.clear()


def _load(fd, length: int):
    import torch
    return torch.load(io.BytesIO(fd.read(length)), weights_only=False)
