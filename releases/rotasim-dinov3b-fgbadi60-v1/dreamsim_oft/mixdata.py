"""Confluence mix3: the multi-corpus mixture dataset (2026-09-30, kerok directive).

One training item = one pairwise (ref, p0, p1, target) row drawn from an EQUAL-THIRDS
mixture of three corpora (FGResQ sv2 excl34 / BAPPS train / DiffIQA train cc-types).
Design notes:

* Equal thirds BY CORPUS, not by row count (151,400 BAPPS vs ~44K FGResQ would otherwise
  give BAPPS 3.4x the gradient mass). Each corpus contributes `slots_per_corpus` virtual
  slots; slot j of corpus i maps to row  prng(seed, i, j) % len(ds_i).  The loader's
  own RandomSampler shuffles the virtual index space, so batches are mixed and the plan
  is a pure function of (seed, lengths) -- reproducible, epoch-independent.
* All three datasets return the SAME tuple shape (ref, p0, p1, target, idx) with the SAME
  target convention (>= 0.5 == "p1 more similar/better") -- see bapps.py (target = judge)
  and diffiqa.py (target = 1 - gt). FGResQDataset is the pairwise eval class and carries
  the same convention; the d0 line trained on it directly.
* One epoch = 3 * slots_per_corpus / batch steps; with slots=60,000 and b60t that is
  3,000 steps/epoch, so a 600-step run never crosses an epoch boundary (avoids the
  worker-respawn boundary entirely).
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset

SLOTS_PER_CORPUS = 60_000


class Mix3Dataset(Dataset):
    """Equal-thirds mixture; __getitem__(i) -> (corpus_idx, row) via a fixed plan."""

    NAMES = ("fgresq", "bapps", "diffiqa")

    def __init__(self, fgresq_ds, bapps_ds, diffiqa_ds, seed: int = 0,
                 slots_per_corpus: int = SLOTS_PER_CORPUS, verbose: bool = True):
        self.dss = (fgresq_ds, bapps_ds, diffiqa_ds)
        self.seed = int(seed)
        self.slots = int(slots_per_corpus)
        # deterministic plan: slot j -> row
        self._plan = []
        for ci, ds in enumerate(self.dss):
            rng = random.Random(f"mix3-{seed}-{ci}")
            rows = [rng.randrange(len(ds)) for _ in range(self.slots)]
            self._plan.append(rows)
        if verbose:
            print(f"[mix3] corpora: " + " ".join(
                f"{self.NAMES[i]}={len(ds)} rows" for i, ds in enumerate(self.dss))
                + f"; {self.slots} slots each = {3 * self.slots} virtual rows (equal thirds)")

    def __len__(self) -> int:
        return 3 * self.slots

    def __getitem__(self, idx: int):
        ci, j = divmod(idx, self.slots)
        return self.dss[ci][self._plan[ci][j]]
