"""The dataset the SFT loop reads.

Yields `(input_ids, labels)`, the same contract as
`pretraining.WindowedMemmapDataset`, with everything outside an assistant turn
masked to `IGNORE_INDEX`. One trainer therefore serves both stages.

The packed corpus is stored as two files rather than one. `ids` is the uint16
token stream, exactly as in pretraining. `mask` is a parallel uint8 array, 1
where the token is scored. Labels are reconstructed as `where(mask, ids, -100)`.
Storing labels directly would need int32, since -100 does not fit in uint16, and
would cost four bytes per token instead of one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..loader import shard
from ..template import IGNORE_INDEX


class MaskedWindowDataset(IterableDataset):
  """Windows over the packed SFT corpus, with the loss mask applied.

  Windows are shuffled per epoch. The packer already shuffled *examples* before
  binning them, but a window is a bin of several whole examples and the bin
  order is fixed on disk; reshuffling here decorrelates the order the optimizer
  sees across the three SFT epochs.
  """

  def __init__(
    self,
    ids_path: str | Path,
    mask_path: str | Path,
    ctx: int,
    *,
    start_window: int = 0,
    seed: int = 1337,
    shuffle: bool = True,
  ):
    self.ids_path = str(ids_path)
    self.mask_path = str(mask_path)
    self.ctx = int(ctx)
    self.start_window = int(start_window)
    self.seed = int(seed)
    self.shuffle = shuffle

    n_ids = os.path.getsize(self.ids_path) // 2
    n_mask = os.path.getsize(self.mask_path)
    if n_ids != n_mask:
      raise ValueError(
        f"ids/mask length mismatch: {n_ids:,} tokens vs {n_mask:,} mask bytes. "
        "The pack step writes them together; one of the two is truncated."
      )
    self.n_windows = n_ids // self.ctx

  def __iter__(self):
    ids = np.memmap(self.ids_path, dtype=np.uint16, mode="r")
    mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
    order = np.arange(self.n_windows)
    if self.shuffle:
      np.random.default_rng(self.seed).shuffle(order)
    for w in shard(order, self.start_window):
      s = int(w) * self.ctx
      x = ids[s : s + self.ctx].astype(np.int64)
      m = mask[s : s + self.ctx]
      y = np.where(m.astype(bool), x, IGNORE_INDEX)
      yield torch.from_numpy(x), torch.from_numpy(y)
