"""The dataset the pretraining loop reads.

Yields `(input_ids, labels)` with `labels = input_ids`, which is the contract
`sft.MaskedWindowDataset` also honours. That shared contract is the reason one
trainer serves both stages: if the training loop built labels itself, SFT would
need a forked loop and the two would drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..loader import shard


class WindowedMemmapDataset(IterableDataset):
  """Fixed-size windows over a corpus, in shuffled order.
  
  Can read from a single `combined.bin` file or a directory containing 
  multiple `combined_*.bin` chunk files.
  
  The shuffle seed folds in the resume step, so a run that restarts does not
  re-walk the windows it already consumed in the same order.

  > **EDUCATIONAL DEFECT: The DataLoader Bug**
  > In the original run, this class did not use `shard` correctly across workers. 
  > Because PyTorch copies an `IterableDataset` into every DataLoader worker, all 4 
  > workers evaluated `np.random.default_rng(self.seed).shuffle(order)` with the EXACT 
  > SAME SEED. The result? All 4 workers walked the exact same permutation simultaneously, 
  > and the DataLoader served every batch four times consecutively. This slashed the 
  > effective gradient diversity by 4x. It is fixed now via the `shard` utility, which 
  > slices the array correctly by `worker_id`.
  """

  def __init__(self, path: str | Path, ctx: int, *, start_window: int = 0, seed: int = 1337):
    self.path = Path(path)
    self.ctx = int(ctx)
    self.start_window = int(start_window)
    self.seed = int(seed)
    
    if self.path.is_dir():
      # Support chunked data directories
      self.files = sorted(self.path.glob("combined*.bin"))
      if not self.files:
        raise RuntimeError(f"No combined*.bin files found in {self.path}")
    else:
      self.files = [self.path]
      
    self.file_windows = []
    for f in self.files:
      size_bytes = os.path.getsize(f)
      w = (size_bytes // 2) // self.ctx
      self.file_windows.append(w)
      
    self.n_windows = sum(self.file_windows)

  def __iter__(self):
    # Open memmaps for all files
    arrs = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
    
    # Build global index to (file_idx, window_idx) mapping
    # We can just use an array of file indices
    file_indices = np.concatenate([
      np.full(w, i, dtype=np.int16) for i, w in enumerate(self.file_windows)
    ])
    window_indices = np.concatenate([
      np.arange(w, dtype=np.int32) for w in self.file_windows
    ])
    
    order = np.arange(self.n_windows)
    np.random.default_rng(self.seed).shuffle(order)
    
    for w in shard(order, self.start_window):
      global_w = order[w]
      f_idx = file_indices[global_w]
      w_idx = window_indices[global_w]
      
      s = int(w_idx) * self.ctx
      x = torch.from_numpy(arrs[f_idx][s : s + self.ctx].astype(np.int64))
      yield x, x.clone()
