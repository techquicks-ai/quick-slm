"""Worker sharding, and the DataLoader both stages build.

The two stages own their own datasets, `pretraining.WindowedMemmapDataset` and
`sft.MaskedWindowDataset`, because what a window means differs between them. What
does not differ is how a window reaches the GPU, so that lives here rather than
being written twice.

Worker sharding
---------------
An `IterableDataset` is copied into every DataLoader worker, and each copy runs
`__iter__` from the top. With a shuffle seeded identically in each copy, all
`num_workers` workers walk the *same* permutation, and the DataLoader, which
round-robins its workers, hands back `num_workers` consecutive duplicates of
every batch. At `num_workers=4` the corpus effectively shrinks four-fold and
every window is seen four times back to back, inside a single optimizer step.

The pretraining run shipped without this. `shard` is what prevents it. It is
not an optimisation, and `docs/important_notes.md` records what it cost.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def shard(order: np.ndarray, start_window: int) -> np.ndarray:
  """Drop the windows already consumed, then take this worker's stride.

  `start_window` counts sequences consumed globally across all workers, so
  slicing before sharding is what makes a resumed run pick up near where it
  stopped rather than `num_workers` times too early.
  """
  order = order[start_window:]
  info = torch.utils.data.get_worker_info()
  if info is None:
    return order
  return order[info.id :: info.num_workers]


def make_loader(
  dataset: IterableDataset,
  *,
  batch_size: int,
  num_workers: int = 4,
  prefetch_factor: int = 4,
) -> DataLoader:
  return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=num_workers > 0,
  )
