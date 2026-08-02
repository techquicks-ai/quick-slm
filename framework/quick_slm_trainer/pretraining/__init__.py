"""Stage one: build the 10B-token corpus, then read windows off it.

  build_source -> stream_source -> combine_sources -> WindowedMemmapDataset

The corpus half (`sources`, `packing`, `combine`) needs numpy and nothing else.
`dataset` subclasses `torch.utils.data.IterableDataset` and cannot be defined
without torch.

Importing this package eagerly would therefore drag torch into
`01_data_preparation.ipynb`, which is network-bound, runs happily on a CPU
runtime, and has no business owning a CUDA build. The dataset is resolved on
first attribute access instead, so `from quick_slm_trainer.pretraining import
stream_source` works with torch absent and `... import WindowedMemmapDataset`
raises only when it is genuinely needed.

Everything both stages share stays one level up: the model, the trainer, the
checkpointer, the LR schedule, the optimizer, the evaluator, and `loader.py`.

One consequence worth knowing: `from quick_slm_trainer.pretraining import *`
resolves every name in `__all__`, including the dataset, and so does need torch.
Importing the names you want does not. That is the honest failure, not a bug: a
star-import asked for the dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .combine import CombineRefused, combine_sources, mirror_to_local, plan_allocation
from .packing import encode_batch, stream_source, sync_local_to_drive
from .sources import SOURCES, Source, build_source

if TYPE_CHECKING: # pragma: no cover - for type checkers only
  from .dataset import WindowedMemmapDataset

_LAZY = {"WindowedMemmapDataset"}


def __getattr__(name: str):
  if name in _LAZY:
    from . import dataset

    return getattr(dataset, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(__all__)


__all__ = [
  "CombineRefused",
  "SOURCES",
  "Source",
  "WindowedMemmapDataset",
  "build_source",
  "combine_sources",
  "encode_batch",
  "mirror_to_local",
  "plan_allocation",
  "stream_source",
  "sync_local_to_drive",
]
