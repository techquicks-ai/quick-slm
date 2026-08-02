"""Held-out loss, weighted by scored tokens rather than by batch.

Averaging per-batch mean losses is only correct when every batch scores the same
number of tokens. That holds for pretraining, where every token is a target, and
fails for SFT, where one window may be four-fifths context and the next
four-fifths assistant turn. A mean of means would then quietly weight a window
with twelve scored tokens the same as one with three thousand.

Both evaluators here accumulate `sum(loss * n_scored)` and divide once, which is
the true corpus cross-entropy in both stages.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Iterator

from .template import IGNORE_INDEX


def _autocast(device: str, dtype):
  import torch

  device_type = "cuda" if device.startswith("cuda") else "cpu"
  return torch.amp.autocast(device_type, dtype=dtype, enabled=device_type == "cuda")


def scored_tokens(labels) -> int:
  """Targets the loss will actually be taken over.

  The causal shift means position `i` predicts `labels[i + 1]`, so the first
  label is never a target and the count must be taken after the shift. Getting
  this wrong biases every reported loss by a factor of `ctx / (ctx - 1)`,
  which at ctx=4096 is small enough to hide and large enough to matter when
  the number is quoted in a paper.
  """
  return int((labels[:, 1:] != IGNORE_INDEX).sum())


def evaluate_batches(model, batches: Iterable, *, device: str, dtype) -> dict:
  """Token-weighted cross-entropy over an iterable of `(input_ids, labels)`."""
  import torch

  was_training = model.training
  model.eval()
  total_loss, total_items = 0.0, 0
  with torch.no_grad():
    for x, y in batches:
      n = scored_tokens(y)
      if n == 0:
        continue
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      with _autocast(device, dtype):
        out = model(input_ids=x, labels=y)
      total_loss += float(out.loss.float().item()) * n
      total_items += n
  if was_training:
    model.train()

  if total_items == 0:
    raise RuntimeError("evaluation saw no scored tokens; the loss mask is empty")
  loss = total_loss / total_items
  return {"loss": loss, "ppl": math.exp(min(loss, 30.0)), "scored_tokens": total_items}


def memmap_eval_batches(
  path: str | Path,
  ctx: int,
  *,
  micro_batch: int,
  n_batches: int,
  seed: int = 99991,
) -> Iterator:
  """A deterministic random slice of `combined.bin`, as `(x, x)` pairs.

  The seed is fixed and unrelated to the training seed, so the same windows are
  scored at every eval and the curve is comparable across steps. Those windows
  are drawn from the training corpus, so this is a training-loss probe on
  unseen-this-step data, not a true held-out set. Pretraining at one epoch over
  10B tokens makes the distinction academic; SFT, at three epochs, uses the
  real validation split instead.
  """
  import numpy as np
  import torch

  arr = np.memmap(str(path), dtype=np.uint16, mode="r")
  n_windows = arr.size // ctx
  idx = np.random.default_rng(seed).choice(n_windows, size=n_batches * micro_batch, replace=False)

  for b in range(n_batches):
    rows = [
      arr[int(w) * ctx : (int(w) + 1) * ctx].astype(np.int64)
      for w in idx[b * micro_batch : (b + 1) * micro_batch]
    ]
    x = torch.from_numpy(np.stack(rows))
    yield x, x.clone()


def masked_eval_batches(
  ids_path: str | Path,
  mask_path: str | Path,
  ctx: int,
  *,
  micro_batch: int,
  max_batches: int | None = None,
) -> Iterator:
  """Every window of the packed SFT validation split, in order."""
  import numpy as np
  import torch

  ids = np.memmap(str(ids_path), dtype=np.uint16, mode="r")
  mask = np.memmap(str(mask_path), dtype=np.uint8, mode="r")
  if ids.size != mask.size:
    raise ValueError(f"ids/mask length mismatch: {ids.size:,} vs {mask.size:,}")

  n_windows = ids.size // ctx
  n_batches = n_windows // micro_batch
  if max_batches is not None:
    n_batches = min(n_batches, max_batches)

  for b in range(n_batches):
    xs, ys = [], []
    for j in range(micro_batch):
      w = b * micro_batch + j
      s = w * ctx
      x = ids[s : s + ctx].astype(np.int64)
      m = mask[s : s + ctx]
      xs.append(x)
      ys.append(np.where(m.astype(bool), x, IGNORE_INDEX))
    yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def evaluate_pretrain(model, path, ctx, *, micro_batch, n_batches, device, dtype, seed=99991) -> dict:
  return evaluate_batches(
    model,
    memmap_eval_batches(path, ctx, micro_batch=micro_batch, n_batches=n_batches, seed=seed),
    device=device,
    dtype=dtype,
  )


def evaluate_sft(model, ids_path, mask_path, ctx, *, micro_batch, device, dtype, max_batches=None) -> dict:
  return evaluate_batches(
    model,
    masked_eval_batches(ids_path, mask_path, ctx, micro_batch=micro_batch, max_batches=max_batches),
    device=device,
    dtype=dtype,
  )


def corpus_windows(path: str | Path, ctx: int) -> int:
  return (os.path.getsize(str(path)) // 2) // ctx
