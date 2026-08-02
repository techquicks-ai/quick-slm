"""The learning-rate schedule, as a pure function of the step index.

Kept free of torch so its shape can be asserted in a test rather than inspected
on a plot after nine thousand steps have already run.
"""

from __future__ import annotations

import math
from typing import Callable

from .config import OptimConfig


def cosine_with_warmup(
  step: int,
  *,
  lr_peak: float,
  lr_min: float,
  warmup_steps: int,
  total_steps: int,
) -> float:
  """Linear warmup to `lr_peak`, then cosine decay to `lr_min` at `total_steps`.

  `step` is zero-based. Warmup returns `lr_peak * (step + 1) / warmup_steps`,
  so step 0 already carries a non-zero rate; a schedule that starts at exactly
  zero throws away its first optimizer step. This is the formula, preserved
  verbatim, because the base checkpoint was produced under it.

  Past `total_steps` the rate is pinned at `lr_min` rather than continuing the
  cosine into negative territory, which is what an unclamped formula would do
  if a run were extended.
  """
  if warmup_steps > 0 and step < warmup_steps:
    return lr_peak * (step + 1) / warmup_steps
  if step >= total_steps:
    return lr_min
  progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
  return lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_lr_fn(optim: OptimConfig, total_steps: int) -> Callable[[int], float]:
  """Bind an `OptimConfig` and a run length into a one-argument schedule."""

  def lr_at(step: int) -> float:
    return cosine_with_warmup(
      step,
      lr_peak=optim.lr_peak,
      lr_min=optim.lr_min,
      warmup_steps=optim.warmup_steps,
      total_steps=total_steps,
    )

  return lr_at
