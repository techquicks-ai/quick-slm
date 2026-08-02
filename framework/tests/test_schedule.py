from __future__ import annotations

import math

import pytest

from quick_slm_trainer.config import OptimConfig
from quick_slm_trainer.schedule import cosine_with_warmup, make_lr_fn

SCHED = dict(lr_peak=3e-4, lr_min=3e-5, warmup_steps=1_000, total_steps=9_500)


def test_first_step_is_not_zero():
  # A schedule starting at exactly zero throws away its first optimizer step.
  assert cosine_with_warmup(0, **SCHED) == pytest.approx(3e-4 / 1_000)


def test_warmup_ends_exactly_at_peak():
  assert cosine_with_warmup(999, **SCHED) == pytest.approx(3e-4)


def test_cosine_starts_at_peak_and_ends_at_min():
  assert cosine_with_warmup(1_000, **SCHED) == pytest.approx(3e-4)
  assert cosine_with_warmup(9_500, **SCHED) == pytest.approx(3e-5)


def test_midpoint_of_the_cosine_is_the_midpoint_of_the_range():
  mid = (SCHED["warmup_steps"] + SCHED["total_steps"]) // 2
  expected = SCHED["lr_min"] + (SCHED["lr_peak"] - SCHED["lr_min"]) * 0.5
  assert cosine_with_warmup(mid, **SCHED) == pytest.approx(expected, rel=1e-3)


def test_decay_is_monotone():
  rates = [cosine_with_warmup(s, **SCHED) for s in range(1_000, 9_500, 97)]
  assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_past_the_end_is_clamped_not_extrapolated():
  # An unclamped cosine keeps going and turns the rate negative.
  assert cosine_with_warmup(20_000, **SCHED) == pytest.approx(3e-5)


def test_zero_warmup_starts_on_the_cosine():
  assert cosine_with_warmup(0, lr_peak=1e-3, lr_min=0.0, warmup_steps=0, total_steps=100) == pytest.approx(1e-3)


def test_warmup_longer_than_the_run_never_divides_by_zero():
  lr = cosine_with_warmup(50, lr_peak=1e-3, lr_min=1e-4, warmup_steps=100, total_steps=10)
  assert math.isfinite(lr)


def test_make_lr_fn_binds_the_config():
  fn = make_lr_fn(OptimConfig(), 9_500)
  assert fn(1_000) == pytest.approx(OptimConfig().lr_peak)
