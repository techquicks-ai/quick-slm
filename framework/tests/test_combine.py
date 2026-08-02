"""The combine allocation.

Pure arithmetic, and the place a missing shard turns into a corpus whose ratios
are not the ratios the paper claims.
"""

from __future__ import annotations

import pytest

from quick_slm_trainer.pretraining.combine import CombineRefused, plan_allocation

CTX = 4096
BUDGETS = {"a": 100 * CTX, "b": 100 * CTX}


def test_strict_refuses_when_a_source_is_short():
  with pytest.raises(CombineRefused, match="short by"):
    plan_allocation(BUDGETS, {"a": 100 * CTX, "b": 50 * CTX}, ctx=CTX, mode="strict")


def test_strict_names_every_short_source_and_the_way_out():
  with pytest.raises(CombineRefused) as e:
    plan_allocation(BUDGETS, {"a": 0, "b": 0}, ctx=CTX, mode="strict")
  assert "a short by" in str(e.value) and "b short by" in str(e.value)
  assert "cap_proportional" in str(e.value)


def test_strict_allocates_the_full_budget_when_nothing_is_short():
  take = plan_allocation(BUDGETS, {"a": 200 * CTX, "b": 100 * CTX}, ctx=CTX, mode="strict")
  assert take == BUDGETS


def test_cap_proportional_preserves_the_ratio():
  budgets = {"a": 775 * CTX, "b": 100 * CTX}
  actuals = {"a": 775 * CTX, "b": 50 * CTX} # b is at half its budget
  take = plan_allocation(budgets, actuals, ctx=CTX, mode="cap_proportional")
  assert take["a"] / take["b"] == pytest.approx(budgets["a"] / budgets["b"], rel=1e-2)


def test_cap_proportional_never_scales_above_one():
  take = plan_allocation(BUDGETS, {"a": 500 * CTX, "b": 500 * CTX}, ctx=CTX, mode="cap_proportional")
  assert take == BUDGETS


def test_use_all_takes_what_exists_and_skews_the_ratio():
  take = plan_allocation(BUDGETS, {"a": 100 * CTX, "b": 10 * CTX}, ctx=CTX, mode="use_all")
  assert take == {"a": 100 * CTX, "b": 10 * CTX}


def test_use_all_is_still_capped_at_budget():
  take = plan_allocation(BUDGETS, {"a": 999 * CTX, "b": 100 * CTX}, ctx=CTX, mode="use_all")
  assert take["a"] == 100 * CTX


@pytest.mark.parametrize("mode", ["cap_proportional", "use_all"])
def test_every_allocation_is_a_whole_number_of_windows(mode):
  # A partial window at the tail of a source would be zero-padded by the
  # memmap read, silently injecting a block of token id 0 into the corpus.
  actuals = {"a": 100 * CTX + 17, "b": 33 * CTX + 4095}
  take = plan_allocation(BUDGETS, actuals, ctx=CTX, mode=mode)
  assert all(v % CTX == 0 for v in take.values())


def test_allocation_never_exceeds_what_is_on_disk():
  actuals = {"a": 60 * CTX, "b": 100 * CTX}
  for mode in ("cap_proportional", "use_all"):
    take = plan_allocation(BUDGETS, actuals, ctx=CTX, mode=mode)
    assert all(take[k] <= actuals[k] for k in BUDGETS), mode


def test_a_missing_source_is_not_silently_zero_weighted():
  with pytest.raises(CombineRefused):
    plan_allocation(BUDGETS, {"a": 100 * CTX}, ctx=CTX, mode="strict")


def test_unknown_mode_raises():
  with pytest.raises(ValueError, match="unknown combine_mode"):
    plan_allocation(BUDGETS, BUDGETS, ctx=CTX, mode="whatever")
