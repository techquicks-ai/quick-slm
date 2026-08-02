"""Behavioural grading of a fine-tuned checkpoint.

`evaluate_sft` reports loss. That is the right number for the training loop and
the wrong one for the question this corpus was built to answer. Section 9 of the
paper measures the realised corpus at 57.5 percent single-stage, and a model
can hold a low loss on a corpus shaped like that by emitting the most frequent
call for everything. Loss will not distinguish that model from a good one. This
module does.

Three properties are graded, separately, because a 103M model reaches them in
that order and collapsing them hides which one it has reached:

  well-formed  the envelope parses and every call is structurally legal
  correct    the first call matches the reference for this example
  grounded   *both* branches of a counterfactual pair are correct

The third is the one the corpus exists to measure, and it is sharper than it
looks. `validate_pair` rejects any pair whose two branches share an oracle call
(`SWAP_ORACLES_AGREE`), so every surviving pair has two branches whose correct
answers *differ*. A model that emits one call for both branches therefore scores
exactly one, whatever call it picks and however sensible that call is. "Exactly
one correct" is not a near miss and not chance; it is the signature of a model
that never read `<state>`. Reporting per-branch accuracy alone would show such a
model at 50 percent and read as partial credit, which is why `grade_pairs`
reports the joint outcome instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..template import Example, group_of, parse_calls, parse_think
from .oracles import calls_agree
from .tools import Tool, by_name
from .validate import MIN_THINK_CHARS, check_call

# --- Format verdicts. `None` means well-formed. ---------------------------
NO_THINK = "no_think"
SHORT_THINK = "short_think"
NO_RESPONSE = "no_response"
CALLS_NOT_LIST = "calls_not_list"
EMPTY_CALLS = "empty_calls"

#: Outcomes for a counterfactual pair. See the module docstring.
BOTH = "both"
ONE = "one"
NEITHER = "neither"


def grade_format(text: str, tools: Sequence[Tool]) -> str | None:
  """`None` when the generation is well-formed, else the first fault found.

  The faults are ordered outside-in: the envelope, then the call list, then
  each call. A model that has learnt the envelope but not the schema should be
  distinguishable from one that has learnt neither, so the reasons stay
  distinct rather than collapsing into one "malformed" bucket.
  """
  think = parse_think(text)
  if think is None:
    return NO_THINK
  if len(think.strip()) < MIN_THINK_CHARS:
    return SHORT_THINK

  calls = parse_calls(text)
  if calls is None:
    return NO_RESPONSE
  if not isinstance(calls, list):
    return CALLS_NOT_LIST
  if not calls:
    return EMPTY_CALLS

  allowed = by_name(tools)
  for call in calls:
    reason = check_call(call, allowed)
    if reason is not None:
      return reason
  return None


def reference_calls(ex: Example) -> list[dict]:
  """The calls the first assistant turn should make.

  Only the first turn is graded. Everything after it is conditioned on tool
  results the model did not receive, so scoring it would measure the harness
  rather than the model.
  """
  for turn in ex.turns:
    calls = getattr(turn, "calls", None)
    if calls is not None:
      return list(calls)
  return []


def grade_call(predicted: Sequence[dict] | None, reference: Sequence[dict]) -> bool:
  """Does the first predicted call match the first reference call?

  Argument *values* are compared, not just the tool name. `get_weather` with a
  guessed city is the trap category's whole failure mode, and a name-only
  comparison would score it correct.

  The exception is `oracles.FREE_TEXT_ARGS`, which excludes `answer.text`. A
  refusal is therefore graded on whether the model reached for `answer` at all,
  not on how it worded the decline, because no string comparison distinguishes
  a good refusal from a bad one. Read the refusal and trap rows accordingly:
  they measure call *choice*, and the wording needs a judge or an eye.
  """
  if not predicted or not reference:
    return False
  return calls_agree(predicted[0], reference[0])


@dataclass
class Result:
  """One graded generation."""

  example: Example
  text: str
  fault: str | None = None
  correct: bool = False

  @property
  def category(self) -> str:
    return self.example.category

  @property
  def well_formed(self) -> bool:
    return self.fault is None


def grade(ex: Example, text: str) -> Result:
  """Grade one generation against the example it was produced from."""
  fault = grade_format(text, ex.tools)
  correct = False
  if fault is None:
    correct = grade_call(parse_calls(text), reference_calls(ex))
  return Result(example=ex, text=text, fault=fault, correct=correct)


def grade_pairs(results: Iterable[Result]) -> dict:
  """Joint outcome over counterfactual pairs.

  Only groups with both branches present are counted. A pair split across the
  train/validation boundary would be graded against a twin the model trained
  on, and `split_examples` is group-aware precisely so that cannot happen; a
  lone branch here means the caller assembled the set by some other route, and
  it is dropped rather than counted as half a pair.
  """
  groups: dict[str, list[Result]] = {}
  for r in results:
    key = group_of(r.example)
    if key is not None:
      groups.setdefault(key, []).append(r)

  outcomes: Counter[str] = Counter()
  complete = 0
  for branches in groups.values():
    if len(branches) != 2:
      continue
    complete += 1
    n = sum(1 for b in branches if b.correct)
    outcomes[BOTH if n == 2 else ONE if n == 1 else NEITHER] += 1

  return {
    "pairs": complete,
    "dropped_incomplete": len(groups) - complete,
    BOTH: outcomes[BOTH],
    ONE: outcomes[ONE],
    NEITHER: outcomes[NEITHER],
    "grounded_rate": (outcomes[BOTH] / complete) if complete else 0.0,
  }


@dataclass
class Report:
  """Everything worth printing about one checkpoint."""

  overall: dict = field(default_factory=dict)
  by_category: dict = field(default_factory=dict)
  pairs: dict = field(default_factory=dict)
  faults: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return {
      "overall": self.overall,
      "by_category": self.by_category,
      "pairs": self.pairs,
      "faults": self.faults,
    }

  def table(self) -> str:
    rows = [f"{'category':<24}{'n':>7}{'well-formed':>14}{'correct':>10}"]
    rows.append("-" * len(rows[0]))
    for key, s in sorted(self.by_category.items()):
      rows.append(
        f"{key:<24}{s['n']:>7,}{s['well_formed_rate']:>13.1%}{s['correct_rate']:>10.1%}"
      )
    o = self.overall
    rows.append("-" * len(rows[0]))
    rows.append(
      f"{'ALL':<24}{o['n']:>7,}{o['well_formed_rate']:>13.1%}{o['correct_rate']:>10.1%}"
    )
    p = self.pairs
    if p.get("pairs"):
      rows += [
        "",
        f"counterfactual pairs: {p['pairs']:,}",
        f" both branches correct (grounded) {p[BOTH]:>6,} {p['grounded_rate']:.1%}",
        f" exactly one correct (state-blind) {p[ONE]:>6,}",
        f" neither correct          {p[NEITHER]:>6,}",
      ]
    if self.faults:
      rows.append("")
      rows.append("format faults:")
      for reason, n in sorted(self.faults.items(), key=lambda kv: -kv[1]):
        rows.append(f" {reason:<28}{n:>7,}")
    return "\n".join(rows)


def summarise(results: Sequence[Result]) -> Report:
  """Aggregate graded generations into a printable report."""
  by_category: dict[str, dict] = {}
  for key in sorted({r.category for r in results}):
    rs = [r for r in results if r.category == key]
    by_category[key] = _stats(rs)

  faults = Counter(r.fault for r in results if r.fault is not None)
  return Report(
    overall=_stats(results),
    by_category=by_category,
    pairs=grade_pairs(results),
    faults=dict(faults),
  )


def _stats(rs: Sequence[Result]) -> dict:
  n = len(rs)
  wf = sum(1 for r in rs if r.well_formed)
  ok = sum(1 for r in rs if r.correct)
  return {
    "n": n,
    "well_formed": wf,
    "correct": ok,
    "well_formed_rate": (wf / n) if n else 0.0,
    "correct_rate": (ok / n) if n else 0.0,
  }
