"""Diagnostics that explain a fine-tuned checkpoint's failures.

`grade.py` says how much the model gets right. This module exists to say *why*
it gets the rest wrong, and specifically to separate the two explanations that
lead to opposite decisions for the next version:

  composition  the model is fine but the corpus taught it the wrong priors.
         Fixed by rebalancing data. Cheap.
  capacity   the model cannot represent the behaviour at this size.
         Fixed by a bigger model. Expensive.

The two are not distinguishable from validation accuracy alone, and they predict
the same held-out numbers. `memorisation_gap` is what separates them, and it is
the reason this module grades a sample of *training* examples as well: a model
that fails on data it was fit to has not been out-competed by the corpus mix, it
has run out of capacity. A model that nails training and fails validation has
memorised. A model that fails both only where a cell is thin has a composition
problem, and `cell_report` localises that to the cell.

Everything here takes graded `Result` objects and plain corpora, returns plain
dicts, and touches no model. That is deliberate: the whole report can be
exercised in tests against synthetic results with known answers, which is the
only way to trust a diagnostic that will be read as evidence.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from ..template import Example, group_of, parse_calls, parse_think
from .dedup import example_fingerprint
from .grade import BOTH, NEITHER, ONE, Result, reference_calls

#: Cells thinner than this are reported but flagged: an accuracy over a handful
#: of examples is noise, and reading it as signal is how a diagnostic misleads.
MIN_CELL_N = 20


def cell_of(ex: Example) -> tuple[str, str]:
  """The (category, subtype) cell an example belongs to."""
  return (ex.category, str(ex.meta.get("subtype", "")))


# --------------------------------------------------------------------------
# Composition versus capacity
# --------------------------------------------------------------------------
def cell_report(
  results: Sequence[Result],
  train_examples: Sequence[Example],
  train_results: Sequence[Result] = (),
) -> list[dict]:
  """Per-cell training volume against held-out accuracy, and train accuracy.

  The three columns together are the diagnosis. Read them as:

    train high, val high   the cell works
    train high, val low   memorised; the cell needs more variety
    train low, val low   never learnt. Look at `n_train`: if the cell is
                 thin it is composition, if it is fat it is
                 capacity or a broken specification
  """
  train_n = Counter(cell_of(ex) for ex in train_examples)
  by_cell: dict[tuple[str, str], list[Result]] = defaultdict(list)
  for r in results:
    by_cell[cell_of(r.example)].append(r)
  train_by_cell: dict[tuple[str, str], list[Result]] = defaultdict(list)
  for r in train_results:
    train_by_cell[cell_of(r.example)].append(r)

  rows = []
  total_train = sum(train_n.values()) or 1
  for cell in sorted(set(by_cell) | set(train_n)):
    rs = by_cell.get(cell, [])
    trs = train_by_cell.get(cell, [])
    n = len(rs)
    row = {
      "category": cell[0],
      "subtype": cell[1],
      "n_train": train_n.get(cell, 0),
      "train_share": train_n.get(cell, 0) / total_train,
      "n_val": n,
      "well_formed_rate": (sum(r.well_formed for r in rs) / n) if n else None,
      "correct_rate": (sum(r.correct for r in rs) / n) if n else None,
      "n_train_graded": len(trs),
      "train_correct_rate": (sum(r.correct for r in trs) / len(trs)) if trs else None,
      "thin": n < MIN_CELL_N,
    }
    rows.append(row)
  return rows


def memorisation_gap(cells: Sequence[dict]) -> dict:
  """Train accuracy minus validation accuracy, over cells that have both.

  A large positive gap means the model fit the corpus and did not generalise
  from it. A gap near zero with low accuracy on both sides means it never fit
  the corpus in the first place, which no amount of rebalancing repairs.
  """
  paired = [
    c for c in cells
    if c["train_correct_rate"] is not None and c["correct_rate"] is not None
    and c["n_train_graded"] >= MIN_CELL_N and c["n_val"] >= MIN_CELL_N
  ]
  if not paired:
    return {"cells": 0, "verdict": "not measured: grade training examples too"}

  gaps = [c["train_correct_rate"] - c["correct_rate"] for c in paired]
  train_mean = statistics.fmean(c["train_correct_rate"] for c in paired)
  val_mean = statistics.fmean(c["correct_rate"] for c in paired)
  median = statistics.median(gaps)

  # Cells the model gets wrong every single time. A cell at exactly zero is
  # qualitatively different from a weak one: the model is not attempting the
  # behaviour, it is emitting something else, and averaging hides that.
  dead = [
    {"category": c["category"], "subtype": c["subtype"], "n_train": c["n_train"]}
    for c in paired if c["correct_rate"] == 0.0
  ]

  if train_mean < 0.5:
    verdict = "capacity: the model does not fit even its training data"
  elif median > 0.25:
    verdict = "memorisation: fits training, does not transfer"
  elif val_mean < 0.6 or dead:
    # The gap is small but the level is not. A model with a strong prior
    # looks exactly like this: consistent across train and held-out, and
    # consistently wrong wherever the prior does not apply. Calling it
    # "generalising" because the gap is small would be true and useless.
    verdict = (
      "uniform: train and held-out agree, but at low accuracy. Not a "
      "generalisation gap -- read the per-cell table and default_call, "
      "this is usually a prior rather than a shortfall"
    )
  else:
    verdict = "generalising: train and held-out track each other"

  return {
    "cells": len(paired),
    "train_mean": train_mean,
    "val_mean": val_mean,
    "median_gap": median,
    "max_gap": max(gaps),
    "dead_cells": dead,
    "verdict": verdict,
    "worst": sorted(
      ({"category": c["category"], "subtype": c["subtype"],
       "gap": c["train_correct_rate"] - c["correct_rate"]} for c in paired),
      key=lambda d: -d["gap"],
    )[:5],
  }


def volume_accuracy_correlation(cells: Sequence[dict]) -> dict:
  """Does per-cell accuracy track per-cell training volume?

  A strong positive rank correlation is the signature of a composition
  problem: the cells the corpus starved are the cells the model fails. Near
  zero means volume is not what is limiting, and rebalancing the corpus would
  not have helped, which is an argument for spending 's budget elsewhere.

  Spearman is computed directly rather than pulled from scipy, which is not a
  dependency of this package and would not be worth adding for one statistic.
  """
  usable = [c for c in cells if c["correct_rate"] is not None and not c["thin"]]
  if len(usable) < 4:
    return {"cells": len(usable), "rho": None,
        "note": "too few non-thin cells to correlate"}

  def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order): # average ranks over ties
      j = i
      while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
        j += 1
      avg = (i + j) / 2 + 1
      for k in range(i, j + 1):
        r[order[k]] = avg
      i = j + 1
    return r

  rx = ranks([c["n_train"] for c in usable])
  ry = ranks([c["correct_rate"] for c in usable])
  n = len(usable)
  mx, my = statistics.fmean(rx), statistics.fmean(ry)
  num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
  den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
  rho = (num / den) if den else 0.0

  if rho > 0.5:
    reading = "composition: thin cells are the failing cells"
  elif rho < -0.5:
    reading = "inverted: the fattest cells fail, look for a broken cell"
  else:
    reading = "volume is not the limiting factor"
  return {"cells": n, "rho": rho, "reading": reading}


# --------------------------------------------------------------------------
# What the model does instead
# --------------------------------------------------------------------------
def default_call(results: Iterable[Result]) -> dict:
  """The call the model reaches for when it is wrong.

  One call dominating means the model has a prior rather than a policy, and
  the share of the corpus that call occupies is the first place to look.
  """
  wrong = [r for r in results if r.well_formed and not r.correct]
  picked: Counter[str] = Counter()
  for r in wrong:
    calls = parse_calls(r.text) or []
    picked[str(calls[0].get("name")) if calls else "(none)"] += 1
  top = picked.most_common(1)
  return {
    "n_wrong": len(wrong),
    "calls": dict(picked.most_common(12)),
    "top_call": top[0][0] if top else None,
    "top_share": (top[0][1] / len(wrong)) if wrong and top else 0.0,
  }


def confusion(results: Iterable[Result], limit: int = 25) -> list[dict]:
  """Reference tool name against emitted tool name, most frequent first.

  Restricted to well-formed generations: a malformed one has no emitted name
  and would otherwise all collapse into a single meaningless row.
  """
  pairs: Counter[tuple[str, str]] = Counter()
  for r in results:
    if not r.well_formed:
      continue
    ref = reference_calls(r.example)
    got = parse_calls(r.text) or []
    pairs[(
      str(ref[0].get("name")) if ref else "(none)",
      str(got[0].get("name")) if got else "(none)",
    )] += 1
  return [
    {"expected": a, "got": b, "n": n, "correct_name": a == b}
    for (a, b), n in pairs.most_common(limit)
  ]


def think_call_coupling(results: Sequence[Result]) -> dict:
  """Does the reasoning block agree with the call that follows it?

  A model whose `<think>` names one tool and whose `<response>` calls another
  has learnt to produce both fields without conditioning the second on the
  first. That is a different defect from not knowing the answer, it is not
  visible in accuracy, and it changes what should do: coupling is a
  format and masking question, not a data-volume one.

  Only generations that name exactly one offered tool in the reasoning are
  counted, because "mentions two tools" has no unambiguous reading.
  """
  considered = decoupled = agreed = 0
  knew_but_missed = 0
  examples: list[dict] = []

  for r in results:
    if not r.well_formed:
      continue
    think = parse_think(r.text) or ""
    got = parse_calls(r.text) or []
    if not got:
      continue
    called = str(got[0].get("name"))

    names = {str(t.get("name")) for t in r.example.tools}
    mentioned = {n for n in names if re.search(rf"\b{re.escape(n)}\b", think)}
    if len(mentioned) == 1:
      considered += 1
      only = next(iter(mentioned))
      if only == called:
        agreed += 1
      else:
        decoupled += 1
        if len(examples) < 20:
          examples.append({"category": r.category, "think_named": only,
                   "called": called, "think": think[:220]})

    # Separately: the reasoning contains the reference argument value but the
    # call does not use it. The model had the answer and did not emit it.
    ref = reference_calls(r.example)
    if ref and not r.correct:
      want = [v for v in (ref[0].get("arguments") or {}).values()
          if isinstance(v, str) and len(v) > 2]
      used = {str(v) for v in (got[0].get("arguments") or {}).values()}
      if want and any(w.lower() in think.lower() for w in want) and not (set(want) & used):
        knew_but_missed += 1

  return {
    "considered": considered,
    "agreed": agreed,
    "decoupled": decoupled,
    "decoupled_rate": (decoupled / considered) if considered else None,
    "think_had_answer_call_did_not": knew_but_missed,
    "examples": examples,
  }


# --------------------------------------------------------------------------
# The counterfactual category, per specification
# --------------------------------------------------------------------------
def pair_diagnostics(results: Sequence[Result]) -> dict:
  """Grounded rate broken out by specification, plus branch bias.

  Per-specification, because Table 11 shows the specifications are wildly
  unequal in capacity and there is no reason to expect them equal in
  difficulty either. A specification at zero is a concrete thing to fix.

  Branch bias is the control on the pair metric itself. If the model favours
  one branch label systematically, the pairs are not symmetric and the "exactly
  one correct" reading is measuring the construction rather than the model.
  """
  groups: dict[str, list[Result]] = defaultdict(list)
  for r in results:
    key = group_of(r.example)
    if key is not None:
      groups[key].append(r)

  by_spec: dict[str, Counter] = defaultdict(Counter)
  branch_wins: Counter[str] = Counter()
  for branches in groups.values():
    if len(branches) != 2:
      continue
    spec = str(branches[0].example.meta.get("spec_id", "?"))
    n = sum(1 for b in branches if b.correct)
    by_spec[spec][BOTH if n == 2 else ONE if n == 1 else NEITHER] += 1
    if n == 1:
      winner = next(b for b in branches if b.correct)
      branch_wins[str(winner.example.meta.get("branch", "?"))] += 1

  rows = []
  for spec, c in sorted(by_spec.items()):
    total = sum(c.values())
    rows.append({
      "spec_id": spec, "pairs": total,
      BOTH: c[BOTH], ONE: c[ONE], NEITHER: c[NEITHER],
      "grounded_rate": c[BOTH] / total if total else 0.0,
    })
  rows.sort(key=lambda d: (d["grounded_rate"], -d["pairs"]))

  total_one = sum(branch_wins.values())
  return {
    "by_spec": rows,
    "branch_bias": {
      "wins": dict(branch_wins),
      # 0.5 is unbiased. Far from it means the pair construction, not the
      # model, is producing the "exactly one" outcomes.
      "skew": (max(branch_wins.values()) / total_one) if total_one else None,
    },
  }


# --------------------------------------------------------------------------
# Why the corpus lost what it lost
# --------------------------------------------------------------------------
def dedup_forensics(validated: Sequence[Example], kept: Sequence[Example]) -> list[dict]:
  """Per cell: how repetitive the teacher was, and what dedup therefore took.

  Dedup is near-duplicate over MinHash, but exact fingerprint collisions are
  the cheap and legible part of the same story: `largest_cluster` is the number
  of records the teacher produced that were identical under the fingerprint,
  and a large value means the cell had nothing left to say long before the
  planner stopped asking.
  """
  kept_n = Counter(cell_of(ex) for ex in kept)
  by_cell: dict[tuple[str, str], list[Example]] = defaultdict(list)
  for ex in validated:
    by_cell[cell_of(ex)].append(ex)

  rows = []
  for cell, exs in sorted(by_cell.items()):
    fps = Counter(example_fingerprint(e) for e in exs)
    n, k = len(exs), kept_n.get(cell, 0)
    rows.append({
      "category": cell[0],
      "subtype": cell[1],
      "n_validated": n,
      "n_kept": k,
      "kept_rate": k / n if n else 0.0,
      "distinct_fingerprints": len(fps),
      "exact_duplicate_rate": 1 - len(fps) / n if n else 0.0,
      "largest_cluster": max(fps.values()) if fps else 0,
      "records_per_kept": n / k if k else None,
    })
  rows.sort(key=lambda d: d["kept_rate"])
  return rows


def length_buckets(results: Sequence[Result], edges: Sequence[int] = (256, 512, 1024, 2048)) -> list[dict]:
  """Accuracy against prompt length. Degradation with length is its own defect.

  Length is measured in characters of the rendered prompt rather than tokens,
  so this needs no tokenizer and stays runnable offline.
  """
  from ..template import render_prompt

  buckets: dict[str, list[Result]] = defaultdict(list)
  for r in results:
    n = len(render_prompt(r.example))
    label = f"<{edges[0]}"
    for lo, hi in zip(edges, list(edges[1:]) + [None]):
      if n >= lo:
        label = f">={lo}" if hi is None else f"{lo}-{hi}"
    buckets[label].append(r)

  order = [f"<{edges[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(edges, edges[1:])] + [f">={edges[-1]}"]
  return [
    {"bucket": b, "n": len(rs),
     "correct_rate": sum(r.correct for r in rs) / len(rs),
     "well_formed_rate": sum(r.well_formed for r in rs) / len(rs)}
    for b in order if (rs := buckets.get(b))
  ]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def failure_samples(results: Sequence[Result], per_category: int = 12) -> list[dict]:
  """A bounded, balanced sample of failures, with enough text to read."""
  out: list[dict] = []
  by_cat: dict[str, list[Result]] = defaultdict(list)
  for r in results:
    if not r.correct:
      by_cat[r.category].append(r)
  for cat in sorted(by_cat):
    for r in by_cat[cat][:per_category]:
      out.append({
        "category": cat,
        "subtype": r.example.meta.get("subtype", ""),
        "spec_id": r.example.meta.get("spec_id"),
        "user": r.example.turns[0].text[:300],
        "state": r.example.state,
        "expected": reference_calls(r.example),
        "got": parse_calls(r.text),
        "think": (parse_think(r.text) or "")[:300],
        "fault": r.fault,
      })
  return out


def build_report(
  *,
  results: Sequence[Result],
  train_examples: Sequence[Example],
  validated: Sequence[Example] = (),
  kept: Sequence[Example] = (),
  train_results: Sequence[Result] = (),
  meta: dict | None = None,
) -> dict:
  """The whole diagnostic, as one JSON-serialisable dict."""
  cells = cell_report(results, train_examples, train_results)
  report: dict[str, Any] = {
    "meta": meta or {},
    "n_graded": len(results),
    "n_train_graded": len(train_results),
    "cells": cells,
    "memorisation": memorisation_gap(cells),
    "volume_vs_accuracy": volume_accuracy_correlation(cells),
    "default_call": default_call(results),
    "confusion": confusion(results),
    "think_call_coupling": think_call_coupling(results),
    "pairs": pair_diagnostics(results),
    "length": length_buckets(results),
    "failures": failure_samples(results),
  }
  if validated and kept:
    report["dedup"] = dedup_forensics(validated, kept)
  return report
