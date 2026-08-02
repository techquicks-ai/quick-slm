"""The counterfactual state swap: category 3, generated and accepted in pairs.

The filter this replaces accepted a `<think>` block that contained the words
"state" and "memory". That rewards the surface form of reasoning. A teacher that
narrates a conflict and then copies the call off the user turn passes it; a
teacher that silently reads state and emits the right call fails it. A corpus
selected that way cannot support the claim that the model grounds its calls in
live state, because nothing in the selection ever tested that.

Here every category-3 sample is one branch of a pair. The two branches share a
system turn, a tool list, a `<memory>` block, and a user turn, byte for byte, and
differ at exactly one field of `<state>`, the `decisive_path`. An oracle, written
once per tool family and never shown to the teacher, computes the correct call
from state alone. A pair is kept only when both branches match their oracle, and
both branches are then trained on.

That is the whole mechanism. The same user turn appears twice with two different
correct answers, so the user turn carries no information about which answer is
right, and `<state>` is the only feature that predicts it. A model that
pattern-matches the request gets exactly half of them wrong, whichever way it
guesses.

If the teacher gets one branch right and the other wrong, both are dropped. The
correct branch on its own is indistinguishable from a lucky pattern match, which
is the sample the pair was built to exclude.

Leakage
-------
The paper's evaluation is this same construction, so `TRAIN_SPECS` and
`EVAL_SPECS` share no tool family, no decisive path, and no state schema.
`assert_no_leakage` enforces that, and `tests/test_sft_conflict.py` calls it.
Partitioning at the sample level would leak, because the two branches of a pair
share everything but one field.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..template import AssistantTurn, Example, UserTurn, dumps_compact, format_memory, render_prompt
from . import pointer
from .oracles import ORACLES, Call, Oracle, Request, calls_agree, canonical_call
from .tools import ALL_TOOLS, ANSWER, DOMAIN_SYSTEM, Tool
from .validate import (
  NO_CALLS,
  NOT_JSON,
  SWAP_CALL_MISMATCH,
  SWAP_MEMORY_DIVERGED,
  SWAP_MISSING_BRANCH,
  SWAP_MULTIPLE_CALLS,
  SWAP_ORACLES_AGREE,
  SWAP_PARTNER_FAILED,
  SWAP_REQUEST_DIVERGED,
  SWAP_STATE_DIVERGED,
  check_call,
  check_think,
  extract_json,
)


class SpecRejected(ValueError):
  """The spec cannot produce a counterfactual pair. Not a sample-level reject."""


# ==========================================================================
# Spec model
# ==========================================================================
@dataclass(frozen=True)
class Scenario:
  """One randomised instance of a spec, before the decisive field is set.

  `request` is the user turn plus whatever the scenario already knew when it
  wrote that turn. Both branches receive the identical dict, so it can carry no
  signal about which branch is which.
  """

  state: dict
  memory: dict # {"recent": [...], "last_results": [...]}
  request: Request


@dataclass(frozen=True)
class Branch:
  label: str # "a" or "b"
  state: dict
  call: Call # what the oracle says; never shown to the teacher


@dataclass(frozen=True)
class Pair:
  pair_id: str
  spec_id: str
  family: str
  domain: str
  subtype: str
  tools: list[Tool]
  memory: dict
  request: Request
  a: Branch
  b: Branch

  def branch(self, label: str) -> Branch:
    if label not in ("a", "b"):
      raise KeyError(f"branch must be 'a' or 'b', got {label!r}")
    return self.a if label == "a" else self.b

  def branches(self) -> tuple[Branch, Branch]:
    return (self.a, self.b)


@dataclass(frozen=True)
class SwapSpec:
  id: str
  family: str
  domain: str
  subtype: str
  #: Domain tools offered, before `answer` is appended. `answer` is never listed
  #: here: it is universal and appended by `tools()`.
  tool_names: tuple[str, ...]
  decisive_path: str
  variants: tuple[Any, Any]
  oracle_key: str
  scenario: Callable[[random.Random], Scenario] = field(repr=False)
  #: The branch whose `<state>` contradicts the shared `<memory>`. Documentation
  #: and a test hook; the pipeline does not branch on it.
  stale_under: str = "b"

  @property
  def oracle(self) -> Oracle:
    return ORACLES[self.oracle_key]

  def tools(self) -> list[Tool]:
    return [ALL_TOOLS[n] for n in self.tool_names] + [ANSWER]

  def system(self) -> str:
    return DOMAIN_SYSTEM[self.domain]


# ==========================================================================
# Pair construction
# ==========================================================================
def build_pair(spec: SwapSpec, rng: random.Random, *, pair_id: str = "") -> Pair:
  """Instantiate `spec` into a validated counterfactual pair.

  Raises `SpecRejected` when the construction cannot hold, which is a fault in
  the spec rather than in any sample.
  """
  scenario = spec.scenario(rng)

  if not pointer.exists(scenario.state, spec.decisive_path):
    raise SpecRejected(f"{spec.id}: decisive_path {spec.decisive_path!r} does not resolve")
  if spec.variants[0] == spec.variants[1]:
    raise SpecRejected(f"{spec.id}: the two variants are the same value")

  state_a = pointer.with_value(scenario.state, spec.decisive_path, spec.variants[0])
  state_b = pointer.with_value(scenario.state, spec.decisive_path, spec.variants[1])

  # The load-bearing assertion. Any incidental difference is a spurious cue.
  differing = pointer.diff_paths(state_a, state_b)
  if differing != {spec.decisive_path}:
    raise SpecRejected(
      f"{spec.id}: branches differ at {sorted(differing)}, expected only {spec.decisive_path!r}"
    )

  call_a = spec.oracle(state_a, scenario.request)
  call_b = spec.oracle(state_b, scenario.request)

  # Criterion 1. Under the canonical form, `answer` compares by name alone, so
  # a spec whose branches both fall through to prose is rejected here.
  if calls_agree(call_a, call_b):
    raise SpecRejected(
      f"{spec.id}: flipping {spec.decisive_path!r} does not change the correct call "
      f"({canonical_call(call_a)}); the pair cannot separate grounding from imitation"
    )

  allowed = {t["name"]: t for t in spec.tools()}
  for label, call in (("a", call_a), ("b", call_b)):
    reason = check_call(call, allowed)
    if reason is not None:
      raise SpecRejected(f"{spec.id}: oracle branch {label} emits an invalid call ({reason}): {call}")

  return Pair(
    pair_id=pair_id or f"{spec.id}-00000",
    spec_id=spec.id,
    family=spec.family,
    domain=spec.domain,
    subtype=spec.subtype,
    tools=spec.tools(),
    memory=scenario.memory,
    request=scenario.request,
    a=Branch("a", state_a, call_a),
    b=Branch("b", state_b, call_b),
  )


def check_spec(spec: SwapSpec, *, trials: int = 24) -> None:
  """Build the spec repeatedly. Raises `SpecRejected` on the first failure.

  A spec that holds for one random instance and not another is worse than one
  that never holds, because the failure surfaces as an unexplained reject rate
  part-way through generation.
  """
  for i in range(trials):
    build_pair(spec, random.Random(1000 + i))


def plan_pairs(specs: Sequence[SwapSpec], n_pairs: int, rng: random.Random) -> list[Pair]:
  """Round-robin over `specs`, so each contributes an equal share."""
  if not specs:
    raise ValueError("no specs to plan from")
  return [
    build_pair(specs[i % len(specs)], rng, pair_id=f"{specs[i % len(specs)].id}-{i // len(specs):05d}")
    for i in range(n_pairs)
  ]


def distinct_capacity(specs: Sequence[SwapSpec], *, trials: int = 2_000, seed: int = 0) -> dict[str, int]:
  """How many distinct pair-documents each spec can produce, ever.

  A spec randomises a building type, a resource, and a phrasing, and nothing
  else that survives `dedup.example_fingerprint`: the building ids are stripped
  and `answer`'s prose is excluded. `_building_run` therefore tops out at six
  types times six phrasings, or 36 pairs, however many instances are planned.

  Planning more pairs than this returns only duplicates, at full teacher price,
  and the dedup pass then deletes them. Nothing about that is visible until
  after the tokens are spent, so it is computed here instead: no teacher, no
  GPU, a second of CPU. Compare it against `plan_counts` before generating.

  `trials` bounds the sampling. It saturates well below the default for every
  spec in `conflict_specs`; a spec that has not saturated is a spec whose
  capacity is at least `trials`, which is already more than enough.
  """
  from .dedup import example_fingerprint

  capacity: dict[str, int] = {}
  for spec in specs:
    rng = random.Random(seed)
    seen: set[str] = set()
    for i in range(trials):
      pair = build_pair(spec, rng, pair_id=f"{spec.id}-{i:05d}")
      seen.add(
        "\n".join(
          example_fingerprint(example_from_branch(pair, label, "x", [pair.branch(label).call]))
          for label in ("a", "b")
        )
      )
    capacity[spec.id] = len(seen)
  return capacity


_TRAIN_CAPACITY_CACHE: dict[str, int] | None = None


def train_capacity() -> int:
  """Total distinct pairs `TRAIN_SPECS` can produce, memoised.

  Computed once at first call, from the same fingerprint the dedup pass uses, so
  the planner and the deduper agree about the ceiling. `distinct_capacity` builds
  thousands of pairs, so the result is cached: `plan_requests` calls this on every
  invocation and the test suite calls `plan_requests` a great many times.
  """
  global _TRAIN_CAPACITY_CACHE
  if _TRAIN_CAPACITY_CACHE is None:
    from .conflict_specs import TRAIN_SPECS

    _TRAIN_CAPACITY_CACHE = distinct_capacity(TRAIN_SPECS, trials=6_000)
  return sum(_TRAIN_CAPACITY_CACHE.values())


_RECOMMENDED_CACHE: dict[tuple[float, int], int] = {}


def conflict_yield_curve(planned: Sequence[int], *, seed: int = 20240201) -> list[tuple[int, int]]:
  """Distinct surviving pairs against pairs planned, for the round-robin planner.

  The ceiling `train_capacity` reports is reached only with unlimited draws. At a
  finite plan the yield is lower, and by more than a coupon-collector argument on
  one urn would suggest: `plan_pairs` round-robins over the specs, so the small
  ones saturate almost at once and keep drawing duplicates while the argument-only
  specs, whose ceilings are 288 to 384, are still filling. This walks the real
  curve by counting distinct fingerprints, the exact quantity dedup collapses to,
  with no minhash in the loop.
  """
  from .conflict_specs import TRAIN_SPECS
  from .dedup import example_fingerprint

  targets = sorted(planned)
  rng = random.Random(seed)
  seen: set[str] = set()
  out: list[tuple[int, int]] = []
  drawn = 0
  for target in targets:
    while drawn < target:
      spec = TRAIN_SPECS[drawn % len(TRAIN_SPECS)]
      pair = build_pair(spec, rng)
      seen.add(
        "\n".join(
          example_fingerprint(example_from_branch(pair, label, "x", [pair.branch(label).call]))
          for label in ("a", "b")
        )
      )
      drawn += 1
    out.append((target, len(seen)))
  return out


def recommended_conflict_pairs(*, target_fraction: float = 0.75, max_multiple: int = 8) -> int:
  """Pairs to plan to reach `target_fraction` of the distinguishable ceiling.

  Not `ceiling * margin`: that undersamples badly, because the round-robin planner
  spends most of a small plan re-drawing the specs that saturated first. This
  walks the yield curve and returns the smallest plan whose distinct yield clears
  the target. At the default 0.75 that is roughly three times the ceiling, the
  knee of the curve; chasing the last quartile costs several times more teacher
  generations for each further percent. The result is memoised.

  Duplicates below this point are not waste: the argument-only specs genuinely
  have more distinct scenarios than any single spec's draws reach quickly, and
  every duplicate that dedup drops is one the teacher had to be asked for to get
  the distinct ones interleaved with it. Above the knee that trade stops paying.
  """
  key = (target_fraction, max_multiple)
  if key in _RECOMMENDED_CACHE:
    return _RECOMMENDED_CACHE[key]

  ceiling = train_capacity()
  target = target_fraction * ceiling
  from .conflict_specs import TRAIN_SPECS
  from .dedup import example_fingerprint

  rng = random.Random(20240201)
  seen: set[str] = set()
  cap = ceiling * max_multiple
  result = cap
  for i in range(cap):
    spec = TRAIN_SPECS[i % len(TRAIN_SPECS)]
    pair = build_pair(spec, rng)
    seen.add(
      "\n".join(
        example_fingerprint(example_from_branch(pair, label, "x", [pair.branch(label).call]))
        for label in ("a", "b")
      )
    )
    if len(seen) >= target:
      result = i + 1
      break

  _RECOMMENDED_CACHE[key] = result
  return result


def describe_capacity(specs: Sequence[SwapSpec], planned_pairs: int, **kw) -> str:
  """Planned pairs against what the spec table can actually distinguish."""
  cap = distinct_capacity(specs, **kw)
  total = sum(cap.values())
  per_spec = planned_pairs / max(1, len(specs))
  lines = [
    f"{planned_pairs:,} pairs planned over {len(specs)} specs "
    f"({per_spec:,.0f} per spec); the table can distinguish {total:,}",
    "",
  ]
  for spec_id, n in sorted(cap.items(), key=lambda kv: kv[1]):
    flag = " <- saturated" if per_spec > n else ""
    lines.append(f" {spec_id:<20s} {n:>5,} distinct pairs{flag}")
  if total < planned_pairs:
    lines += [
      "",
      f" Planning {planned_pairs:,} pairs from a table that distinguishes {total:,} buys "
      f"{planned_pairs - total:,} duplicates",
      " at full teacher price. Add specs, phrasings, or argument-only variants first.",
    ]
  return "\n".join(lines)


# ==========================================================================
# Rendering
# ==========================================================================
def _example(pair: Pair, label: str, think: str, calls: Sequence[dict]) -> Example:
  branch = pair.branch(label)
  return Example(
    tools=pair.tools,
    turns=[UserTurn(pair.request["text"]), AssistantTurn(think=think, calls=list(calls))],
    state=branch.state,
    memory=format_memory(
      recent=pair.memory.get("recent", ()),
      last_results=pair.memory.get("last_results", ()),
    ),
    system=DOMAIN_SYSTEM[pair.domain],
    category="state_memory_conflict",
    meta={
      "subtype": pair.subtype,
      "spec_id": pair.spec_id,
      "family": pair.family,
      "pair_id": pair.pair_id,
      "branch": label,
      # Both branches share a group, so dedup and the train/val split move
      # them together or not at all.
      "group": pair.pair_id,
    },
  )


def prompt_block(pair: Pair, label: str) -> str:
  """Exactly the prompt the student will condition on, for this branch.

  Built through `template.render_prompt` rather than restated, so the teacher
  reads the bytes the student will read.
  """
  return render_prompt(_example(pair, label, think="", calls=[]))


def example_from_branch(pair: Pair, label: str, think: str, calls: Sequence[dict]) -> Example:
  return _example(pair, label, think, calls)


# ==========================================================================
# Validation
# ==========================================================================
def validate_branch(raw: str, pair_tools: Sequence[Tool]) -> tuple[str | None, dict | None]:
  """Parse and structurally check one branch's teacher output.

  Returns `(reject_reason, record)`. `<think>` is checked for being non-empty
  and free of leaked markup, and for nothing else. Its wording is never a
  criterion, in either direction.
  """
  record = extract_json(raw or "")
  if record is None:
    return NOT_JSON, None

  reason = check_think(record.get("think"))
  if reason is not None:
    return reason, None

  calls = record.get("calls")
  if not isinstance(calls, list) or not calls:
    return NO_CALLS, None
  if len(calls) != 1:
    # The oracle names one call. Two calls cannot be compared against it, and
    # a teacher that emits two has not answered the question that was asked.
    return SWAP_MULTIPLE_CALLS, None

  allowed = {t["name"]: t for t in pair_tools}
  reason = check_call(calls[0], allowed)
  if reason is not None:
    return reason, None

  return None, record


def validate_pair(
  rows: dict[str, dict],
  spec: SwapSpec,
) -> tuple[dict[str, str], dict[str, Example]]:
  """Accept or reject both branches of one pair, together.

  `rows` maps branch label to the raw shard record. Returns `(reasons,
  examples)`: `reasons` holds a reject reason per branch and is empty on
  acceptance, `examples` holds both branches and is empty on rejection.

  The oracle is recomputed here from `spec` rather than read out of the shard.
  The shard stores the inputs; this module owns the ground truth. Correcting an
  oracle therefore only costs a revalidation pass, never a regeneration run.
  """
  present = [label for label in ("a", "b") if label in rows]
  if len(present) != 2:
    return ({label: SWAP_MISSING_BRANCH for label in present}, {})

  memories = {label: dumps_compact(rows[label]["memory"]) for label in ("a", "b")}
  if memories["a"] != memories["b"]:
    # The model could win by reading memory, so the pair stops testing state.
    return ({"a": SWAP_MEMORY_DIVERGED, "b": SWAP_MEMORY_DIVERGED}, {})

  states = {label: rows[label]["state"] for label in ("a", "b")}
  differing = pointer.diff_paths(states["a"], states["b"])
  if differing != {spec.decisive_path}:
    return ({"a": SWAP_STATE_DIVERGED, "b": SWAP_STATE_DIVERGED}, {})

  request: Request = rows["a"]["request"]
  if dumps_compact(request) != dumps_compact(rows["b"]["request"]):
    return ({"a": SWAP_REQUEST_DIVERGED, "b": SWAP_REQUEST_DIVERGED}, {})

  oracles = {label: spec.oracle(states[label], request) for label in ("a", "b")}
  if calls_agree(oracles["a"], oracles["b"]):
    return ({"a": SWAP_ORACLES_AGREE, "b": SWAP_ORACLES_AGREE}, {})

  tools = rows["a"]["tools"]
  reasons: dict[str, str] = {}
  records: dict[str, dict] = {}
  for label in ("a", "b"):
    reason, record = validate_branch(rows[label].get("raw", ""), tools)
    if reason is not None:
      reasons[label] = reason
    else:
      records[label] = record

  if not reasons:
    for label in ("a", "b"):
      if not calls_agree(records[label]["calls"][0], oracles[label]):
        reasons[label] = SWAP_CALL_MISMATCH

  if reasons:
    # One branch right and one wrong means both go. In isolation, a call the
    # teacher got right by pattern-matching the user turn is indistinguishable
    # from one it got right by reading state.
    return ({label: reasons.get(label, SWAP_PARTNER_FAILED) for label in ("a", "b")}, {})

  pair = _pair_from_rows(rows, spec, oracles)
  return ({}, {label: example_from_branch(pair, label, records[label]["think"], records[label]["calls"]) for label in ("a", "b")})


def _pair_from_rows(rows: dict[str, dict], spec: SwapSpec, oracles: dict[str, Call]) -> Pair:
  return Pair(
    pair_id=rows["a"]["pair_id"],
    spec_id=spec.id,
    family=spec.family,
    domain=spec.domain,
    subtype=rows["a"].get("subtype", spec.subtype),
    tools=rows["a"]["tools"],
    memory=rows["a"]["memory"],
    request=rows["a"]["request"],
    a=Branch("a", rows["a"]["state"], oracles["a"]),
    b=Branch("b", rows["b"]["state"], oracles["b"]),
  )


# ==========================================================================
# Leakage
# ==========================================================================
def state_schema(state: Any, prefix: str = "") -> frozenset[str]:
  """Leaf pointers of a state document, with list indices collapsed to `#`."""
  if isinstance(state, dict):
    out: set[str] = set()
    for key in state:
      out |= state_schema(state[key], f"{prefix}/{pointer.escape(str(key))}")
    return frozenset(out)
  if isinstance(state, list):
    out = set()
    for item in state:
      out |= state_schema(item, f"{prefix}/#")
    return frozenset(out)
  return frozenset({prefix})


def _schemas(specs: Iterable[SwapSpec]) -> set[frozenset[str]]:
  return {state_schema(build_pair(s, random.Random(0)).a.state) for s in specs}


def assert_no_leakage(train: Sequence[SwapSpec], eval_: Sequence[SwapSpec]) -> None:
  """Train and eval share no spec id, tool family, decisive path, or state schema.

  Partitioning at the sample level would leak: the two branches of a pair share
  everything but one field, and two instances of one spec share everything but
  a handful of randomised ids.
  """
  train_ids, eval_ids = {s.id for s in train}, {s.id for s in eval_}
  if train_ids & eval_ids:
    raise ValueError(f"specs on both sides: {sorted(train_ids & eval_ids)}")

  train_families, eval_families = {s.family for s in train}, {s.family for s in eval_}
  if train_families & eval_families:
    raise ValueError(f"tool families on both sides: {sorted(train_families & eval_families)}")

  train_paths, eval_paths = {s.decisive_path for s in train}, {s.decisive_path for s in eval_}
  if train_paths & eval_paths:
    raise ValueError(f"decisive paths on both sides: {sorted(train_paths & eval_paths)}")

  shared_schema = _schemas(train) & _schemas(eval_)
  if shared_schema:
    raise ValueError(f"state schemas on both sides: {sorted(map(sorted, shared_schema))}")

  if not eval_families - train_families:
    raise ValueError("no tool family is held out for evaluation")

  if not {s.domain for s in eval_} - {s.domain for s in train}:
    raise ValueError("no domain is held out for evaluation")
