"""The counterfactual state swap.

The highest-stakes tests in the repository. The paper's central claim is that a
103M model emits tool calls grounded in live state rather than pattern-matched
from the user turn, and category 3 is the data that teaches it. A filter that
admits pattern-matched samples cannot be detected downstream: the loss curve
looks the same, the corpus statistics look the same, and the claim is gone.

So the invariants are asserted directly. Both branches of a pair are identical
outside `decisive_path`. Flipping that field changes the correct call. The
teacher's reasoning is never inspected for wording. A teacher that gets one
branch right and the other wrong loses both.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import random

import pytest

from quick_slm_trainer.sft import conflict
from quick_slm_trainer.sft import validate as V
from quick_slm_trainer.sft.conflict import SpecRejected, assert_no_leakage, build_pair, check_spec
from quick_slm_trainer.sft.conflict_specs import ALL_SPECS, EVAL_SPECS, TRAIN_SPECS, eval_pairs
from quick_slm_trainer.sft.oracles import FREE_TEXT_ARGS, canonical_call, calls_agree
from quick_slm_trainer.sft.pointer import diff_paths
from quick_slm_trainer.sft.specs import CATEGORIES
from quick_slm_trainer.sft.tools import ALL_TOOLS
from quick_slm_trainer.template import dumps_compact, render, render_prompt

EVERY_SPEC = list(ALL_SPECS.values())
SPEC_IDS = [s.id for s in EVERY_SPEC]


# ==========================================================================
# The oracles' canonical form
# ==========================================================================
def test_answer_compares_by_name_alone():
  # No teacher reproduces a hand-written sentence verbatim, and demanding it
  # would reject every correct refusal. This exclusion is also what gives
  # criterion 1 its teeth: a spec whose branches both fall through to prose
  # compares equal here and is rejected.
  a = {"name": "answer", "arguments": {"text": "one wording"}}
  b = {"name": "answer", "arguments": {"text": "an entirely different wording"}}
  assert calls_agree(a, b)
  assert canonical_call(a) == canonical_call(b) == ("answer", ())


def test_only_answers_text_is_exempt_from_comparison():
  assert set(FREE_TEXT_ARGS) == {"answer"}
  assert FREE_TEXT_ARGS["answer"] == frozenset({"text"})


def test_structured_arguments_are_compared():
  a = {"name": "buy", "arguments": {"resource_id": "coal", "quantity": 380}}
  b = {"name": "buy", "arguments": {"resource_id": "coal", "quantity": 20}}
  assert not calls_agree(a, b)


def test_argument_order_does_not_matter():
  a = {"name": "connect", "arguments": {"from_id": "b_1", "to_id": "b_2"}}
  b = {"name": "connect", "arguments": {"to_id": "b_2", "from_id": "b_1"}}
  assert calls_agree(a, b)


def test_an_integer_equals_the_same_float():
  a = {"name": "buy", "arguments": {"resource_id": "coal", "quantity": 20}}
  b = {"name": "buy", "arguments": {"resource_id": "coal", "quantity": 20.0}}
  assert calls_agree(a, b)


def test_a_boolean_does_not_equal_an_integer():
  a = {"name": "toggle_power", "arguments": {"building_id": "b_1", "on": False}}
  b = {"name": "toggle_power", "arguments": {"building_id": "b_1", "on": 0}}
  assert not calls_agree(a, b)


def test_whitespace_around_a_string_argument_is_forgiven():
  a = {"name": "research", "arguments": {"tech_id": "smelting_2"}}
  b = {"name": "research", "arguments": {"tech_id": " smelting_2 "}}
  assert calls_agree(a, b)


# ==========================================================================
# Spec integrity
# ==========================================================================
@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_every_spec_builds_a_pair_for_many_seeds(spec_id):
  check_spec(ALL_SPECS[spec_id], trials=24)


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_the_decisive_flip_changes_the_correct_call(spec_id):
  # Criterion 1. If flipping the field does not change the call, the pair cannot
  # separate grounding from imitation, and the spec is rejected, not the sample.
  spec = ALL_SPECS[spec_id]
  for seed in range(6):
    pair = build_pair(spec, random.Random(seed))
    assert not calls_agree(pair.a.call, pair.b.call), (spec_id, seed)


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_the_branches_differ_at_exactly_the_decisive_path(spec_id):
  spec = ALL_SPECS[spec_id]
  for seed in range(6):
    pair = build_pair(spec, random.Random(seed))
    assert diff_paths(pair.a.state, pair.b.state) == {spec.decisive_path}


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_the_prompts_of_a_pair_differ_only_inside_the_state_block(spec_id):
  # Any incidental difference, an id or a reordered key, is a cue the model can
  # read instead of the state.
  spec = ALL_SPECS[spec_id]
  for seed in range(4):
    pair = build_pair(spec, random.Random(seed))
    changed = [
      line
      for line in difflib.unified_diff(
        conflict.prompt_block(pair, "a").split("\n"),
        conflict.prompt_block(pair, "b").split("\n"),
        lineterm="",
      )
      if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 2, (spec_id, seed, changed)
    assert all("<state>" in line for line in changed), (spec_id, seed, changed)


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_memory_and_the_user_turn_are_shared_by_both_branches(spec_id):
  pair = build_pair(ALL_SPECS[spec_id], random.Random(3))
  # `memory` and `request` hang off the pair, not the branch, so they cannot
  # diverge by construction. Assert the construction rather than trust it.
  assert pair.memory and pair.request["text"]
  assert "<memory>" in conflict.prompt_block(pair, "a")
  assert conflict.prompt_block(pair, "a").count(pair.request["text"]) == 1


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_both_oracle_calls_are_legal_against_the_offered_tools(spec_id):
  spec = ALL_SPECS[spec_id]
  offered = {t["name"] for t in spec.tools()}
  for seed in range(4):
    pair = build_pair(spec, random.Random(seed))
    for branch in pair.branches():
      assert branch.call["name"] in offered


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_answer_is_offered_and_never_listed_explicitly(spec_id):
  spec = ALL_SPECS[spec_id]
  assert "answer" not in spec.tool_names
  assert spec.tools()[-1] is ALL_TOOLS["answer"]


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_the_variants_are_two_different_values(spec_id):
  a, b = ALL_SPECS[spec_id].variants
  assert a != b or type(a) is not type(b)


@pytest.mark.parametrize("spec_id", SPEC_IDS)
def test_the_subtype_is_one_the_category_declares(spec_id):
  assert ALL_SPECS[spec_id].subtype in CATEGORIES["state_memory_conflict"].subtypes


@pytest.mark.parametrize("spec_id", [s.id for s in TRAIN_SPECS])
def test_training_specs_offer_a_tool_count_the_category_allows(spec_id):
  lo, hi = CATEGORIES["state_memory_conflict"].n_tools
  assert lo <= len(ALL_SPECS[spec_id].tool_names) <= hi


def test_at_least_one_spec_keeps_the_tool_fixed_and_moves_only_an_argument():
  # The sharpest form of the test. A model that guesses `buy` from the user turn
  # still has to read `<state>` to fill the quantity.
  pair = build_pair(ALL_SPECS["inventory_amount"], random.Random(1))
  assert pair.a.call["name"] == pair.b.call["name"] == "buy"
  assert pair.a.call["arguments"]["quantity"] != pair.b.call["arguments"]["quantity"]


def test_a_spec_whose_flip_changes_nothing_is_rejected():
  # `power_priority` asks for priority 5. Flipping between two values that are
  # both not 5 leaves `set_priority(5)` correct on both branches.
  broken = dataclasses.replace(ALL_SPECS["power_priority"], variants=(1, 2))
  with pytest.raises(SpecRejected, match="does not change the correct call"):
    build_pair(broken, random.Random(0))


def test_a_spec_with_a_typoed_decisive_path_is_rejected():
  broken = dataclasses.replace(ALL_SPECS["building_run"], decisive_path="/buildings/0/pausd")
  with pytest.raises(SpecRejected, match="does not resolve"):
    build_pair(broken, random.Random(0))


def test_a_spec_with_identical_variants_is_rejected():
  broken = dataclasses.replace(ALL_SPECS["building_run"], variants=(True, True))
  with pytest.raises(SpecRejected, match="same value"):
    build_pair(broken, random.Random(0))


# ==========================================================================
# Leakage
# ==========================================================================
def test_train_and_eval_share_nothing():
  assert_no_leakage(TRAIN_SPECS, EVAL_SPECS)


def test_a_spec_on_both_sides_is_caught():
  with pytest.raises(ValueError, match="specs on both sides"):
    assert_no_leakage(TRAIN_SPECS, (*EVAL_SPECS, TRAIN_SPECS[0]))


def test_a_family_on_both_sides_is_caught():
  # A different spec id, but the same tool family. The design doc forbids it:
  # the eval is the same construction, so a shared family is a shared prior.
  intruder = dataclasses.replace(TRAIN_SPECS[0], id="intruder")
  with pytest.raises(ValueError, match="tool families on both sides"):
    assert_no_leakage(TRAIN_SPECS, (*EVAL_SPECS, intruder))


def test_a_decisive_path_on_both_sides_is_caught():
  intruder = dataclasses.replace(EVAL_SPECS[0], id="intruder", family="research_alt",
                  decisive_path=TRAIN_SPECS[0].decisive_path)
  with pytest.raises(ValueError, match="decisive paths on both sides"):
    assert_no_leakage(TRAIN_SPECS, (*EVAL_SPECS, intruder))


def test_holding_nothing_out_is_caught():
  with pytest.raises(ValueError, match="no tool family is held out"):
    assert_no_leakage(TRAIN_SPECS, ())


def test_at_least_one_domain_is_held_out():
  # `weather_cache` is the only world-domain conflict spec, so state-grounding
  # is measured on a domain no training conflict spec ever showed the model.
  assert {s.domain for s in EVAL_SPECS} - {s.domain for s in TRAIN_SPECS} == {"world"}


def test_the_held_out_families_are_named_by_no_training_spec():
  assert {s.family for s in EVAL_SPECS} & {s.family for s in TRAIN_SPECS} == set()


def test_eval_pairs_need_no_teacher():
  # The oracle is the label, so the evaluation harness generates its own answers.
  # A model that reads the user turn instead of `<state>` scores 50% whichever
  # way it guesses, because both branches carry the same user turn.
  pairs = eval_pairs(6, random.Random(0))
  assert len(pairs) == 6
  assert {p.spec_id for p in pairs} <= {s.id for s in EVAL_SPECS}
  for pair in pairs:
    assert not calls_agree(pair.a.call, pair.b.call)
    assert pair.request["text"] == pair.request["text"]


# ==========================================================================
# Pair validation
# ==========================================================================
THINK = "The state block is what I act on, so I read it and choose accordingly."


def rows_for(pair, call_a, call_b, *, think_a=THINK, think_b=THINK) -> dict[str, dict]:
  calls = {"a": call_a, "b": call_b}
  thinks = {"a": think_a, "b": think_b}
  return {
    label: {
      "pair_id": pair.pair_id,
      "branch": label,
      "spec_id": pair.spec_id,
      "subtype": pair.subtype,
      "tools": pair.tools,
      "state": pair.branch(label).state,
      "memory": pair.memory,
      "request": pair.request,
      "raw": json.dumps({"think": thinks[label], "calls": [calls[label]]}),
    }
    for label in ("a", "b")
  }


@pytest.fixture
def pair():
  return build_pair(ALL_SPECS["inventory_amount"], random.Random(3))


@pytest.fixture
def spec():
  return ALL_SPECS["inventory_amount"]


def test_both_branches_correct_is_accepted(pair, spec):
  reasons, examples = conflict.validate_pair(rows_for(pair, pair.a.call, pair.b.call), spec)
  assert reasons == {}
  assert set(examples) == {"a", "b"}


def test_both_branches_go_into_training(pair, spec):
  _, examples = conflict.validate_pair(rows_for(pair, pair.a.call, pair.b.call), spec)
  assert examples["a"].category == examples["b"].category == "state_memory_conflict"
  assert examples["a"].meta["group"] == examples["b"].meta["group"] == pair.pair_id


def test_one_right_and_one_wrong_drops_both(pair, spec):
  reasons, examples = conflict.validate_pair(rows_for(pair, pair.a.call, pair.a.call), spec)
  assert examples == {}
  assert reasons["b"] == V.SWAP_CALL_MISMATCH
  assert reasons["a"] == V.SWAP_PARTNER_FAILED


def test_the_other_order_also_drops_both(pair, spec):
  reasons, examples = conflict.validate_pair(rows_for(pair, pair.b.call, pair.b.call), spec)
  assert examples == {}
  assert reasons["a"] == V.SWAP_CALL_MISMATCH
  assert reasons["b"] == V.SWAP_PARTNER_FAILED


def test_perfect_narration_cannot_rescue_a_pattern_matched_call(pair, spec):
  # The exact sample the old `<think>` filter admitted: reasoning that recites
  # the policy, and a call read off the user turn rather than off the state.
  copied = {
    "name": "buy",
    "arguments": {"resource_id": pair.request["resource_id"], "quantity": pair.request["target"]},
  }
  narration = "The state says one thing and memory says another. State is ground truth, so I trust it."
  reasons, examples = conflict.validate_pair(
    rows_for(pair, copied, copied, think_a=narration, think_b=narration), spec
  )
  assert examples == {}
  assert V.SWAP_CALL_MISMATCH in reasons.values()


def test_wording_of_think_is_never_a_criterion(pair, spec):
  # The inverse: reasoning that names neither concept, with both calls right.
  terse = "Stock is short of the target, so I top it up by the difference."
  reasons, examples = conflict.validate_pair(
    rows_for(pair, pair.a.call, pair.b.call, think_a=terse, think_b=terse), spec
  )
  assert reasons == {}
  assert set(examples) == {"a", "b"}


def test_answer_prose_need_not_match_the_oracle():
  spec = ALL_SPECS["building_run"]
  pair = build_pair(spec, random.Random(4))
  loose = {"name": "answer", "arguments": {"text": "It is already running, nothing to do."}}
  reasons, examples = conflict.validate_pair(rows_for(pair, pair.a.call, loose), spec)
  assert reasons == {}
  assert set(examples) == {"a", "b"}


def test_a_missing_branch_is_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  del rows["b"]
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons == {"a": V.SWAP_MISSING_BRANCH}


def test_divergent_memory_is_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["b"]["memory"] = {"recent": ["something else entirely"], "last_results": []}
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == reasons["b"] == V.SWAP_MEMORY_DIVERGED


def test_state_differing_anywhere_else_is_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["b"]["state"] = {**rows["b"]["state"], "order": {"resource_id": "zzz", "target": 9}}
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.SWAP_STATE_DIVERGED


def test_a_divergent_user_turn_is_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["b"]["request"] = {**rows["b"]["request"], "text": "a completely different question"}
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.SWAP_REQUEST_DIVERGED


def test_two_calls_cannot_be_compared_to_a_single_call_oracle(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["a"]["raw"] = json.dumps({"think": THINK, "calls": [pair.a.call, pair.b.call]})
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.SWAP_MULTIPLE_CALLS


def test_an_empty_think_is_still_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["a"]["raw"] = json.dumps({"think": "ok", "calls": [pair.a.call]})
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.EMPTY_THINK


def test_unparseable_teacher_output_is_rejected(pair, spec):
  rows = rows_for(pair, pair.a.call, pair.b.call)
  rows["a"]["raw"] = "I am terribly sorry, I cannot help with that."
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.NOT_JSON


def test_a_hallucinated_tool_is_rejected(pair, spec):
  rows = rows_for(pair, {"name": "teleport", "arguments": {}}, pair.b.call)
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["a"] == V.UNDEFINED_TOOL


def test_the_oracle_is_recomputed_rather_than_trusted(pair, spec):
  # The shard holds inputs. Planting a verdict in it must change nothing.
  rows = rows_for(pair, pair.a.call, pair.a.call)
  rows["a"]["oracle"] = pair.a.call
  rows["b"]["oracle"] = pair.a.call # a forged label agreeing with the wrong call
  reasons, examples = conflict.validate_pair(rows, spec)
  assert examples == {}
  assert reasons["b"] == V.SWAP_CALL_MISMATCH


def test_accepted_examples_share_a_prompt_and_differ_in_the_response(pair, spec):
  _, examples = conflict.validate_pair(rows_for(pair, pair.a.call, pair.b.call), spec)
  a, b = examples["a"], examples["b"]

  prompt_diff = [
    line
    for line in difflib.unified_diff(render_prompt(a).split("\n"), render_prompt(b).split("\n"), lineterm="")
    if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
  ]
  assert len(prompt_diff) == 2 and all("<state>" in line for line in prompt_diff)

  # The same user turn, two different correct answers. That is the mechanism.
  assert dumps_compact(a.turns[0].text) == dumps_compact(b.turns[0].text)
  assert a.turns[1].calls != b.turns[1].calls
  assert "<response>" in render(a) and "<response>" in render(b)


def test_the_scored_span_is_the_assistant_turn_only(pair, spec, tok):
  from quick_slm_trainer.template import IGNORE_INDEX, encode

  _, examples = conflict.validate_pair(rows_for(pair, pair.a.call, pair.b.call), spec)
  ids, labels = encode(examples["a"], tok)
  scored = [i for i, v in enumerate(labels) if v != IGNORE_INDEX]
  assert scored and len(scored) < len(ids)
  # `<state>` is server-injected at inference time; grading it is pure waste.
  state_ids = tok("<state>")["input_ids"]
  assert all(labels[i] == IGNORE_INDEX for i, v in enumerate(ids) if v == state_ids[0])


# ==========================================================================
# How much the spec table can actually distinguish
# ==========================================================================
# A spec randomises a building type, a resource, and a phrasing. Nothing else it
# randomises survives `dedup.example_fingerprint`: the building ids are stripped
# and `answer`'s prose is excluded, both because `oracles.FREE_TEXT_ARGS` already
# says no comparison may depend on that prose. So a spec's capacity is the product
# of its own small choice lists, and planning past it buys duplicates at teacher
# price. These numbers are pinned so that raising them is a visible, deliberate act.
# The trial count has to exceed the largest per-spec capacity by enough for the
# random sampler to hit every distinct document. The argument-only specs top out
# at 384, so 6,000 saturates with room to spare; the numbers are stable from
# ~3,000 up (checked by hand at 6,000 vs 9,000).
_SAT = 6_000


def test_the_training_table_capacity_is_pinned():
  # Pinned so that raising it is a deliberate, reviewed act, and so that a change
  # to `_bid`, to the fingerprint, or to a phrasing list that quietly shrinks
  # capacity fails here. The argument-only specs (inventory_*, market_selldown)
  # carry the category: their call argument is a state-derived quantity, so
  # randomising the target/floor multiplies distinct documents where the
  # tool-name specs are stuck at building-types x phrasings.
  cap = conflict.distinct_capacity(TRAIN_SPECS, trials=_SAT)
  assert cap == {
    "connection_link": 8,   # 8 phrasings; both ids normalised away
    "production_recipe": 18,  # 2 building types x 9 phrasings
    "power_shed": 48,     # 6 x 8
    "power_priority": 48,   # 6 x 8
    "building_run": 60,    # 6 x 10
    "building_stop": 60,    # 6 x 10
    "inventory_or_skip": 288, # 6 resources x 8 phrasings x 6 targets
    "inventory_amount": 384,  # 6 x 8 x 8 targets
    "market_selldown": 384,  # 6 x 8 x 8 floors
  }
  assert sum(cap.values()) == 1_298


def test_the_eval_table_capacity_is_pinned():
  # The paper's headline evaluation runs on these. Two held-out world families
  # now, weather and stocks, so the number is not one scenario's luck. The
  # research family stays phrasing-bound because its target tech is tied to the
  # spec's `variants` constant and cannot be randomised per instance.
  cap = conflict.distinct_capacity(EVAL_SPECS, trials=_SAT)
  assert cap == {
    "research_start": 10,  # 10 phrasings, fixed tech
    "research_switch": 10, # 10 phrasings, fixed tech
    "stock_cache": 80,   # 10 tickers x 8 phrasings
    "weather_cache": 96,  # 12 cities x 8 phrasings
  }
  assert sum(cap.values()) == 196


def test_capacity_is_deterministic():
  a = conflict.distinct_capacity(TRAIN_SPECS, trials=120)
  b = conflict.distinct_capacity(TRAIN_SPECS, trials=120)
  assert a == b


def test_describe_capacity_names_the_saturated_specs_and_the_waste():
  text = conflict.describe_capacity(TRAIN_SPECS, 8_000, trials=_SAT)
  assert "saturated" in text
  assert "6,702 duplicates" in text # 8,000 planned - 1,298 distinguishable
  assert "at full teacher price" in text


def test_the_yield_curve_rises_toward_the_ceiling_without_reaching_it():
  # The ceiling is the unlimited-draw limit. At a finite plan the round-robin
  # planner yields less, and the curve must climb monotonically toward it.
  ceiling = conflict.train_capacity()
  curve = conflict.conflict_yield_curve([1_000, 2_000, 4_000])
  planned = [p for p, _ in curve]
  distinct = [d for _, d in curve]
  assert planned == [1_000, 2_000, 4_000]
  assert distinct == sorted(distinct)     # more planned, more distinct
  assert all(d < ceiling for d in distinct)  # finite draws never saturate


def test_recommended_pairs_reaches_its_target_fraction_by_overshooting_the_ceiling():
  # The bug this pins: `recommended` used to be `ceiling * 1.3`, which yields
  # only ~half the ceiling because the round-robin planner spends a small plan
  # re-drawing the specs that saturated first. The honest recommendation walks
  # the yield curve and overshoots the ceiling to reach a *distinct* target.
  ceiling = conflict.train_capacity()
  n = conflict.recommended_conflict_pairs()
  (planned, distinct), = conflict.conflict_yield_curve([n])
  assert planned == n
  assert 0.74 <= distinct / ceiling <= 0.80, distinct
  assert n > ceiling, "must plan above the ceiling to reach a large fraction of it"


def test_recommended_pairs_is_deterministic_and_cached():
  assert conflict.recommended_conflict_pairs() == conflict.recommended_conflict_pairs()
