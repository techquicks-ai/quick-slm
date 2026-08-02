"""Category specs and the tool universe.

The specs are read by two things that must not disagree: the prompt builder,
which tells the teacher what to write, and the validator, which decides whether
it wrote it. A spec whose turn bounds contradict its own sub-type text produces a
category that rejects most of what it asked for, at full teacher cost.
"""

from __future__ import annotations

import random

import pytest

from quick_slm_trainer.config import SFTConfig
from quick_slm_trainer.sft.prompts import (
  build_output_schema,
  build_teacher_prompt,
  seed_pool,
  seed_topic,
  usable_seeds,
)
from quick_slm_trainer.sft.specs import CATEGORIES, CategorySpec, cycle, plan_counts
from quick_slm_trainer.sft.tools import (
  ALL_TOOLS,
  ANSWER,
  DOMAIN_SYSTEM,
  DOMAIN_TOOLS,
  has_optional_args,
  properties_of,
  required_args,
  sample_tools,
  tool_names,
)

UNPAIRED = sorted(k for k, s in CATEGORIES.items() if not s.paired)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def test_every_tool_schema_is_well_formed():
  for name, tool in ALL_TOOLS.items():
    assert tool["name"] == name
    assert tool["description"].strip()
    params = tool["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) <= set(params["properties"]), name


def test_answer_is_the_only_universal_tool():
  assert "answer" in ALL_TOOLS
  assert required_args(ANSWER) == {"text"}


def test_the_universe_is_about_thirty_tools():
  # `docs/sft_readme.md` asks for a fixed set of ~30.
  assert 28 <= len(ALL_TOOLS) <= 36


def test_some_tools_declare_optional_parameters():
  # Without them the `optional_args` sub-type has nothing to exercise, and the
  # model never learns to omit a field the user did not mention.
  optional = [n for n, t in ALL_TOOLS.items() if set(properties_of(t)) - required_args(t)]
  assert len(optional) >= 3, f"only {optional} have optional parameters"


def test_every_domain_has_a_system_prompt():
  assert set(DOMAIN_SYSTEM) == set(DOMAIN_TOOLS)


def test_sample_tools_always_appends_answer():
  rng = random.Random(0)
  for domain in DOMAIN_TOOLS:
    tools = sample_tools(rng, domain, 3)
    assert tools[-1]["name"] == "answer"
    assert len(tools) == 4


def test_sample_tools_honours_exclude():
  # The no-relevant-tool trap only works if the obvious tool is absent.
  rng = random.Random(1)
  for _ in range(20):
    tools = sample_tools(rng, "factory", 6, exclude=["inspect", "list_buildings"])
    assert "inspect" not in tool_names(tools)
    assert "list_buildings" not in tool_names(tools)


def test_sample_tools_honours_include():
  rng = random.Random(2)
  for _ in range(20):
    assert "web_search" in tool_names(sample_tools(rng, "world", 4, include=["web_search"]))


def test_exclude_beats_include_rather_than_producing_a_contradiction():
  rng = random.Random(3)
  tools = sample_tools(rng, "world", 4, include=["web_search"], exclude=["web_search"])
  assert "web_search" not in tool_names(tools)


def test_sample_tools_does_not_repeat_a_tool():
  rng = random.Random(4)
  tools = sample_tools(rng, "factory", 8, include=["inspect"])
  names = [t["name"] for t in tools]
  assert len(names) == len(set(names))


def test_asking_for_more_tools_than_exist_is_capped_not_an_error():
  rng = random.Random(5)
  tools = sample_tools(rng, "world", 999)
  assert len(tools) == len(DOMAIN_TOOLS["world"]) + 1


def test_sample_tools_rejects_an_unknown_domain():
  with pytest.raises(KeyError, match="unknown domain"):
    sample_tools(random.Random(6), "nonsense", 3)


def test_sample_tools_refuses_to_pin_a_tool_from_another_domain():
  # A world tool pinned into a factory list is offered to the teacher and then
  # rejected by the validator, which checks the call against the tools on offer.
  # A mistyped `Seed.requires` should fail here, not eleven hours into a run.
  with pytest.raises(KeyError, match="cannot pin"):
    sample_tools(random.Random(6), "factory", 4, include=["get_weather"])
  with pytest.raises(KeyError, match="cannot pin"):
    sample_tools(random.Random(6), "world", 4, include=["inspect"])


def test_sample_tools_refuses_to_pin_answer_because_it_is_always_appended():
  with pytest.raises(KeyError, match="cannot pin"):
    sample_tools(random.Random(6), "world", 4, include=["answer"])


def test_exactly_the_tools_with_optional_parameters_report_as_such():
  optional = {n for n, t in ALL_TOOLS.items() if has_optional_args(t)}
  assert optional == {"get_weather", "web_search", "list_buildings", "sell"}
  assert not has_optional_args(ANSWER)


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------
def test_the_five_categories_of_the_design_doc_are_present():
  assert set(CATEGORIES) == {
    "single_stage",
    "multi_stage",
    "state_memory_conflict",
    "traps",
    "refusals",
  }


def test_spec_shares_sum_to_one():
  assert sum(s.share for s in CATEGORIES.values()) == pytest.approx(1.0)


def test_default_config_shares_match_the_specs():
  cfg = SFTConfig()
  for key, spec in CATEGORIES.items():
    assert cfg.category_shares[key] == pytest.approx(spec.share)


@pytest.mark.parametrize("key", sorted(CATEGORIES))
def test_turn_bounds_are_sane(key):
  lo, hi = CATEGORIES[key].n_assistant_turns
  assert 1 <= lo <= hi


def test_multi_stage_allows_a_four_step_chain_plus_its_answer_turn():
  # `chain_4` asks for four tool-calling turns and a final `answer` turn.
  # A bound of (2, 4) would reject every example the prompt requested.
  lo, hi = CATEGORIES["multi_stage"].n_assistant_turns
  assert lo >= 3 and hi >= 5


def test_every_category_whose_guidance_demands_a_final_answer_enforces_it():
  # `multi_stage` asked for one in prose and checked for nothing: its guidance
  # says "The final assistant turn calls `answer` to deliver the result", and
  # `final_call_must_be_answer` was left False. A chain that stops on its last
  # tool call never learns to report what it found.
  forced = {k for k, s in CATEGORIES.items() if s.final_call_must_be_answer}
  assert forced == {"multi_stage", "traps", "refusals"}

  # Single-stage calls one real tool; the paired category is compared against a
  # single-call oracle. Neither delivers prose.
  assert not CATEGORIES["single_stage"].final_call_must_be_answer
  assert not CATEGORIES["state_memory_conflict"].final_call_must_be_answer


def test_no_category_constrains_the_wording_of_the_think_block():
  # `think_must_mention` is deleted rather than emptied. An empty tuple would
  # have left a filter that silently passes; a missing field makes a spec that
  # tries to set it fail loudly. Selecting on the wording of a reasoning block
  # rewards narration over behaviour, which is what `conflict.py` replaced.
  for spec in CATEGORIES.values():
    assert not hasattr(spec, "think_must_mention"), spec.key

  with pytest.raises(TypeError):
    CategorySpec(
      key="x", share=1.0, domains=("world",), subtypes=("a",),
      n_tools=(1, 1), n_assistant_turns=(1, 1),
      think_must_mention=("state", "memory"),
    )


def test_only_the_conflict_category_is_paired():
  paired = {k for k, s in CATEGORIES.items() if s.paired}
  assert paired == {"state_memory_conflict"}


def test_the_paired_category_takes_exactly_one_assistant_turn():
  # The oracle names one call. Two assistant turns cannot be compared to it.
  assert CATEGORIES["state_memory_conflict"].n_assistant_turns == (1, 1)


def test_state_and_memory_are_required_together():
  for spec in CATEGORIES.values():
    assert spec.needs_state == spec.needs_memory, spec.key


def test_every_subtype_has_guidance():
  for spec in CATEGORIES.values():
    for sub in spec.subtypes:
      assert spec.subtype_guidance.get(sub, "").strip(), f"{spec.key}/{sub}"


def test_every_domain_a_spec_names_actually_exists():
  for spec in CATEGORIES.values():
    assert set(spec.domains) <= set(DOMAIN_TOOLS), spec.key


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def test_plan_counts_sums_exactly_to_the_target():
  cfg = SFTConfig(target_examples=80_000)
  counts = plan_counts(cfg)
  assert sum(counts.values()) == 80_000


def test_plan_counts_absorbs_rounding_into_the_largest_unpaired_category():
  counts = plan_counts(SFTConfig(target_examples=7))
  assert sum(counts.values()) == 7
  assert counts["single_stage"] == max(counts.values())


@pytest.mark.parametrize("target", [7, 101, 999, 1_000, 80_000])
def test_the_paired_category_is_always_planned_in_whole_pairs(target):
  # An odd count plans half a pair. Its surviving branch is exactly the
  # isolated sample the construction exists to exclude.
  counts = plan_counts(SFTConfig(target_examples=target))
  assert counts["state_memory_conflict"] % 2 == 0
  assert sum(counts.values()) == target


def test_plan_counts_rejects_shares_that_do_not_sum_to_one():
  cfg = SFTConfig()
  cfg.category_shares["traps"] = 0.5
  with pytest.raises(ValueError, match="sum to"):
    plan_counts(cfg)


def test_plan_counts_rejects_an_unknown_category():
  cfg = SFTConfig()
  cfg.category_shares["not_a_category"] = 0.0
  with pytest.raises(KeyError, match="do not exist"):
    plan_counts(cfg)


def test_plan_counts_rejects_a_missing_category():
  cfg = SFTConfig()
  del cfg.category_shares["refusals"]
  cfg.category_shares["traps"] += 0.05
  with pytest.raises(KeyError, match="omits"):
    plan_counts(cfg)


def test_cycle_is_a_deterministic_round_robin():
  assert [cycle(("a", "b", "c"), i) for i in range(4)] == ["a", "b", "c", "a"]


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def test_teacher_prompt_lists_the_tools_it_permits():
  rng = random.Random(7)
  tools = sample_tools(rng, "factory", 4)
  prompt = build_teacher_prompt(
    CATEGORIES["single_stage"], tools, subtype="direct",
    seed_topic=seed_topic(rng, "single_stage", "factory", "direct"), domain="factory",
  )
  for tool in tools:
    assert f'"{tool["name"]}"' in prompt


def test_teacher_prompt_states_the_answer_only_rule_for_traps():
  rng = random.Random(8)
  prompt = build_teacher_prompt(
    CATEGORIES["traps"], sample_tools(rng, "world", 3), subtype="no_relevant_tool",
    seed_topic="x", domain="world",
  )
  assert "calls `answer` and nothing else" in prompt


def test_the_conflict_prompt_never_asks_the_teacher_to_narrate():
  # It once demanded the literal words "state" and "memory" in `<think>`, and
  # the validator then checked for them. Asking for narration is how a filter
  # comes to select for narration. The prompt states the policy, the oracle
  # decides whether it was applied, and the wording is never mentioned.
  import random as _random

  from quick_slm_trainer.sft.conflict import build_pair, prompt_block
  from quick_slm_trainer.sft.conflict_specs import TRAIN_SPECS
  from quick_slm_trainer.sft.prompts import build_conflict_prompt

  spec = TRAIN_SPECS[0]
  pair = build_pair(spec, _random.Random(0))
  prompt = build_conflict_prompt(
    prompt_block(pair, "a"),
    guidance=CATEGORIES["state_memory_conflict"].guidance,
    subtype_guidance=CATEGORIES["state_memory_conflict"].subtype_guidance[spec.subtype],
  )
  assert '"state" and "memory"' not in prompt
  assert "must name the disagreement" not in prompt
  assert "wording is not" in prompt
  assert "outranks" in prompt


def test_teacher_prompt_nulls_state_and_memory_where_they_do_not_belong():
  rng = random.Random(10)
  prompt = build_teacher_prompt(
    CATEGORIES["single_stage"], sample_tools(rng, "world", 3), subtype="direct",
    seed_topic="x", domain="world",
  )
  assert "must both be null" in prompt


def test_seed_topics_exist_for_every_reachable_category_domain_subtype():
  # The paired category is absent: its scenarios come from `conflict_specs`,
  # not from a seed pool, because a counterfactual pair needs a decisive field
  # and an oracle rather than a topic.
  rng = random.Random(11)
  for key, spec in CATEGORIES.items():
    if spec.paired:
      continue
    for domain in spec.domains:
      for subtype in spec.subtypes_for(domain):
        assert seed_topic(rng, key, domain, subtype).strip(), f"{key}/{domain}/{subtype}"


def test_the_factory_domain_never_asks_for_a_web_search_fallback():
  # `web_search` is a world tool. A factory chain that falls back to it is a
  # request the teacher cannot satisfy and the validator will reject.
  assert "search_fallback" not in CATEGORIES["multi_stage"].subtypes_for("factory")
  assert "search_fallback" in CATEGORIES["multi_stage"].subtypes_for("world")


def test_domain_subtypes_are_a_subset_of_the_declared_subtypes():
  for spec in CATEGORIES.values():
    for domain, subs in spec.domain_subtypes.items():
      assert domain in spec.domains, f"{spec.key} restricts a domain it does not use"
      assert set(subs) <= set(spec.subtypes), spec.key
      assert subs, f"{spec.key}/{domain} has no usable sub-type"


def test_trap_and_refusal_seeds_are_keyed_by_subtype():
  # A `no_relevant_tool` prompt paired with a missing-argument seed gives the
  # teacher two contradictory instructions.
  for key in ("traps", "refusals"):
    spec = CATEGORIES[key]
    pools = {
      (d, s): seed_pool(key, d, s)
      for d in spec.domains
      for s in spec.subtypes_for(d)
    }
    for domain in spec.domains:
      subs = spec.subtypes_for(domain)
      seen = [pools[(domain, s)] for s in subs]
      assert len({id(p) for p in seen}) == len(subs), f"{key}/{domain} reuses one pool"


# --------------------------------------------------------------------------
# The output schema, built per category
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", UNPAIRED)
def test_the_output_schema_agrees_with_the_spec_the_validator_reads(key):
  # One schema served all five categories. It showed a populated `memory`
  # object and three turns, one of them a tool turn, to categories whose own
  # STRUCTURE section demanded null memory and a single turn. Every reject it
  # would have produced -- MEMORY_UNEXPECTED, BAD_TURN_ORDER,
  # TURN_COUNT_OUT_OF_BOUNDS -- is asserted against here.
  spec = CATEGORIES[key]
  schema = build_output_schema(spec)
  lo, _hi = spec.n_assistant_turns

  assert ('"state": null' in schema) == (not spec.needs_state), key
  assert ('"memory": null' in schema) == (not spec.needs_memory), key
  assert schema.count('"role": "assistant"') == lo, key
  assert schema.count('"role": "tool"') == lo - 1, key

  final_turn = schema.rsplit('"role": "assistant"', 1)[1]
  assert ('"name": "answer"' in final_turn) == spec.final_call_must_be_answer, key


@pytest.mark.parametrize("key", UNPAIRED)
def test_a_schema_that_omits_legal_turns_says_how_to_add_them(key):
  spec = CATEGORIES[key]
  prompt = build_teacher_prompt(
    spec, sample_tools(random.Random(13), spec.domains[0], spec.n_tools[0]),
    subtype=spec.subtypes[0], seed_topic="x", domain=spec.domains[0],
  )
  lo, hi = spec.n_assistant_turns
  if lo == hi:
    assert "shortest legal shape" not in prompt, key
  else:
    # The schema shows `lo` turns. Nothing else in the prompt shows the shape
    # of a tool turn, so the note has to carry it.
    assert f"Up to {hi} are allowed" in prompt, key
    assert '"role": "tool"' in prompt, key


@pytest.mark.parametrize("key", UNPAIRED)
def test_a_record_shaped_like_the_schema_passes_the_validator(key):
  # The spec is read by the prompt builder, which tells the teacher what to
  # write, and by the validator, which decides whether it wrote it. This module
  # exists so the two cannot disagree. A teacher that copies the schema exactly
  # must be accepted; when it was not, the reject arrived after the tokens were
  # paid for.
  from quick_slm_trainer.sft.validate import validate_record

  spec = CATEGORIES[key]
  lo, _hi = spec.n_assistant_turns
  tools = sample_tools(random.Random(16), spec.domains[0], spec.n_tools[1])

  filler = {"string": "x", "integer": 1, "number": 1.0, "boolean": True, "object": {}, "array": []}

  def args_for(tool):
    return {k: filler[properties_of(tool)[k]["type"]] for k in required_args(tool)}

  real = next(t for t in tools if t["name"] != "answer")
  tail = ANSWER if spec.final_call_must_be_answer else real

  turns = []
  for _ in range(lo - 1):
    turns.append(
      {"role": "assistant", "think": "reasoning, long enough", "calls": [{"name": real["name"], "arguments": args_for(real)}]}
    )
    turns.append({"role": "tool", "result": {"ok": True}})
  turns.append(
    {"role": "assistant", "think": "reasoning, long enough", "calls": [{"name": tail["name"], "arguments": args_for(tail)}]}
  )

  record = {
    "user": "A specific, self-contained request.",
    "state": {} if spec.needs_state else None,
    "memory": None,
    "turns": turns,
  }
  assert validate_record(record, spec, tools) is None, key


def test_the_single_stage_prompt_no_longer_contradicts_itself():
  # STRUCTURE and OUTPUT FORMAT are two halves of one prompt. STRUCTURE said
  # one turn and null memory; the schema below it showed three turns and a
  # populated memory object. Forty percent of the corpus read both.
  prompt = build_teacher_prompt(
    CATEGORIES["single_stage"], sample_tools(random.Random(14), "world", 3),
    subtype="direct", seed_topic="x", domain="world",
  )
  assert "`turns` has length 1 and contains no tool turn" in prompt
  assert "must both be null" in prompt
  assert '"role": "tool"' not in prompt
  assert prompt.count('"role": "assistant"') == 1
  assert '"state": null' in prompt and '"memory": null' in prompt


def test_the_multi_stage_schema_shows_a_chain_that_ends_in_an_answer():
  # Counted in the schema, not in the whole prompt: `_turn_note` carries a
  # fourth `"role": "tool"` to show how the chain extends to five turns.
  schema = build_output_schema(CATEGORIES["multi_stage"])
  assert schema.count('"role": "assistant"') == 3
  assert schema.count('"role": "tool"') == 2
  assert schema.rsplit('"role": "assistant"', 1)[1].count('"name": "answer"') == 1

  prompt = build_teacher_prompt(
    CATEGORIES["multi_stage"], sample_tools(random.Random(15), "world", 4),
    subtype="chain_2", seed_topic="x", domain="world",
  )
  assert "calls `answer` and nothing else" in prompt
  assert "Up to 5 are allowed" in prompt


# --------------------------------------------------------------------------
# Seeds and the tools they presuppose
# --------------------------------------------------------------------------
def _all_pools():
  for key in UNPAIRED:
    spec = CATEGORIES[key]
    for domain in spec.domains:
      for subtype in spec.subtypes_for(domain):
        yield key, domain, subtype, seed_pool(key, domain, subtype)


def test_every_seed_requires_only_tools_from_its_own_domain():
  for key, domain, subtype, pool in _all_pools():
    names = {t["name"] for t in DOMAIN_TOOLS[domain]}
    for seed in pool:
      assert set(seed.requires) <= names, (key, domain, subtype, seed.topic)


def test_no_seed_pins_more_tools_than_the_smallest_tool_budget_admits():
  # `sample_tools` returns `pinned + sample(pool, k - len(pinned))`. A seed that
  # pins more tools than `n_tools[0]` overruns the bound the plan asserts on.
  smallest = min(s.n_tools[0] for s in CATEGORIES.values())
  for key, domain, subtype, pool in _all_pools():
    for seed in pool:
      assert len(seed.requires) <= smallest, (key, domain, subtype, seed.topic)


def test_the_traps_and_refusals_that_must_not_be_servable_pin_nothing():
  # Pinning a tool into a `no_relevant_tool` trap gives the request a tool that
  # serves it. Pinning one into a refusal offers a way to comply.
  for domain in CATEGORIES["traps"].domains:
    for seed in seed_pool("traps", domain, "no_relevant_tool"):
      assert seed.requires == (), seed.topic
  for domain in CATEGORIES["refusals"].domains:
    for subtype in CATEGORIES["refusals"].subtypes:
      for seed in seed_pool("refusals", domain, subtype):
        assert seed.requires == (), seed.topic


def test_the_traps_that_need_a_tool_to_fail_on_name_one():
  # `missing_argument` needs a tool whose required argument the user omitted;
  # `tool_error` needs a tool for the failing call to be made against. Neither
  # holds when the tool the seed names is absent from the list.
  for subtype in ("missing_argument", "tool_error"):
    for domain in CATEGORIES["traps"].domains:
      for seed in seed_pool("traps", domain, subtype):
        assert seed.requires, (domain, subtype, seed.topic)


def test_every_optional_args_seed_names_a_tool_with_an_optional_parameter():
  spec = CATEGORIES["single_stage"]
  for domain in spec.domains:
    pool = usable_seeds("single_stage", domain, "optional_args")
    assert pool, domain
    for seed in pool:
      assert any(has_optional_args(ALL_TOOLS[n]) for n in seed.requires), seed.topic


def test_only_optional_args_narrows_its_pool_and_it_narrows_it_strictly():
  for key, domain, subtype, pool in _all_pools():
    narrowed = usable_seeds(key, domain, subtype)
    if subtype == "optional_args":
      assert set(narrowed) < set(pool), (key, domain)
    else:
      # The same tuple object, so `test_trap_and_refusal_seeds_are_keyed_by
      # _subtype` keeps testing pool identity rather than a fresh copy.
      assert narrowed is pool, (key, domain, subtype)
