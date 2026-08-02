"""The generation plan.

`plan_requests` is built in full, from a seeded RNG, before a single teacher
token is spent. That makes it worth asserting on: an incoherent request is one
the teacher cannot satisfy, and it is paid for at full price before the
validator throws it away.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from quick_slm_trainer.config import SFTConfig
from quick_slm_trainer.sft.generate import (
  Request,
  batch_seed,
  describe_plan,
  describe_preflight,
  plan_requests,
  preflight,
)
from quick_slm_trainer.sft.specs import CATEGORIES
from quick_slm_trainer.sft.tools import has_optional_args, required_args, tool_names

CFG = SFTConfig(target_examples=2_000)


@pytest.fixture(scope="module")
def requests():
  return plan_requests(CFG)


def test_the_plan_has_exactly_the_requested_number_of_examples(requests):
  assert len(requests) == CFG.target_examples


def test_the_plan_is_deterministic():
  a = plan_requests(CFG, seed=99)
  b = plan_requests(CFG, seed=99)
  assert [r.id for r in a] == [r.id for r in b]
  assert [r.seed_topic for r in a] == [r.seed_topic for r in b]
  assert [tool_names(r.tools) for r in a] == [tool_names(r.tools) for r in b]


def test_a_different_seed_gives_a_different_plan():
  a = plan_requests(CFG, seed=1)
  b = plan_requests(CFG, seed=2)
  assert [r.seed_topic for r in a] != [r.seed_topic for r in b]


def test_every_request_id_is_unique(requests):
  ids = [r.id for r in requests]
  assert len(set(ids)) == len(ids)


def test_category_shares_match_the_config(requests):
  counts = Counter(r.category for r in requests)
  for key, share in CFG.category_shares.items():
    assert counts[key] == pytest.approx(CFG.target_examples * share, abs=2)


def test_every_request_uses_a_domain_its_category_declares(requests):
  for r in requests:
    assert r.domain in CATEGORIES[r.category].domains, r.id


def test_every_request_uses_a_subtype_valid_for_its_domain(requests):
  for r in requests:
    assert r.subtype in CATEGORIES[r.category].subtypes_for(r.domain), r.id


def test_no_factory_request_asks_for_a_web_search_fallback(requests):
  for r in requests:
    assert not (r.domain == "factory" and r.subtype == "search_fallback"), r.id


def test_search_fallback_always_has_web_search_available(requests):
  for r in requests:
    if r.subtype == "search_fallback":
      assert "web_search" in tool_names(r.tools), r.id


def test_the_no_relevant_tool_trap_withholds_web_search(requests):
  # Left in, a general-purpose search plausibly serves "translate this", which
  # turns the trap into a reasonable call and the example into a rejection.
  for r in requests:
    if r.category == "traps" and r.subtype == "no_relevant_tool" and r.domain == "world":
      assert "web_search" not in tool_names(r.tools), r.id


def test_every_request_offers_the_answer_tool(requests):
  # Every trap and every refusal resolves to `answer`. Without it in the list
  # the model's only options are to call a tool it should not, or invent one.
  for r in requests:
    assert "answer" in tool_names(r.tools), r.id


def test_tool_counts_stay_inside_the_specs_bounds(requests):
  for r in requests:
    lo, hi = CATEGORIES[r.category].n_tools
    n_domain_tools = len(r.tools) - 1 # `answer` is appended, not sampled
    assert lo <= n_domain_tools <= hi, f"{r.id}: {n_domain_tools} not in [{lo}, {hi}]"


def test_no_request_repeats_a_tool(requests):
  for r in requests:
    names = [t["name"] for t in r.tools]
    assert len(names) == len(set(names)), r.id


def test_every_offered_tool_has_a_well_formed_schema(requests):
  for r in requests:
    for tool in r.tools:
      assert set(tool["parameters"]["required"]) <= set(tool["parameters"]["properties"])
      assert required_args(tool) == set(tool["parameters"]["required"])


def test_every_unpaired_request_renders_a_prompt_naming_its_seed_and_tools(requests):
  for r in requests[::97]: # a sparse sweep; rendering all 2000 is slow and pointless
    if r.is_paired:
      continue
    prompt = r.prompt()
    assert r.seed_topic in prompt
    assert r.subtype in prompt
    for tool in r.tools:
      assert f'"{tool["name"]}"' in prompt


def test_every_request_has_a_system_prompt(requests):
  for r in requests[::97]:
    assert r.system().strip()


def test_subtypes_are_evenly_distributed_within_a_category(requests):
  # Round-robin, so no sub-type may be starved. The paired category is absent:
  # its sub-types are properties of the swap specs, which are round-robined at
  # spec granularity rather than sub-type granularity.
  for key in ("single_stage", "refusals"):
    counts = Counter(r.subtype for r in requests if r.category == key)
    assert max(counts.values()) - min(counts.values()) <= 2, (key, counts)


def test_every_domain_and_subtype_combination_a_spec_declares_is_planned(requests):
  # `domain` and `subtype` were both indexed by the loop counter. Two nested
  # round-robins keyed on one index reach only the combinations their lengths'
  # common orbit passes through: `single_stage` planned four of its eight, so
  # `factory` never once saw a `direct` request and `world` never saw
  # `optional_args`; `multi_stage`/`factory` never saw a two- or four-step
  # chain, in the simulation that is the paper's demo environment.
  for key, spec in CATEGORIES.items():
    if spec.paired:
      continue
    got = {(r.domain, r.subtype) for r in requests if r.category == key}
    want = {(d, s) for d in spec.domains for s in spec.subtypes_for(d)}
    assert got == want, (key, "never planned:", sorted(want - got))


def test_every_subtype_is_evenly_distributed_within_each_domain(requests):
  # The share has to hold per domain, not only per category. Summing over
  # domains hides a sub-type that one domain never draws and the other
  # draws twice as often.
  for key, spec in CATEGORIES.items():
    if spec.paired:
      continue
    for domain in spec.domains:
      counts = Counter(r.subtype for r in requests if r.category == key and r.domain == domain)
      assert set(counts) == set(spec.subtypes_for(domain)), (key, domain)
      assert max(counts.values()) - min(counts.values()) <= 1, (key, domain, counts)


# --------------------------------------------------------------------------
# Capping the capacity-bound paired category
# --------------------------------------------------------------------------
def _conflict_pairs(reqs):
  return {r.pair_id for r in reqs if r.category == "state_memory_conflict"}


def test_the_default_config_does_not_cap_and_keeps_the_exact_count():
  # `None` preserves the old contract: the plan is exactly `target_examples`.
  cfg = SFTConfig(target_examples=2_000)
  assert cfg.conflict_max_pairs is None
  assert len(plan_requests(cfg)) == 2_000


def test_a_pair_cap_limits_the_paired_category_and_nothing_else():
  base = SFTConfig(target_examples=4_000)
  capped = SFTConfig(target_examples=4_000, conflict_max_pairs=200)

  r_base, r_capped = plan_requests(base), plan_requests(capped)
  assert len(_conflict_pairs(r_base)) == 400  # 20% of 4,000 = 800 examples = 400 pairs
  assert len(_conflict_pairs(r_capped)) == 200 # capped

  # Unpaired categories are untouched: same ids, same order.
  unpaired = lambda rs: [r.id for r in rs if not r.is_paired]
  assert unpaired(r_base) == unpaired(r_capped)


def test_a_cap_above_the_planned_count_is_a_no_op():
  cfg = SFTConfig(target_examples=4_000, conflict_max_pairs=10_000)
  assert len(_conflict_pairs(plan_requests(cfg))) == 400


def test_sft_config_caps_the_paired_category_at_the_tables_capacity():
  # The whole point of the cap: `sft_config` plans the counterfactual category at
  # what the spec table can distinguish (plus a reject margin), not at 20% of
  # 80,000, which would have generated ~6,700 pairs of pure duplicate.
  from quick_slm_trainer.config import sft_config
  from quick_slm_trainer.sft.conflict import recommended_conflict_pairs

  cfg = sft_config()
  assert cfg.sft.conflict_max_pairs == recommended_conflict_pairs()
  pairs = len(_conflict_pairs(plan_requests(cfg.sft)))
  assert pairs == cfg.sft.conflict_max_pairs
  # Far below the uncapped 8,000, and above the 1,298 distinct so rejects have slack.
  assert 1_298 <= pairs < 8_000


# --------------------------------------------------------------------------
# Capping the seed-bound unpaired categories
# --------------------------------------------------------------------------
def _cells(reqs):
  """planned count per (category, domain, subtype), unpaired only."""
  return Counter((r.category, r.domain, r.subtype) for r in reqs if not r.is_paired)


def test_the_default_config_does_not_cap_per_seed():
  # `None` preserves the old contract, the same way `conflict_max_pairs` does.
  cfg = SFTConfig(target_examples=2_000)
  assert cfg.max_examples_per_seed is None
  assert len(plan_requests(cfg)) == 2_000


def test_a_per_seed_cap_holds_every_cell_to_its_pool_times_the_cap():
  from quick_slm_trainer.sft.prompts import usable_seeds

  cfg = SFTConfig(target_examples=4_000, max_examples_per_seed=2)
  cells = _cells(plan_requests(cfg))
  assert cells, "no unpaired request was planned"
  for (category, domain, subtype), planned in cells.items():
    pool = len(usable_seeds(category, domain, subtype))
    assert planned <= 2 * pool, (category, domain, subtype, planned, pool)


def test_a_per_seed_cap_shrinks_the_plan_and_leaves_the_paired_category_alone():
  base = SFTConfig(target_examples=4_000)
  capped = SFTConfig(target_examples=4_000, max_examples_per_seed=2)

  r_base, r_capped = plan_requests(base), plan_requests(capped)
  assert len(r_capped) < len(r_base)
  # The paired category is capped by `conflict_max_pairs`, not by this.
  assert _conflict_pairs(r_base) == _conflict_pairs(r_capped)


def test_a_capped_plan_is_an_exact_subsequence_of_the_uncapped_one():
  # Ids are keyed off the round-robin index, and each request is built before it
  # is kept or dropped, so a capped cell changes nothing about the requests that
  # survive. Two properties ride on this. A resume matches the shard by id, so a
  # shifted id would re-generate work already on disk under a new name. And
  # raising the cap later must only add requests: if dropping one request
  # re-drew the seeds of every later one, the ids already generated would keep
  # their old content while the plan claimed something else for them.
  base = {r.id: r for r in plan_requests(SFTConfig(target_examples=4_000)) if not r.is_paired}
  capped = [r for r in plan_requests(SFTConfig(target_examples=4_000, max_examples_per_seed=2)) if not r.is_paired]

  assert {r.id for r in capped} < set(base) # a strict subset: some were dropped
  for r in capped:
    assert r == base[r.id], r.id

  # And it really is a subsequence: relative order is preserved, not just membership.
  order = list(base)
  assert [r.id for r in capped] == [i for i in order if i in {c.id for c in capped}]


def test_a_per_seed_cap_larger_than_the_plan_is_a_no_op():
  cfg = SFTConfig(target_examples=2_000, max_examples_per_seed=10_000)
  assert len(plan_requests(cfg)) == 2_000


def test_sft_config_caps_the_unpaired_categories_at_the_recommended_yield():
  # The unpaired twin of the paired cap: `sft_config` plans each cell at what its
  # hand-written seed pool can realise, not at its share of 80,000, which asked
  # for thousands of examples per pool of tens.
  from quick_slm_trainer.config import sft_config
  from quick_slm_trainer.sft.prompts import recommended_examples_per_seed, usable_seeds

  cfg = sft_config()
  assert cfg.sft.max_examples_per_seed == recommended_examples_per_seed()

  planned = plan_requests(cfg.sft)
  cells = _cells(planned)
  for (category, domain, subtype), n in cells.items():
    pool = len(usable_seeds(category, domain, subtype))
    assert n <= cfg.sft.max_examples_per_seed * pool, (category, domain, subtype)

  # The widest cell has 29 seeds, so the cap holds it near 1,160 rather than the
  # ~4,000 its raw share asks for.
  assert max(cells.values()) <= 40 * 29
  uncapped = len(plan_requests(SFTConfig(**{**cfg.sft.to_dict(), "max_examples_per_seed": None})))
  assert len(planned) < uncapped


# --------------------------------------------------------------------------
# Seeds and the tools they presuppose
# --------------------------------------------------------------------------
def test_every_seed_is_offered_the_tools_its_scenario_presupposes(requests):
  # The seed and the tool subset were drawn independently, so the tool a seed
  # named was usually absent: 74% of `missing_argument` and `tool_error` traps,
  # 55% of the single-stage world scenarios. The teacher's ways out of that bind
  # both pass validation and both corrupt the corpus, one by drifting off the
  # seed, the other by writing a `no_relevant_tool` example under the
  # `missing_argument` label.
  for r in requests:
    assert set(r.requires) <= tool_names(r.tools), (r.id, r.seed_topic)


def test_optional_args_always_offers_a_tool_that_has_an_optional_parameter(requests):
  # "Choose a tool with optional parameters and have the user supply one of
  # them" was addressed to a tool list that had no such tool 62% of the time.
  seen = 0
  for r in requests:
    if r.subtype != "optional_args":
      continue
    seen += 1
    assert any(has_optional_args(t) for t in r.tools), (r.id, r.seed_topic)
  assert seen, "no optional_args request was planned"


def test_the_no_relevant_tool_trap_pins_nothing(requests):
  # A pinned tool is a tool the request can use, which is the opposite of what
  # this trap is.
  for r in requests:
    if r.category == "traps" and r.subtype == "no_relevant_tool":
      assert r.requires == (), r.id


def test_every_request_is_generated_under_the_teachers_system_prompt(requests):
  # `system()` returned `DOMAIN_SYSTEM[domain]`, which is what the *student* is
  # conditioned on at inference. Handing it to the teacher tells the teacher it
  # is the tool-calling assistant, and it answers the request rather than
  # authoring an example around it. `generate_batch` never called the method,
  # so every request, paired or not, went out under `TEACHER_SYSTEM`.
  from quick_slm_trainer.sft.prompts import CONFLICT_SYSTEM, TEACHER_SYSTEM
  from quick_slm_trainer.sft.tools import DOMAIN_SYSTEM

  for r in requests[::13]:
    assert r.system() == (CONFLICT_SYSTEM if r.is_paired else TEACHER_SYSTEM), r.id
    assert r.system() not in DOMAIN_SYSTEM.values(), r.id


# --------------------------------------------------------------------------
# Sampling seeds
# --------------------------------------------------------------------------
def _fake_requests(n: int, prefix: str = "single_stage") -> list[Request]:
  return [
    Request(id=f"{prefix}-{i:06d}", category="single_stage", subtype="direct", domain="world")
    for i in range(n)
  ]


def test_a_batch_samples_by_its_contents_and_not_by_its_position():
  # `run_generation` resumes by skipping ids already in the shard, so a resumed
  # run's batches sit at different offsets in the sampling stream than an
  # uninterrupted run's. A seed derived from the batch's own ids does not care.
  batch = _fake_requests(8)
  assert batch_seed(1, batch) == batch_seed(1, list(batch))
  assert batch_seed(1, batch) != batch_seed(2, batch)
  assert batch_seed(1, batch) != batch_seed(1, _fake_requests(8, "traps"))
  assert batch_seed(1, batch[:4]) != batch_seed(1, batch[4:])


def test_the_batch_seed_survives_a_new_process():
  # A golden value. `hash()` is salted per interpreter, so a seed derived from
  # it would differ between a run and its own resume.
  assert batch_seed(20240201, _fake_requests(4)) == 1_199_924_749


def test_generation_is_seeded_from_the_same_constant_as_the_plan():
  from quick_slm_trainer.sft import generate as gen

  assert gen.PLAN_SEED == 20240201
  assert plan_requests(CFG) == plan_requests(CFG, seed=gen.PLAN_SEED)


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
def _reply(preamble: str = "", think: str = "The user wants a direct answer here.") -> str:
  return preamble + json.dumps(
    {
      "user": "What is the weather in Oslo?",
      "state": None,
      "memory": None,
      "turns": [
        {
          "role": "assistant",
          "think": think,
          "calls": [{"name": "answer", "arguments": {"text": "It is cold."}}],
        }
      ],
    }
  )


def _replies(*texts):
  def fake_generate(model, tok, cfg, batch, *, seed=None):
    assert seed is not None, "generation must be seeded"
    return list(texts)[: len(batch)]

  return fake_generate


def test_preflight_accepts_a_clean_reply_and_flags_one_that_opens_with_prose(requests):
  singles = [r for r in requests if r.category == "single_stage"][:2]
  results = preflight(
    None, None, CFG, singles, n=2, generate=_replies(_reply(), _reply("Here is the example:\n"))
  )
  assert [r.usable for r in results] == [True, True]
  assert results[0].preamble == ""
  assert results[1].preamble.strip() == "Here is the example:"
  assert "1/2 replies emit text before their first '{'" in describe_preflight(results)


def test_preflight_names_the_reason_a_reply_is_unusable(requests):
  singles = [r for r in requests if r.category == "single_stage"][:2]
  results = preflight(
    None, None, CFG, singles, n=2, generate=_replies("no json anywhere", _reply(think="short"))
  )
  assert [r.parsed for r in results] == [False, True]
  assert [r.reject for r in results] == ["not_json", "empty_think"]
  assert "0 usable" in describe_preflight(results)


def test_preflight_samples_every_category(requests):
  sampled = preflight(None, None, CFG, requests, n=5, generate=_replies(*[_reply()] * 5))
  assert len({r.category for r in sampled}) == len(CATEGORIES)


# --------------------------------------------------------------------------
# The paired category
# --------------------------------------------------------------------------
def _conflict(requests):
  return [r for r in requests if r.category == "state_memory_conflict"]


def test_the_paired_category_is_planned_as_two_branches_per_pair(requests):
  branches: dict[str, list[str]] = {}
  for r in _conflict(requests):
    branches.setdefault(r.pair_id, []).append(r.branch)
  assert branches, "no conflict requests were planned"
  assert all(sorted(v) == ["a", "b"] for v in branches.values())
  assert len(_conflict(requests)) == 2 * len(branches)


def test_paired_requests_carry_their_inputs_and_never_the_oracle(requests):
  # The shard stores inputs; `conflict.py` owns the ground truth and recomputes
  # it at validation. A corrected oracle then costs a revalidation pass rather
  # than another generation run.
  for r in _conflict(requests):
    assert r.state and r.memory and r.request and r.spec_id
    assert not hasattr(r, "oracle")
    assert r.seed_topic == ""


def test_only_training_specs_are_ever_planned(requests):
  from quick_slm_trainer.sft.conflict_specs import EVAL_SPECS, TRAIN_SPECS

  planned = {r.spec_id for r in _conflict(requests)}
  assert planned <= {s.id for s in TRAIN_SPECS}
  assert not planned & {s.id for s in EVAL_SPECS}


def test_the_two_branches_of_a_pair_share_everything_but_one_state_field(requests):
  from quick_slm_trainer.sft.conflict_specs import ALL_SPECS
  from quick_slm_trainer.sft.pointer import diff_paths
  from quick_slm_trainer.template import dumps_compact

  by_pair: dict[str, dict[str, object]] = {}
  for r in _conflict(requests):
    by_pair.setdefault(r.pair_id, {})[r.branch] = r

  for pair_id, branches in list(by_pair.items())[:60]:
    a, b = branches["a"], branches["b"]
    spec = ALL_SPECS[a.spec_id]
    assert diff_paths(a.state, b.state) == {spec.decisive_path}, pair_id
    assert dumps_compact(a.memory) == dumps_compact(b.memory), pair_id
    assert dumps_compact(a.request) == dumps_compact(b.request), pair_id
    assert [t["name"] for t in a.tools] == [t["name"] for t in b.tools], pair_id


def test_the_prompts_of_a_pair_differ_only_inside_the_state_block(requests):
  import difflib

  by_pair: dict[str, dict[str, object]] = {}
  for r in _conflict(requests):
    by_pair.setdefault(r.pair_id, {})[r.branch] = r

  for pair_id, branches in list(by_pair.items())[:12]:
    a, b = branches["a"].prompt(), branches["b"].prompt()
    changed = [
      line
      for line in difflib.unified_diff(a.split("\n"), b.split("\n"), lineterm="")
      if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 2, (pair_id, changed)
    assert all("<state>" in line for line in changed), (pair_id, changed)


def test_a_paired_prompt_never_reveals_the_answer(requests):
  for r in _conflict(requests)[:20]:
    lowered = r.prompt().lower()
    assert "oracle" not in lowered
    assert "correct call" not in lowered


def test_describe_plan_is_printable(requests):
  text = describe_plan(requests)
  assert f"{len(requests):,} requests planned" in text
  for key in CATEGORIES:
    assert key in text
