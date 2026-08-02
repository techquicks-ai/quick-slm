from __future__ import annotations

import pytest

from quick_slm_trainer.sft import grade as G
from quick_slm_trainer.template import AssistantTurn, Example, UserTurn

TOOLS = [
  {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
    },
  },
  {
    "name": "answer",
    "description": "Reply to the user.",
    "parameters": {
      "type": "object",
      "properties": {"text": {"type": "string"}},
      "required": ["text"],
    },
  },
]


def gen(think: str, calls_json: str) -> str:
  return f"<think>\n{think}\n</think>\n<response>{calls_json}</response>"


def example(calls: list[dict], *, category: str = "single_stage", state=None, meta=None) -> Example:
  return Example(
    tools=TOOLS,
    turns=[UserTurn("what is the weather in Tokyo?"), AssistantTurn(think="because", calls=calls)],
    state=state,
    category=category,
    meta=meta or {},
  )


WEATHER_TOKYO = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
WEATHER_OSAKA = [{"name": "get_weather", "arguments": {"city": "Osaka"}}]
THINK = "The user wants the weather, so I will call get_weather."


# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------
def test_a_well_formed_generation_has_no_fault():
  text = gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]')
  assert G.grade_format(text, TOOLS) is None


@pytest.mark.parametrize(
  "text, fault",
  [
    ('<response>[{"name":"get_weather","arguments":{"city":"T"}}]</response>', G.NO_THINK),
    (gen("no", '[{"name":"get_weather","arguments":{"city":"T"}}]'), G.SHORT_THINK),
    (f"<think>\n{THINK}\n</think>", G.NO_RESPONSE),
    (gen(THINK, "[]"), G.EMPTY_CALLS),
    (gen(THINK, '[{"name":"book_flight","arguments":{}}]'), "undefined_tool"),
    (gen(THINK, '[{"name":"get_weather","arguments":{}}]'), "missing_required_arg"),
    (gen(THINK, '[{"name":"get_weather","arguments":{"city":7}}]'), "bad_arg_type"),
    (gen(THINK, '[{"name":"get_weather","arguments":{"city":"T","when":"x"}}]'), "unknown_arg"),
  ],
)
def test_each_fault_is_reported_distinctly(text, fault):
  # Collapsing these into one "malformed" bucket would hide whether a model has
  # learnt the envelope but not the schema, which is the interesting middle state.
  assert G.grade_format(text, TOOLS) == fault


def test_a_malformed_generation_is_never_scored_correct():
  ex = example(WEATHER_TOKYO)
  r = G.grade(ex, gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"},}]'))
  assert not r.well_formed and not r.correct


# --------------------------------------------------------------------------
# Call correctness
# --------------------------------------------------------------------------
def test_argument_values_are_compared_not_only_the_tool_name():
  # The trap category's entire failure mode is the right tool with a guessed
  # argument. A name-only comparison would score it correct.
  ex = example(WEATHER_TOKYO)
  assert G.grade(ex, gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]')).correct
  assert not G.grade(ex, gen(THINK, '[{"name":"get_weather","arguments":{"city":"Japan"}}]')).correct


def test_refusal_wording_is_not_compared():
  # `oracles.FREE_TEXT_ARGS` excludes `answer.text`: no string comparison tells
  # a good decline from a bad one, so the row measures call choice only.
  ex = example([{"name": "answer", "arguments": {"text": "I cannot help with that."}}])
  text = gen(THINK, '[{"name":"answer","arguments":{"text":"Sorry, that is out of scope."}}]')
  assert G.grade(ex, text).correct


def test_only_the_first_call_is_graded():
  # Later turns are conditioned on tool results the model never received.
  ex = Example(
    tools=TOOLS,
    turns=[
      UserTurn("weather in Tokyo?"),
      AssistantTurn(think="first", calls=WEATHER_TOKYO),
      AssistantTurn(think="second", calls=WEATHER_OSAKA),
    ],
    category="multi_stage",
  )
  assert G.reference_calls(ex) == WEATHER_TOKYO


# --------------------------------------------------------------------------
# The pair metric
# --------------------------------------------------------------------------
def _pair(pair_id: str, a_call: list[dict], b_call: list[dict]) -> list[Example]:
  return [
    # `meta` mirrors what `conflict.py` actually writes: `group` is the key
    # `template.group_of` reads, and it carries the pair id.
    example(a_call, category="state_memory_conflict", state={"x": 1},
        meta={"pair_id": pair_id, "branch": "a", "group": pair_id}),
    example(b_call, category="state_memory_conflict", state={"x": 2},
        meta={"pair_id": pair_id, "branch": "b", "group": pair_id}),
  ]


def test_a_state_blind_model_scores_exactly_one_on_every_pair():
  # THE test for this module. Validation guarantees a pair's two branches have
  # different oracle calls, so a model emitting one call for both scores
  # exactly one -- whatever call it picks. That is not chance and not partial
  # credit; it is the signature of a model that never read <state>. Per-branch
  # accuracy would report this model at 50% and read as encouraging.
  examples = _pair("p1", WEATHER_TOKYO, WEATHER_OSAKA) + _pair("p2", WEATHER_OSAKA, WEATHER_TOKYO)
  blind = gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]')
  results = [G.grade(ex, blind) for ex in examples]

  per_branch = sum(r.correct for r in results) / len(results)
  assert per_branch == 0.5, "per-branch accuracy flatters a state-blind model"

  pairs = G.grade_pairs(results)
  assert pairs["pairs"] == 2
  assert pairs[G.ONE] == 2
  assert pairs[G.BOTH] == 0 and pairs["grounded_rate"] == 0.0


def test_a_grounded_model_scores_both():
  examples = _pair("p1", WEATHER_TOKYO, WEATHER_OSAKA)
  results = [
    G.grade(examples[0], gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]')),
    G.grade(examples[1], gen(THINK, '[{"name":"get_weather","arguments":{"city":"Osaka"}}]')),
  ]
  pairs = G.grade_pairs(results)
  assert pairs[G.BOTH] == 1 and pairs["grounded_rate"] == 1.0


def test_a_pair_missing_a_branch_is_dropped_not_counted_as_half():
  examples = _pair("p1", WEATHER_TOKYO, WEATHER_OSAKA)
  results = [G.grade(examples[0], gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]'))]
  pairs = G.grade_pairs(results)
  assert pairs["pairs"] == 0 and pairs["dropped_incomplete"] == 1


def test_unpaired_examples_do_not_enter_the_pair_metric():
  results = [G.grade(example(WEATHER_TOKYO), gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]'))]
  assert G.grade_pairs(results)["pairs"] == 0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def test_summarise_splits_by_category_and_counts_faults():
  good = gen(THINK, '[{"name":"get_weather","arguments":{"city":"Tokyo"}}]')
  results = [
    G.grade(example(WEATHER_TOKYO, category="single_stage"), good),
    G.grade(example(WEATHER_OSAKA, category="single_stage"), good),     # wrong city
    G.grade(example(WEATHER_TOKYO, category="traps"), gen(THINK, "not json")),
  ]
  rep = G.summarise(results)
  assert rep.by_category["single_stage"] == {
    "n": 2, "well_formed": 2, "correct": 1,
    "well_formed_rate": 1.0, "correct_rate": 0.5,
  }
  assert rep.by_category["traps"]["well_formed"] == 0
  assert rep.faults[G.NO_RESPONSE] == 1
  assert rep.overall["n"] == 3 and rep.overall["correct"] == 1
  assert "single_stage" in rep.table()
