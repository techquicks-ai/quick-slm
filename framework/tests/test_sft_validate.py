"""The validation filter.

Every check rejects rather than repairs, and every reject reason gets a test,
because the reject histogram is quoted in the paper's data section and a filter
that never fires is indistinguishable from one that always passes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from quick_slm_trainer.sft import validate as V
from quick_slm_trainer.sft.specs import CATEGORIES
from quick_slm_trainer.sft.tools import ALL_TOOLS, ANSWER
from quick_slm_trainer.sft.validate import (
  ValidationStats,
  extract_json,
  record_to_example,
  validate_and_convert,
  validate_record,
)

SINGLE = CATEGORIES["single_stage"]
MULTI = CATEGORIES["multi_stage"]
CONFLICT = CATEGORIES["state_memory_conflict"]
TRAPS = CATEGORIES["traps"]

#: `validate_record` refuses paired specs, and `state_memory_conflict` is the
#: only category that asks for `<state>` and `<memory>`. This unpaired clone of
#: it is what keeps the two ambient-context filters under test.
STATEFUL = dataclasses.replace(CONFLICT, key="stateful_probe", paired=False)

WORLD = [ALL_TOOLS["get_weather"], ALL_TOOLS["get_capital_city"], ANSWER]
FACTORY = [ALL_TOOLS["inspect"], ALL_TOOLS["resume_building"], ANSWER]


def good_single() -> dict:
  return {
    "user": "What's the weather in Tokyo?",
    "state": None,
    "memory": None,
    "turns": [
      {
        "role": "assistant",
        "think": "The user wants weather for Tokyo. get_weather takes a city.",
        "calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
      }
    ],
  }


def good_multi() -> dict:
  return {
    "user": "Weather in the capital of France?",
    "state": None,
    "memory": None,
    "turns": [
      {"role": "assistant", "think": "Need the capital before the weather.",
       "calls": [{"name": "get_capital_city", "arguments": {"country": "France"}}]},
      {"role": "tool", "result": {"capital": "Paris"}},
      {"role": "assistant", "think": "Paris came back. Now the weather there.",
       "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
      {"role": "tool", "result": {"temp_c": 14}},
      {"role": "assistant", "think": "Everything needed is present. Summarise it.",
       "calls": [{"name": "answer", "arguments": {"text": "14 C in Paris."}}]},
    ],
  }


def good_conflict() -> dict:
  return {
    "user": "Is the smelter producing?",
    "state": {"buildings": [{"id": "b_004", "paused": False}]},
    "memory": {"recent": ["user asked to pause b_004"], "last_results": ["pause_building OK"]},
    "turns": [
      {
        "role": "assistant",
        "think": "My memory says I paused b_004, but the state sampled now shows "
             "paused=false. State is ground truth, so it must have been resumed.",
        "calls": [{"name": "answer", "arguments": {"text": "It is running."}}],
      }
    ],
  }


def good_trap() -> dict:
  return {
    "user": "What's the weather tomorrow?",
    "state": None,
    "memory": None,
    "turns": [
      {"role": "assistant", "think": "get_weather needs a city and none was given. Ask.",
       "calls": [{"name": "answer", "arguments": {"text": "Which city?"}}]},
    ],
  }


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------
def test_the_good_records_pass():
  assert validate_record(good_single(), SINGLE, WORLD) is None
  assert validate_record(good_multi(), MULTI, WORLD) is None
  assert validate_record(good_conflict(), STATEFUL, FACTORY) is None
  assert validate_record(good_trap(), TRAPS, WORLD) is None


# --------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------
def test_extract_json_recovers_from_a_markdown_fence():
  rec = good_single()
  text = f"Here is the example:\n```json\n{json.dumps(rec)}\n```\nHope that helps!"
  assert extract_json(text) == rec


def test_extract_json_is_not_fooled_by_a_brace_inside_a_string():
  text = '{"user": "what about } this", "turns": []}'
  assert extract_json(text) == {"user": "what about } this", "turns": []}


def test_extract_json_survives_an_escaped_quote():
  text = r'{"user": "he said \"hi\"", "turns": []}'
  assert extract_json(text)["user"] == 'he said "hi"'


@pytest.mark.parametrize("text", ["", "no json here", '{"unbalanced": ', "[1, 2, 3]"])
def test_extract_json_returns_none_rather_than_raising(text):
  assert extract_json(text) is None


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
def test_empty_user_is_rejected():
  rec = good_single() | {"user": "  "}
  assert validate_record(rec, SINGLE, WORLD) == V.EMPTY_USER


def test_a_turn_list_starting_with_a_tool_turn_is_rejected():
  rec = good_single()
  rec["turns"].insert(0, {"role": "tool", "result": {}})
  assert validate_record(rec, SINGLE, WORLD) == V.BAD_TURN_ORDER


def test_a_turn_list_ending_on_a_tool_turn_is_rejected():
  rec = good_multi()
  rec["turns"].append({"role": "tool", "result": {}})
  assert validate_record(rec, MULTI, WORLD) == V.BAD_TURN_ORDER


def test_consecutive_assistant_turns_are_rejected():
  rec = good_multi()
  del rec["turns"][1] # drop the tool turn between two assistant turns
  assert validate_record(rec, MULTI, WORLD) == V.BAD_TURN_ORDER


def test_too_many_assistant_turns_for_the_category_is_rejected():
  rec = good_multi() # three assistant turns
  assert validate_record(rec, SINGLE, WORLD) == V.TURN_COUNT_OUT_OF_BOUNDS


def test_too_few_assistant_turns_for_the_category_is_rejected():
  assert validate_record(good_single(), MULTI, WORLD) == V.TURN_COUNT_OUT_OF_BOUNDS


# --------------------------------------------------------------------------
# think
# --------------------------------------------------------------------------
def test_empty_think_is_rejected():
  rec = good_single()
  rec["turns"][0]["think"] = "ok"
  assert validate_record(rec, SINGLE, WORLD) == V.EMPTY_THINK


def test_a_think_block_that_leaked_the_template_is_rejected():
  rec = good_single()
  rec["turns"][0]["think"] = "I will emit <response>[{...}]</response> for this request."
  assert validate_record(rec, SINGLE, WORLD) == V.THINK_LEAKED_MARKUP


def test_the_paired_category_refuses_a_per_sample_verdict():
  # Judging one branch in isolation re-admits exactly the sample the pair was
  # built to exclude: a call the teacher got right by pattern-matching the user
  # turn is, on its own, indistinguishable from one it got right by reading
  # state. `conflict.validate_pair` is the only way in.
  with pytest.raises(ValueError, match="paired"):
    validate_record(good_conflict(), CONFLICT, FACTORY)


def test_bare_state_and_memory_words_do_not_trip_the_markup_filter():
  # The markup filter looks for `<think>`-style tags, not for the concepts. A
  # reasoning block that discusses state and memory by name is ordinary prose.
  from quick_slm_trainer.sft.validate import check_think

  assert check_think("The state says paused=false while memory claims I paused it.") is None
  assert check_think("I will emit <response>[]</response>") == V.THINK_LEAKED_MARKUP


# --------------------------------------------------------------------------
# Calls
# --------------------------------------------------------------------------
def test_a_hallucinated_tool_is_rejected():
  rec = good_single()
  rec["turns"][0]["calls"] = [{"name": "book_flight", "arguments": {"to": "Tokyo"}}]
  assert validate_record(rec, SINGLE, WORLD) == V.UNDEFINED_TOOL


def test_a_missing_required_argument_is_rejected():
  rec = good_single()
  rec["turns"][0]["calls"] = [{"name": "get_weather", "arguments": {}}]
  assert validate_record(rec, SINGLE, WORLD) == V.MISSING_REQUIRED_ARG


def test_an_undeclared_argument_is_rejected():
  rec = good_single()
  rec["turns"][0]["calls"][0]["arguments"]["when"] = "tomorrow"
  assert validate_record(rec, SINGLE, WORLD) == V.UNKNOWN_ARG


def test_an_optional_argument_is_allowed():
  rec = good_single()
  rec["turns"][0]["calls"][0]["arguments"]["units"] = "fahrenheit"
  assert validate_record(rec, SINGLE, WORLD) is None


def test_a_wrongly_typed_argument_is_rejected():
  rec = good_single()
  rec["turns"][0]["calls"][0]["arguments"]["city"] = 42
  assert validate_record(rec, SINGLE, WORLD) == V.BAD_ARG_TYPE


def test_a_boolean_does_not_pass_as_an_integer():
  # `bool` subclasses `int` in Python. Without an explicit check, `True` binds
  # as a quantity of 1 to every tool that takes a count.
  tools = [ALL_TOOLS["buy"], ANSWER]
  rec = good_single()
  rec["turns"][0]["calls"] = [{"name": "buy", "arguments": {"resource_id": "iron", "quantity": True}}]
  assert validate_record(rec, SINGLE, tools) == V.BAD_ARG_TYPE


def test_an_integer_passes_as_a_number():
  tools = [ALL_TOOLS["convert_currency"], ANSWER]
  rec = good_single()
  rec["turns"][0]["calls"] = [
    {"name": "convert_currency",
     "arguments": {"amount": 20, "from_currency": "USD", "to_currency": "EUR"}}
  ]
  assert validate_record(rec, SINGLE, tools) is None


def test_a_tool_with_no_required_arguments_may_be_called_bare():
  tools = [ALL_TOOLS["get_inventory"], ANSWER]
  rec = good_single()
  rec["turns"][0]["calls"] = [{"name": "get_inventory", "arguments": {}}]
  assert validate_record(rec, SINGLE, tools) is None


def test_an_assistant_turn_with_no_calls_is_rejected():
  rec = good_single()
  rec["turns"][0]["calls"] = []
  assert validate_record(rec, SINGLE, WORLD) == V.NO_CALLS


# --------------------------------------------------------------------------
# Category-specific
# --------------------------------------------------------------------------
def test_a_trap_that_calls_a_real_tool_is_rejected():
  # The teacher taking the easy way out. `docs/sft_readme.md` budgets ~4%.
  rec = good_trap()
  rec["turns"][0]["calls"] = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
  assert validate_record(rec, TRAPS, WORLD) == V.FINAL_CALL_NOT_ANSWER


def test_a_trap_may_call_a_real_tool_before_it_fails():
  # The tool_error sub-type: call, receive an error, then answer.
  rec = {
    "user": "Weather in Atlantis?",
    "state": None,
    "memory": None,
    "turns": [
      {"role": "assistant", "think": "Atlantis is a city name. Try get_weather.",
       "calls": [{"name": "get_weather", "arguments": {"city": "Atlantis"}}]},
      {"role": "tool", "result": {"error": "city not found"}},
      {"role": "assistant", "think": "The lookup failed. Do not retry it unchanged.",
       "calls": [{"name": "answer", "arguments": {"text": "I could not find Atlantis."}}]},
    ],
  }
  assert validate_record(rec, TRAPS, WORLD) is None


def test_a_trap_answering_alongside_a_real_tool_is_rejected():
  rec = good_trap()
  rec["turns"][0]["calls"] = [
    {"name": "answer", "arguments": {"text": "Which city?"}},
    {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  ]
  assert validate_record(rec, TRAPS, WORLD) == V.FINAL_CALL_NOT_ANSWER


def test_missing_state_on_a_stateful_category_is_rejected():
  rec = good_conflict() | {"state": None}
  assert validate_record(rec, STATEFUL, FACTORY) == V.STATE_MISSING


def test_missing_memory_on_a_stateful_category_is_rejected():
  rec = good_conflict() | {"memory": {"recent": [], "last_results": []}}
  assert validate_record(rec, STATEFUL, FACTORY) == V.MEMORY_MISSING


def test_unexpected_state_on_a_stateless_category_is_rejected():
  rec = good_single() | {"state": {"anything": 1}}
  assert validate_record(rec, SINGLE, WORLD) == V.STATE_UNEXPECTED


def test_unexpected_memory_on_a_stateless_category_is_rejected():
  rec = good_single() | {"memory": {"recent": ["something"]}}
  assert validate_record(rec, SINGLE, WORLD) == V.MEMORY_UNEXPECTED


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def test_record_to_example_preserves_the_turn_structure():
  ex = record_to_example(good_multi(), MULTI, WORLD, system="sys", subtype="chain_2")
  kinds = [type(t).__name__ for t in ex.turns]
  assert kinds == ["UserTurn", "AssistantTurn", "ToolTurn", "AssistantTurn", "ToolTurn", "AssistantTurn"]
  assert ex.category == "multi_stage"
  assert ex.meta["subtype"] == "chain_2"


def test_record_to_example_renders_memory_into_the_wire_format():
  ex = record_to_example(good_conflict(), STATEFUL, FACTORY, system="sys")
  assert "<recent>" in ex.memory and "<last_results>" in ex.memory
  assert ex.state == {"buildings": [{"id": "b_004", "paused": False}]}


def test_record_to_example_leaves_memory_none_when_absent():
  ex = record_to_example(good_single(), SINGLE, WORLD, system="sys")
  assert ex.memory is None and ex.state is None


def test_validate_and_convert_returns_none_and_tallies_on_reject():
  stats = ValidationStats()
  rec = good_single()
  rec["turns"][0]["calls"] = [{"name": "nope", "arguments": {}}]
  assert validate_and_convert(rec, SINGLE, WORLD, system="s", stats=stats) is None
  assert stats.rejected[V.UNDEFINED_TOOL] == 1
  assert stats.accepted == 0


def test_validate_and_convert_tallies_on_accept():
  stats = ValidationStats()
  assert validate_and_convert(good_single(), SINGLE, WORLD, system="s", stats=stats) is not None
  assert stats.accepted == 1
  assert stats.to_dict()["reject_rate"] == 0.0


def test_stats_report_is_printable_when_empty():
  assert "seen 0" in ValidationStats().report()


def test_the_tags_the_paired_prompt_tells_the_teacher_to_read_are_not_leakage():
  # `prompts.py` rule 4 instructs the conflict teacher: "Read the values in
  # <state> before choosing. <state> was sampled this turn and outranks
  # <memory> wherever the two disagree." A teacher that then reasons about the
  # <state> block by name is obeying that instruction, not writing the
  # envelope. Rejecting it cost 481 records on the run, and because the
  # paired category is accepted or rejected as a pair, each one killed a
  # second, blameless branch with it.
  from quick_slm_trainer.sft.validate import check_think

  assert check_think("The <state> block shows b4 paused, so it must be resumed.") is None
  assert check_think("<memory> claims it is running but <state> outranks it here.") is None

  # The envelope's own tags still are leakage: they cannot appear unless the
  # teacher stopped writing content and started writing the template.
  for leaked in ("I will emit <response>[]</response>",
          "Next I output <think> and then the call",
          "Wrap it as <tool>get_weather</tool> for the user",
          "Then <|im_start|>assistant continues the turn"):
    assert check_think(leaked) == V.THINK_LEAKED_MARKUP, leaked
