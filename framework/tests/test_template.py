"""The template and the loss mask.

These are the highest-value tests in the suite. A wrong mask does not crash, does
not spike the loss, and does not show up until the model is graded on reproducing
the system prompt instead of the assistant turn.
"""

from __future__ import annotations

import json

import pytest

from quick_slm_trainer.template import (
  IGNORE_INDEX,
  RESPONSE_CLOSE,
  RESPONSE_OPEN,
  THINK_OPEN,
  AssistantTurn,
  Example,
  ToolTurn,
  UserTurn,
  encode,
  format_memory,
  parse_calls,
  parse_think,
  render,
  render_pretrain_example,
  render_pretrain_prompt,
  render_prompt,
)

CALLS = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]


def simple() -> Example:
  return Example(
    tools=TOOLS,
    turns=[UserTurn("What's the weather in Tokyo?"), AssistantTurn("User wants Tokyo.", CALLS)],
  )


def multi_turn() -> Example:
  return Example(
    tools=TOOLS,
    turns=[
      UserTurn("Weather in the capital of France?"),
      AssistantTurn("Need the capital first.", [{"name": "get_capital_city", "arguments": {"country": "France"}}]),
      ToolTurn({"capital": "Paris"}),
      AssistantTurn("Got Paris.", CALLS),
    ],
    state={"tick": 4},
    memory=format_memory(recent=["user asked about France"], last_results=["ok"]),
  )


# --------------------------------------------------------------------------
# Pretraining template
# --------------------------------------------------------------------------
def test_pretrain_prompt_keeps_its_trailing_space():
  # The corpus was written with a space after `calls:` and the model's first
  # generated token follows it. Stripping it silently shifts every probe.
  prompt = render_pretrain_prompt("hi", TOOLS)
  assert prompt.endswith("Valid function calls: ")


def test_pretrain_passes_prestringified_json_through_untouched():
  # xLAM stores tools and answers as JSON strings. Reserialising them would
  # renormalise whitespace and change the corpus.
  raw = '[{"name": "x", "arguments": {}}]'
  out = render_pretrain_example("q", raw, raw)
  assert out.count(raw) == 2


def test_pretrain_example_extends_the_prompt():
  prompt = render_pretrain_prompt("q", TOOLS)
  assert render_pretrain_example("q", TOOLS, CALLS).startswith(prompt)


# --------------------------------------------------------------------------
# ChatML template
# --------------------------------------------------------------------------
def test_example_without_assistant_turn_is_refused():
  with pytest.raises(ValueError, match="no loss"):
    Example(tools=TOOLS, turns=[UserTurn("hi")])


def test_example_must_start_with_a_user_turn():
  with pytest.raises(ValueError, match="first turn"):
    Example(tools=TOOLS, turns=[AssistantTurn("t", CALLS)])


def test_state_and_memory_sit_outside_the_chatml_turns():
  text = render(multi_turn())
  state_at = text.index("<state>")
  first_user_at = text.index("<|im_start|>user")
  assert state_at < first_user_at


def test_render_prompt_stops_before_the_first_think():
  prompt = render_prompt(simple())
  assert THINK_OPEN not in prompt
  assert prompt.endswith("<|im_start|>assistant\n")
  assert render(simple()).startswith(prompt)


# --------------------------------------------------------------------------
# The loss mask
# --------------------------------------------------------------------------
def test_encode_lengths_agree(tok):
  ids, labels = encode(simple(), tok)
  assert len(ids) == len(labels)


def test_encode_scores_only_the_assistant_turn(tok):
  ex = simple()
  ids, labels = encode(ex, tok)

  scored = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
  assert scored, "an example with an assistant turn must score something"

  # Scored labels mirror their ids exactly; unscored ones are the ignore index.
  for i, label in enumerate(labels):
    assert label == ids[i] or label == IGNORE_INDEX

  # The scored span is contiguous and begins at `<think>`.
  assert scored == list(range(scored[0], scored[-1] + 1))
  think_id = tok(THINK_OPEN)["input_ids"][0]
  assert ids[scored[0]] == think_id


def test_encode_scores_every_assistant_turn(tok):
  ids, labels = encode(multi_turn(), tok)
  think_id = tok(THINK_OPEN)["input_ids"][0]
  starts = [i for i, v in enumerate(ids) if v == think_id and labels[i] != IGNORE_INDEX]
  assert len(starts) == 2


def test_encode_never_scores_the_user_or_tool_turns(tok):
  ex = multi_turn()
  ids, labels = encode(ex, tok)

  # Every id belonging to a tool result must be ignored. The result is
  # server-supplied at inference time; grading it is pure waste.
  tool_ids = tok('{"capital":"Paris"}')["input_ids"]
  for i in range(len(ids) - len(tool_ids)):
    if ids[i : i + len(tool_ids)] == tool_ids:
      assert all(labels[j] == IGNORE_INDEX for j in range(i, i + len(tool_ids)))


def test_segments_concatenate_to_the_full_render(tok):
  # `encode` splits on segment boundaries and encodes each independently. That
  # is only sound because boundaries fall next to added special tokens.
  ex = multi_turn()
  ids, _ = encode(ex, tok)
  assert ids == tok(render(ex))["input_ids"]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def test_parse_calls_reads_back_what_render_wrote():
  assert parse_calls(render(simple())) == CALLS


def test_parse_calls_promotes_a_bare_object_to_a_list():
  text = f'{RESPONSE_OPEN}{json.dumps(CALLS[0])}{RESPONSE_CLOSE}'
  assert parse_calls(text) == CALLS


@pytest.mark.parametrize(
  "text",
  [
    "",
    "<response>[{'name': 'x'}]</response>", # single quotes are not JSON
    '<response>[{"name":"x"}</response>', # unbalanced bracket, the top reject bucket
    '<response>["not_an_object"]</response>',
    '[{"name":"x","arguments":{}}]', # no response tags at all
  ],
)
def test_parse_calls_returns_none_rather_than_raising(text):
  assert parse_calls(text) is None


def test_parse_think():
  assert parse_think(render(simple())) == "User wants Tokyo."
  assert parse_think("no tags here") is None


def test_format_memory_omits_empty_sections():
  assert format_memory() == ""
  assert "<last_results>" not in format_memory(recent=["a"])
  assert "<recent>" not in format_memory(last_results=["b"])
