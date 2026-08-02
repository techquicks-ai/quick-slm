"""DPO preference tooling: record round-trips, surface-form fidelity, generators.

No GPU. The teacher path is exercised through the same `generate` seam
`sft.generate.preflight` uses, so the parsing and rendering are tested without
loading Gemma.
"""

from __future__ import annotations

import json

import pytest

from quick_slm_trainer.dpo import (
  Candidate,
  MockGenerator,
  PreferenceRecord,
  PreferenceStore,
  Scenario,
  build_record,
)
from quick_slm_trainer.dpo.generators import TeacherGenerator
from quick_slm_trainer.config import SFTConfig
from quick_slm_trainer.sft.tools import DOMAIN_SYSTEM, sample_tools
from quick_slm_trainer.template import (
  AssistantTurn,
  Example,
  UserTurn,
  encode,
  parse_calls,
  parse_think,
  render,
  render_prompt,
)


def _weather_scenario() -> Scenario:
  import random

  tools = sample_tools(random.Random(0), "world", 4, include=["get_weather"])
  return Scenario(
    category="single_stage",
    subtype="direct",
    domain="world",
    system=DOMAIN_SYSTEM["world"],
    tools=tools,
    user="What's the weather in Istanbul?",
    candidates=[
      Candidate("Call the weather tool.", [{"name": "get_weather", "arguments": {"city": "Istanbul"}}], "good"),
      Candidate("Guess.", [{"name": "answer", "arguments": {"text": "It's sunny."}}], "bad"),
    ],
  )


# --------------------------------------------------------------------------
# Surface form: prompt + completion must equal what the student trains on
# --------------------------------------------------------------------------
def test_prompt_plus_completion_reconstructs_the_full_render():
  scn = _weather_scenario()
  for cand in scn.candidates:
    ex = Example(
      tools=scn.tools,
      turns=[UserTurn(scn.user), AssistantTurn(cand.think, cand.calls)],
      state=scn.state,
      memory=scn.memory,
      system=scn.system,
      category=scn.category,
      meta={"subtype": scn.subtype, "domain": scn.domain},
    )
    assert scn.prompt() == render_prompt(ex)
    assert scn.prompt() + scn.completion(cand) == render(ex)


def test_completion_is_exactly_the_scored_span(tok):
  """The completion bytes are the ones the loss mask covers, no more, no less."""
  scn = _weather_scenario()
  cand = scn.candidates[0]
  ex = Example(
    tools=scn.tools,
    turns=[UserTurn(scn.user), AssistantTurn(cand.think, cand.calls)],
    system=scn.system,
  )
  ids, labels = encode(ex, tok)
  scored = [i for i, lab in zip(ids, labels) if lab != -100]
  completion_ids = tok(scn.completion(cand), add_special_tokens=False)["input_ids"]
  assert scored == completion_ids


def test_completion_parses_back_to_its_calls_and_think():
  scn = _weather_scenario()
  cand = scn.candidates[0]
  body = scn.completion(cand)
  assert parse_calls(body) == cand.calls
  assert parse_think(body) == cand.think


def test_the_two_candidates_share_one_prompt():
  scn = _weather_scenario()
  a, b = scn.candidates
  assert scn.completion(a) != scn.completion(b)
  # The prompt does not depend on which candidate renders it.
  assert render_prompt(scn._example(a)) == render_prompt(scn._example(b))


# --------------------------------------------------------------------------
# Records and the store
# --------------------------------------------------------------------------
def test_build_record_orders_chosen_and_rejected_by_decision():
  scn = _weather_scenario()
  a, b = scn.candidates
  rec_a = build_record(scn, a, b, "a")
  assert rec_a.chosen == scn.completion(a)
  assert rec_a.rejected == scn.completion(b)

  rec_b = build_record(scn, a, b, "b")
  assert rec_b.chosen == scn.completion(b)
  assert rec_b.rejected == scn.completion(a)


@pytest.mark.parametrize("decision", ["tie", "both_bad", "skip"])
def test_non_preferences_record_no_winner(decision):
  scn = _weather_scenario()
  a, b = scn.candidates
  rec = build_record(scn, a, b, decision)
  assert rec.chosen is None and rec.rejected is None
  # Both candidates are still preserved for later analysis.
  assert rec.candidate_a["completion"] == scn.completion(a)
  assert rec.candidate_b["completion"] == scn.completion(b)


def test_bad_decision_is_rejected():
  with pytest.raises(ValueError):
    PreferenceRecord(
      id="x", scenario_id="y", category="c", subtype="s", domain="d",
      system="", tools=[], state=None, memory=None, user="", prompt="",
      decision="left", candidate_a={}, candidate_b={},
    )


def test_store_round_trips_and_counts(tmp_path):
  store = PreferenceStore(tmp_path / "prefs.jsonl")
  assert len(store) == 0

  scn = _weather_scenario()
  a, b = scn.candidates
  store.append(build_record(scn, a, b, "a"))
  store.append(build_record(scn, a, b, "skip"))

  reloaded = PreferenceStore(tmp_path / "prefs.jsonl")
  assert len(reloaded) == 2
  recs = list(reloaded.read())
  assert recs[0].decision == "a" and recs[0].chosen is not None
  assert recs[1].decision == "skip"

  stats = reloaded.stats()
  assert stats["total"] == 2 and stats["pairs"] == 1
  assert stats["by_category"]["single_stage"] == 2


def test_store_skips_a_torn_line(tmp_path):
  path = tmp_path / "prefs.jsonl"
  scn = _weather_scenario()
  a, b = scn.candidates
  good = build_record(scn, a, b, "a").to_json()
  path.write_text(good + "\n" + '{"id": "half", "decisi' + "\n" + good + "\n")
  store = PreferenceStore(path)
  assert len(list(store.read())) == 2


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
def test_mock_generator_is_valid_and_repeatable():
  scn = MockGenerator(seed=1).next()
  assert len(scn.candidates) == 2
  assert scn.system in DOMAIN_SYSTEM.values()
  # Every candidate renders and parses.
  for cand in scn.candidates:
    assert parse_calls(scn.completion(cand)) == cand.calls
  # The tool block always carries `answer` (sample_tools appends it).
  assert any(t["name"] == "answer" for t in scn.tools)


def test_mock_generator_reaches_the_conflict_showcase():
  seen = {MockGenerator(seed=s).next().category for s in range(50)}
  assert "state_memory_conflict" in seen


def test_teacher_generator_parses_authored_setup_and_two_candidates():
  """A stub teacher: authors a setup, then answers it twice. No torch."""

  def fake_generate(model, tok, cfg, batch, *, seed=None):
    assert seed is not None
    out = []
    for item in batch:
      prompt = item.prompt()
      if prompt.startswith("Below is a tool-calling session"):
        # Answering: vary the call by seed so the two candidates differ.
        name = "answer" if seed % 2 else "web_search"
        out.append(json.dumps({"think": f"reasoning {seed}", "calls": [{"name": name, "arguments": {}}]}))
      else:
        # Authoring: a full example whose `turns` we expect to be discarded.
        out.append(json.dumps({
          "user": "A concrete user request.",
          "state": None,
          "memory": None,
          "turns": [{"role": "assistant", "think": "x", "calls": [{"name": "answer", "arguments": {"text": "y"}}]}],
        }))
    return out

  gen = TeacherGenerator(SFTConfig(target_examples=200), model=None, tok=None, generate=fake_generate)
  scn = gen.next()
  assert scn.user == "A concrete user request."
  assert len(scn.candidates) == 2
  assert all(c.source.startswith("teacher#") for c in scn.candidates)
  # Both candidates render into the student surface form and parse back.
  for cand in scn.candidates:
    assert parse_calls(scn.completion(cand)) is not None


def test_teacher_generator_never_authors_the_paired_category():
  def fake_generate(model, tok, cfg, batch, *, seed=None):
    return [json.dumps({"user": "u", "state": None, "memory": None, "turns": []}) for _ in batch]

  gen = TeacherGenerator(SFTConfig(target_examples=400), model=None, tok=None, generate=fake_generate)
  for _ in range(20):
    assert gen.next().category != "state_memory_conflict"
