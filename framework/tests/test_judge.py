"""Tests for model-graded evaluation.

The judge itself is a language model and cannot be asserted on. Everything
around it can: which cases get judged, what the prompt contains and withholds,
how replies are parsed, what happens when the judge misbehaves, and whether the
calibration check would actually catch an unreliable judge. Those are the parts
that decide whether the scores mean anything, so those are the parts under test.
"""

from __future__ import annotations

import pytest

from quick_slm_trainer.sft import judge as J
from quick_slm_trainer.sft.grade import grade
from quick_slm_trainer.template import AssistantTurn, Example, UserTurn


def T(name, *args):
  return {"name": name, "description": name,
      "parameters": {"type": "object",
              "properties": {a: {"type": "string"} for a in args},
              "required": list(args)}}


TOOLS = [T("get_weather", "city"), T("resume_building", "building_id"), T("answer", "text")]
THINK = "The user is asking about the weather, so I will look at the tools."


def gen(think, calls_json):
  return f"<think>\n{think}\n</think>\n<response>{calls_json}</response>"


def weather(city):
  return f'[{{"name":"get_weather","arguments":{{"city":"{city}"}}}}]'


def ex(calls, *, category="single_stage", state=None, memory=None, user="what is the weather?", meta=None):
  return Example(tools=TOOLS,
          turns=[UserTurn(user), AssistantTurn(think="x" * 20, calls=calls)],
          state=state, memory=memory, category=category, meta=meta or {})


def call(name, **args):
  return [{"name": name, "arguments": args}]


WEATHER_TOKYO = call("get_weather", city="Tokyo")


# ==========================================================================
# Parsing
# ==========================================================================
def test_a_well_formed_reply_parses():
  s, r = J.parse_verdict("SCORE: 4\nREASON: declined clearly but was wordy")
  assert s == 4 and r == "declined clearly but was wordy"


@pytest.mark.parametrize("text", ["", "I think this is pretty good actually", "SCORE: seven"])
def test_an_unparseable_reply_is_recorded_as_unparsed_not_as_zero(text):
  # Coercing a rambling judge to 0 would move the mean silently every time it
  # failed to follow the format, and hide how often that happened.
  score, _ = J.parse_verdict(text)
  assert score is None


def test_parsing_tolerates_the_judge_being_chatty_around_the_format():
  s, r = J.parse_verdict("Sure! Here is my assessment.\n\nSCORE: 2\nREASON: vague\n\nHope that helps")
  assert s == 2 and r == "vague"


def test_scores_outside_the_rubric_are_not_accepted():
  assert J.parse_verdict("SCORE: 9\nREASON: x")[0] is None


# ==========================================================================
# Prompt construction -- what the judge is shown, and what it is not
# ==========================================================================
def test_the_reasoning_prompt_withholds_the_call():
  # The dimension is "was the reasoning sound", and showing a correct call
  # alongside bad reasoning invites the judge to score the call instead.
  r = grade(ex(WEATHER_TOKYO, category="state_memory_conflict", state={"b4": "paused"}),
       gen("The state says b4 is paused.", weather("Tokyo")))
  p = J.build_prompt(J.REASONING, r)
  assert "The state says b4 is paused." in p
  assert "get_weather" not in p.split("MODEL REASONING")[1]


def test_the_reasoning_prompt_shows_state_and_memory_and_ranks_them():
  r = grade(ex(WEATHER_TOKYO, category="state_memory_conflict",
         state={"b4": "running"}, memory="b4 was paused"),
       gen(THINK, weather("Tokyo")))
  p = J.build_prompt(J.REASONING, r)
  assert "STATE (authoritative)" in p and "MEMORY (may be stale)" in p
  assert "outranks" in p


def test_the_trap_prompt_lists_the_tools_so_no_tool_applies_is_checkable():
  r = grade(ex(call("answer", text="?"), category="traps"), gen(THINK, weather("Japan")))
  p = J.build_prompt(J.TRAP, r)
  assert "TOOLS AVAILABLE" in p and "get_weather" in p


def test_the_refusal_prompt_does_not_ask_about_syntax():
  # Syntax is the oracle's job; asking twice double-counts it.
  r = grade(ex(call("answer", text="no"), category="refusals"), gen(THINK, weather("X")))
  p = J.build_prompt(J.REFUSAL, r)
  assert "scored separately" in p


def test_the_triage_prompt_forbids_rescoring_correctness():
  r = grade(ex(WEATHER_TOKYO), gen(THINK, weather("Osaka")))
  p = J.build_prompt(J.TRIAGE, r)
  assert "do NOT re-score correctness" in p
  assert "REFERENCE CALL" in p


def test_every_prompt_ends_with_the_response_format():
  r = grade(ex(WEATHER_TOKYO), gen(THINK, weather("Tokyo")))
  for dim in (J.REFUSAL, J.TRAP, J.REASONING, J.TRIAGE, J.CALIBRATION):
    assert J.build_prompt(dim, r).rstrip().endswith("REASON: <one short sentence>")


def test_an_unknown_dimension_is_rejected():
  r = grade(ex(WEATHER_TOKYO), gen(THINK, weather("Tokyo")))
  with pytest.raises(ValueError):
    J.build_prompt("vibes", r)


def test_calls_are_rendered_as_json_not_as_a_python_repr():
  # str() on a dict gives {'city': 'Tokyo'}, which is Python's repr. The judge
  # is being asked to compare two tool calls; give it the JSON it has seen a
  # million of, not a language-specific rendering.
  r = grade(ex(WEATHER_TOKYO), gen(THINK, weather("Osaka")))
  p = J.build_prompt(J.TRIAGE, r)
  assert '"name": "get_weather"' in p and '"city": "Osaka"' in p
  assert "'city'" not in p


def test_state_is_rendered_as_json_too():
  r = grade(ex(WEATHER_TOKYO, category="state_memory_conflict", state={"b4": "paused"}),
       gen(THINK, weather("Tokyo")))
  assert '"b4": "paused"' in J.build_prompt(J.REASONING, r)


def test_long_fields_are_truncated_so_one_example_cannot_blow_the_context():
  r = grade(ex(WEATHER_TOKYO, user="x" * 9000), gen(THINK, weather("Tokyo")))
  assert "[truncated]" in J.build_prompt(J.TRIAGE, r)


# ==========================================================================
# Target selection -- the actual argument of the module
# ==========================================================================
def test_refusals_and_traps_are_judged_because_the_oracle_cannot_see_them():
  results = [
    grade(ex(call("answer", text="no"), category="refusals"),
       gen(THINK, '[{"name":"answer","arguments":{"text":"banana"}}]')),
    grade(ex(call("answer", text="which city?"), category="traps"),
       gen(THINK, '[{"name":"answer","arguments":{"text":"it is sunny"}}]')),
  ]
  # Both score CORRECT against the oracle, because answer.text is free text.
  assert all(r.correct for r in results)
  # ...which is exactly why both must be judged.
  dims = {d for d, _ in J.targets(results)}
  assert dims == {J.REFUSAL, J.TRAP}


def test_the_counterfactual_category_is_judged_on_reasoning_only():
  # Its call correctness is oracle-decided and must not be re-litigated.
  r = grade(ex(WEATHER_TOKYO, category="state_memory_conflict", state={"x": 1},
         meta={"group": "p1", "branch": "a"}),
       gen(THINK, weather("Tokyo")))
  dims = [d for d, _ in J.targets([r])]
  assert dims == [J.REASONING]
  assert J.TRIAGE not in dims


def test_correct_answers_outside_refusals_and_traps_are_not_judged_at_all():
  r = grade(ex(WEATHER_TOKYO), gen(THINK, weather("Tokyo")))
  assert J.targets([r]) == []


def test_malformed_generations_are_not_judged():
  # A parse fault already fully describes them; a score would add nothing.
  r = grade(ex(WEATHER_TOKYO), gen(THINK, "not json at all"))
  assert not r.well_formed
  assert J.targets([r]) == []


def test_triage_is_capped_so_it_stays_a_sample():
  results = [grade(ex(WEATHER_TOKYO), gen(THINK, weather("Wrong"))) for _ in range(500)]
  plan = J.targets(results, triage_limit=25)
  assert sum(1 for d, _ in plan if d == J.TRIAGE) == 25


def test_calibration_targets_are_balanced_between_right_and_wrong():
  # A judge that says "matches" to everything scores 100% on a correct-only
  # sample. The check is worthless unless both classes are present.
  results = [grade(ex(WEATHER_TOKYO), gen(THINK, weather("Tokyo"))) for _ in range(40)]
  results += [grade(ex(WEATHER_TOKYO), gen(THINK, weather("Osaka"))) for _ in range(40)]
  plan = J.calibration_targets(results, n=20)
  picked = [results[i] for _, i in plan]
  assert sum(r.correct for r in picked) == 10
  assert sum(not r.correct for r in picked) == 10


# ==========================================================================
# Running, and misbehaving judges
# ==========================================================================
def test_judge_all_scores_every_planned_pair():
  results = [grade(ex(call("answer", text="no"), category="refusals"),
           gen(THINK, '[{"name":"answer","arguments":{"text":"no"}}]'))]
  verdicts = J.judge_all(results, lambda p: "SCORE: 5\nREASON: clean decline")
  assert len(verdicts) == 1 and verdicts[0].score == 5
  assert verdicts[0].dimension == J.REFUSAL


def test_a_judge_that_raises_costs_one_score_not_the_run():
  results = [grade(ex(call("answer", text="no"), category="refusals"),
           gen(THINK, '[{"name":"answer","arguments":{"text":"no"}}]'))
        for _ in range(3)]
  calls = {"n": 0}

  def flaky(prompt):
    calls["n"] += 1
    if calls["n"] == 2:
      raise RuntimeError("CUDA OOM")
    return "SCORE: 4\nREASON: ok"

  verdicts = J.judge_all(results, flaky)
  assert len(verdicts) == 3
  assert [v.parsed for v in verdicts] == [True, False, True]
  assert "CUDA OOM" in verdicts[1].reason


# ==========================================================================
# Calibration -- would it actually catch a bad judge?
# ==========================================================================
def _cal_setup():
  results = [grade(ex(WEATHER_TOKYO), gen(THINK, weather("Tokyo"))) for _ in range(10)]
  results += [grade(ex(WEATHER_TOKYO), gen(THINK, weather("Osaka"))) for _ in range(10)]
  return results, J.calibration_targets(results, n=20)


def test_a_judge_that_tracks_the_oracle_is_called_trustworthy():
  results, plan = _cal_setup()
  verdicts = J.judge_all(
    results,
    # Reads the MODEL CALL block only, exactly as a real judge would.
    lambda p: ("SCORE: 5\nREASON: match"
          if '"Tokyo"' in p.split("MODEL CALL")[1].split("REFERENCE CALL")[0]
          else "SCORE: 0\nREASON: differs"),
    plan=plan)
  c = J.calibration(verdicts, results)
  assert c["n"] == 20 and c["agreement"] == 1.0 and c["trustworthy"] is True


def test_a_judge_that_passes_everything_is_caught():
  # The self-preference failure mode: flatters the model, agrees with the
  # oracle only on the half that was already correct.
  results, plan = _cal_setup()
  verdicts = J.judge_all(results, lambda p: "SCORE: 5\nREASON: looks fine", plan=plan)
  c = J.calibration(verdicts, results)
  assert c["agreement"] == 0.5
  assert c["trustworthy"] is False
  assert c["judge_said_correct_but_oracle_says_wrong"] == 10
  assert "noise, not measurement" in c["note"]


def test_calibration_reports_nothing_rather_than_guessing_when_it_did_not_run():
  results, _ = _cal_setup()
  assert J.calibration([], results)["agreement"] is None


# ==========================================================================
# Summary
# ==========================================================================
def test_the_summary_separates_dimensions_and_keeps_calibration_out_of_them():
  results = [
    grade(ex(call("answer", text="no"), category="refusals"),
       gen(THINK, '[{"name":"answer","arguments":{"text":"no"}}]')),
    grade(ex(call("answer", text="?"), category="traps"),
       gen(THINK, '[{"name":"answer","arguments":{"text":"sunny"}}]')),
  ]
  verdicts = J.judge_all(results, lambda p: "SCORE: 3\nREASON: middling")
  verdicts += J.judge_all(results, lambda p: "SCORE: 5\nREASON: match",
              plan=[(J.CALIBRATION, 0)])
  s = J.summarise(verdicts, results)

  assert set(s["dimensions"]) == {J.REFUSAL, J.TRAP}, "calibration is not a quality score"
  assert s["dimensions"][J.REFUSAL]["mean"] == 3.0
  assert s["dimensions"][J.TRAP]["distribution"] == {3: 1}


def test_the_summary_states_the_self_preference_bias():
  # The teacher and the judge are the same model. Anywhere these numbers are
  # quoted, that has to travel with them.
  s = J.summarise([], [])
  assert s["judge_is_the_corpus_teacher"] is True
  assert "Gemma 4" in s["self_preference_warning"]


def test_unparsed_replies_are_counted_not_averaged_in():
  results = [grade(ex(call("answer", text="no"), category="refusals"),
           gen(THINK, '[{"name":"answer","arguments":{"text":"no"}}]'))
        for _ in range(4)]
  replies = iter(["SCORE: 4\nREASON: a", "waffle", "SCORE: 2\nREASON: b", "more waffle"])
  verdicts = J.judge_all(results, lambda p: next(replies))
  d = J.summarise(verdicts, results)["dimensions"][J.REFUSAL]
  assert d["n"] == 4 and d["n_scored"] == 2
  assert d["unparsed_rate"] == 0.5
  assert d["mean"] == 3.0, "the two unparsed replies must not be counted as zeros"


def test_report_lines_warn_when_calibration_was_skipped():
  lines = "\n".join(J.report_lines(J.summarise([], [])))
  assert "not run" in lines and "unvalidated" in lines
