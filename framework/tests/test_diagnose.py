from __future__ import annotations

import json

import pytest

from quick_slm_trainer.sft import diagnose as D
from quick_slm_trainer.sft.grade import BOTH, NEITHER, ONE, Result, grade
from quick_slm_trainer.template import AssistantTurn, Example, UserTurn


def T(name, *args):
  return {"name": name, "description": name,
      "parameters": {"type": "object",
              "properties": {a: {"type": "string"} for a in args},
              "required": list(args)}}


TOOLS = [T("get_weather", "city"), T("get_capital_city", "country"),
     T("resume_building", "building_id"), T("answer", "text")]

THINK = "The user wants something, so I will pick a tool."


def gen(think, calls_json):
  return f"<think>\n{think}\n</think>\n<response>{calls_json}</response>"


def weather(city):
  return f'[{{"name":"get_weather","arguments":{{"city":"{city}"}}}}]'


def ex(calls, *, category="single_stage", subtype="direct", user="do the thing", meta=None):
  m = {"subtype": subtype}
  m.update(meta or {})
  return Example(tools=TOOLS, turns=[UserTurn(user), AssistantTurn(think="x" * 20, calls=calls)],
          category=category, meta=m)


def call(name, **args):
  return [{"name": name, "arguments": args}]


# ==========================================================================
# Composition versus capacity -- the decision this module exists for
# ==========================================================================
def _cells(train_n, train_acc, val_acc, n_val=40):
  """Synthetic cells with dialled-in accuracies."""
  return [
    {"category": "c", "subtype": f"s{i}", "n_train": tn, "train_share": 0.1,
     "n_val": n_val, "well_formed_rate": 1.0, "correct_rate": va,
     "n_train_graded": 40, "train_correct_rate": ta, "thin": n_val < D.MIN_CELL_N}
    for i, (tn, ta, va) in enumerate(zip(train_n, train_acc, val_acc))
  ]


def test_a_model_that_fails_its_own_training_data_is_called_capacity():
  # The distinguishing case. Low on both sides is not a data problem, and
  # reading it as one would send to rebalance a corpus that was never the
  # constraint.
  cells = _cells([1000] * 4, train_acc=[0.3, 0.25, 0.2, 0.35], val_acc=[0.28, 0.22, 0.2, 0.3])
  assert D.memorisation_gap(cells)["verdict"].startswith("capacity")


def test_fitting_training_and_failing_validation_is_called_memorisation():
  cells = _cells([1000] * 4, train_acc=[0.95, 0.9, 0.92, 0.98], val_acc=[0.4, 0.35, 0.5, 0.45])
  g = D.memorisation_gap(cells)
  assert g["verdict"].startswith("memorisation")
  assert g["median_gap"] > 0.25
  assert g["worst"][0]["gap"] >= g["worst"][-1]["gap"]


def test_train_and_validation_tracking_each_other_is_called_generalising():
  cells = _cells([1000] * 4, train_acc=[0.85, 0.8, 0.9, 0.82], val_acc=[0.8, 0.78, 0.85, 0.8])
  assert D.memorisation_gap(cells)["verdict"].startswith("generalising")


def test_a_small_gap_at_a_low_level_is_not_called_generalising():
  # The trap this guards: a model with a strong prior is consistent across
  # train and held-out and consistently wrong wherever the prior misses. The
  # gap is ~0, and a verdict reading only the gap calls that healthy.
  cells = _cells([1000] * 4, train_acc=[0.55, 0.52, 0.5, 0.58], val_acc=[0.54, 0.5, 0.48, 0.56])
  v = D.memorisation_gap(cells)
  assert v["verdict"].startswith("uniform")
  assert abs(v["median_gap"]) < 0.05, "the gap really is small; the level is the problem"


def test_a_cell_at_exactly_zero_is_surfaced_even_when_the_average_looks_fine():
  cells = _cells([1000] * 4, train_acc=[0.95] * 4, val_acc=[0.95, 0.92, 0.9, 0.0])
  v = D.memorisation_gap(cells)
  assert v["dead_cells"] and v["dead_cells"][0]["subtype"] == "s3"
  assert not v["verdict"].startswith("generalising")


def test_the_gap_is_not_reported_when_training_was_not_graded():
  # Silence beats a verdict computed from nothing.
  cells = _cells([1000] * 4, train_acc=[0.9] * 4, val_acc=[0.5] * 4)
  for c in cells:
    c["train_correct_rate"] = None
  assert D.memorisation_gap(cells)["cells"] == 0


def test_thin_cells_are_excluded_from_the_verdict():
  cells = _cells([1000] * 4, train_acc=[0.9] * 4, val_acc=[0.1] * 4, n_val=3)
  for c in cells:
    c["n_val"] = 3
  assert D.memorisation_gap(cells)["cells"] == 0


def test_correlation_reads_thin_failing_cells_as_a_composition_problem():
  # Accuracy rising monotonically with training volume is the signature.
  cells = _cells([50, 200, 800, 3000, 9000], train_acc=[0.5] * 5,
          val_acc=[0.1, 0.3, 0.5, 0.7, 0.9])
  v = D.volume_accuracy_correlation(cells)
  assert v["rho"] == pytest.approx(1.0)
  assert v["reading"].startswith("composition")


def test_correlation_says_volume_is_not_the_problem_when_it_is_not():
  cells = _cells([50, 9000, 200, 3000, 800], train_acc=[0.5] * 5, val_acc=[0.5] * 5)
  v = D.volume_accuracy_correlation(cells)
  assert v["rho"] == pytest.approx(0.0)
  assert v["reading"] == "volume is not the limiting factor"


def test_correlation_flags_the_inverted_case():
  cells = _cells([50, 200, 800, 3000, 9000], train_acc=[0.5] * 5,
          val_acc=[0.9, 0.7, 0.5, 0.3, 0.1])
  assert D.volume_accuracy_correlation(cells)["reading"].startswith("inverted")


def test_correlation_declines_to_answer_with_too_few_cells():
  assert D.volume_accuracy_correlation(_cells([1, 2], [0.5] * 2, [0.5] * 2))["rho"] is None


def test_cell_report_joins_train_volume_to_held_out_accuracy():
  train = [ex(call("get_weather", city="A"), subtype="direct")] * 100
  train += [ex(call("answer", text="no"), category="traps", subtype="no_relevant_tool")] * 5
  results = [grade(ex(call("get_weather", city="A"), subtype="direct"), gen(THINK, weather("A")))] * 30
  results += [grade(ex(call("answer", text="no"), category="traps", subtype="no_relevant_tool"),
           gen(THINK, weather("A")))] * 4

  rows = {(r["category"], r["subtype"]): r for r in D.cell_report(results, train)}
  direct = rows[("single_stage", "direct")]
  assert direct["n_train"] == 100 and direct["n_val"] == 30
  assert direct["correct_rate"] == 1.0 and direct["thin"] is False

  trap = rows[("traps", "no_relevant_tool")]
  assert trap["n_train"] == 5 and trap["correct_rate"] == 0.0
  assert trap["thin"] is True, "a 4-example cell must be flagged, not read as 0%"


# ==========================================================================
# What the model does instead
# ==========================================================================
def test_default_call_finds_the_prior_the_model_fell_back_on():
  results = [
    grade(ex(call("answer", text="which city?"), category="traps"), gen(THINK, weather("Japan"))),
    grade(ex(call("answer", text="no tool"), category="traps"), gen(THINK, weather("Tokyo"))),
    grade(ex(call("get_capital_city", country="France"), category="multi_stage"),
       gen(THINK, weather("France"))),
  ]
  d = D.default_call(results)
  assert d["n_wrong"] == 3
  assert d["top_call"] == "get_weather" and d["top_share"] == 1.0


def test_confusion_pairs_expected_against_emitted_names():
  results = [
    grade(ex(call("answer", text="?"), category="traps"), gen(THINK, weather("Japan"))),
    grade(ex(call("get_weather", city="A")), gen(THINK, weather("A"))),
  ]
  rows = {(r["expected"], r["got"]): r for r in D.confusion(results)}
  assert rows[("answer", "get_weather")]["correct_name"] is False
  assert rows[("get_weather", "get_weather")]["correct_name"] is True


def test_confusion_ignores_malformed_generations():
  results = [grade(ex(call("get_weather", city="A")), gen(THINK, "not json"))]
  assert D.confusion(results) == []


# ==========================================================================
# Think/call coupling -- the France probe's failure mode
# ==========================================================================
def test_coupling_catches_reasoning_that_names_one_tool_and_calls_another():
  r = grade(ex(call("get_capital_city", country="France"), category="multi_stage"),
       gen("I should use get_capital_city to find the capital.", weather("France")))
  c = D.think_call_coupling([r])
  assert c["considered"] == 1 and c["decoupled"] == 1 and c["decoupled_rate"] == 1.0
  assert c["examples"][0]["think_named"] == "get_capital_city"
  assert c["examples"][0]["called"] == "get_weather"


def test_coupling_counts_agreement_when_reasoning_and_call_match():
  r = grade(ex(call("get_weather", city="Tokyo")),
       gen("I will use get_weather for this.", weather("Tokyo")))
  c = D.think_call_coupling([r])
  assert c["agreed"] == 1 and c["decoupled"] == 0


def test_coupling_abstains_when_the_reasoning_names_two_tools():
  # "mentions two tools" has no unambiguous reading, so it is not counted
  # either way rather than guessed at.
  r = grade(ex(call("get_weather", city="Tokyo")),
       gen("Not get_capital_city, I will use get_weather.", weather("Tokyo")))
  assert D.think_call_coupling([r])["considered"] == 0


def test_coupling_flags_reasoning_that_held_the_answer_the_call_dropped():
  # The France probe exactly: the think block says Paris, the call says France.
  r = grade(ex(call("get_weather", city="Paris"), category="multi_stage"),
       gen("The capital is Paris so the city should be Paris.", weather("France")))
  assert D.think_call_coupling([r])["think_had_answer_call_did_not"] == 1


def test_coupling_does_not_flag_a_correct_answer():
  r = grade(ex(call("get_weather", city="Paris")),
       gen("The city is Paris.", weather("Paris")))
  assert D.think_call_coupling([r])["think_had_answer_call_did_not"] == 0


# ==========================================================================
# Pairs
# ==========================================================================
def _pair(pid, spec, a_city, b_city):
  return [
    ex(call("get_weather", city=a_city), category="state_memory_conflict",
      subtype="stale_memory", meta={"group": pid, "pair_id": pid, "branch": "a", "spec_id": spec}),
    ex(call("get_weather", city=b_city), category="state_memory_conflict",
      subtype="stale_memory", meta={"group": pid, "pair_id": pid, "branch": "b", "spec_id": spec}),
  ]


def test_pair_diagnostics_separates_specifications():
  good = _pair("p1", "spec_ok", "A", "B")
  bad = _pair("p2", "spec_bad", "C", "D")
  results = [
    grade(good[0], gen(THINK, weather("A"))), grade(good[1], gen(THINK, weather("B"))),
    grade(bad[0], gen(THINK, weather("C"))), grade(bad[1], gen(THINK, weather("C"))),
  ]
  rows = {r["spec_id"]: r for r in D.pair_diagnostics(results)["by_spec"]}
  assert rows["spec_ok"][BOTH] == 1 and rows["spec_ok"]["grounded_rate"] == 1.0
  assert rows["spec_bad"][ONE] == 1 and rows["spec_bad"]["grounded_rate"] == 0.0


def test_pair_rows_are_sorted_worst_first():
  ps = _pair("p1", "ok", "A", "B") + _pair("p2", "bad", "C", "D")
  results = [
    grade(ps[0], gen(THINK, weather("A"))), grade(ps[1], gen(THINK, weather("B"))),
    grade(ps[2], gen(THINK, weather("C"))), grade(ps[3], gen(THINK, weather("C"))),
  ]
  assert D.pair_diagnostics(results)["by_spec"][0]["spec_id"] == "bad"


def test_branch_bias_detects_a_lopsided_pair_construction():
  # If every "exactly one" win lands on branch a, the construction is asymmetric
  # and the pair metric is measuring the data, not the model.
  results = []
  for i in range(6):
    a, b = _pair(f"p{i}", "spec", "A", "B")
    results += [grade(a, gen(THINK, weather("A"))), grade(b, gen(THINK, weather("A")))]
  bias = D.pair_diagnostics(results)["branch_bias"]
  assert bias["wins"] == {"a": 6} and bias["skew"] == 1.0


def test_neither_correct_is_recorded_distinctly_from_one():
  a, b = _pair("p1", "spec", "A", "B")
  results = [grade(a, gen(THINK, weather("Z"))), grade(b, gen(THINK, weather("Z")))]
  row = D.pair_diagnostics(results)["by_spec"][0]
  assert row[NEITHER] == 1 and row[ONE] == 0 and row[BOTH] == 0


# ==========================================================================
# Dedup forensics -- "why did the data get deduplicated"
# ==========================================================================
def test_forensics_shows_how_repetitive_the_teacher_was_per_cell():
  # 10 identical records and 2 distinct ones in one cell.
  same = [ex(call("get_weather", city="Tokyo"), subtype="direct", user="same question") for _ in range(10)]
  diff = [ex(call("get_weather", city="Osaka"), subtype="direct", user="another question"),
      ex(call("get_weather", city="Kyoto"), subtype="direct", user="a third question")]
  validated = same + diff
  kept = [same[0]] + diff

  row = next(r for r in D.dedup_forensics(validated, kept) if r["subtype"] == "direct")
  assert row["n_validated"] == 12 and row["n_kept"] == 3
  assert row["distinct_fingerprints"] == 3
  assert row["largest_cluster"] == 10, "the 10 identical records must show as one cluster"
  assert row["exact_duplicate_rate"] == pytest.approx(0.75)
  assert row["records_per_kept"] == pytest.approx(4.0)


def test_forensics_sorts_the_worst_surviving_cell_first():
  a = [ex(call("get_weather", city="A"), subtype="bad", user="q") for _ in range(10)]
  b = [ex(call("get_weather", city=f"C{i}"), subtype="good", user=f"q{i}") for i in range(10)]
  rows = D.dedup_forensics(a + b, [a[0]] + b)
  assert rows[0]["subtype"] == "bad" and rows[0]["kept_rate"] < rows[1]["kept_rate"]


# ==========================================================================
# Length and assembly
# ==========================================================================
def test_length_buckets_report_accuracy_by_prompt_size():
  # Deliberately: the short prompt is answered correctly and the long one is
  # not, so a bucketing that collapsed them would show 50% in one row instead
  # of 100% and 0% in two, and the degradation would be invisible.
  short = grade(ex(call("get_weather", city="A"), user="hi"), gen(THINK, weather("A")))
  long_ = grade(ex(call("get_weather", city="A"), user="x" * 4000), gen(THINK, weather("B")))
  rows = {r["bucket"]: r for r in D.length_buckets([short, long_])}

  assert len(rows) == 2, f"both prompts landed in one bucket: {list(rows)}"
  assert sum(r["n"] for r in rows.values()) == 2
  assert rows[">=2048"] == {"bucket": ">=2048", "n": 1,
               "correct_rate": 0.0, "well_formed_rate": 1.0}
  short_bucket = next(b for b in rows if b != ">=2048")
  assert rows[short_bucket]["correct_rate"] == 1.0


def test_length_buckets_put_a_prompt_in_exactly_the_band_it_belongs_to():
  for n_chars, expect in ((10, "<256"), (400, "256-512"), (700, "512-1024"),
              (1500, "1024-2048"), (9000, ">=2048")):
    r = grade(ex(call("get_weather", city="A"), user="x" * n_chars), gen(THINK, weather("A")))
    rows = D.length_buckets([r])
    assert len(rows) == 1 and rows[0]["n"] == 1
    # The rendered prompt is longer than the user turn (tools, system), so
    # assert monotonicity of the band rather than an exact equality.
    bands = ["<256", "256-512", "512-1024", "1024-2048", ">=2048"]
    assert bands.index(rows[0]["bucket"]) >= bands.index(expect) - 1


def test_failure_samples_are_bounded_and_carry_enough_to_read():
  results = [grade(ex(call("answer", text="?"), category="traps"), gen(THINK, weather("X")))] * 50
  s = D.failure_samples(results, per_category=12)
  assert len(s) == 12
  assert s[0]["expected"] and s[0]["got"] and s[0]["user"]


def test_the_whole_report_is_json_serialisable():
  # It is written to Drive and read back elsewhere; a non-serialisable value
  # would only surface at the end of a long GPU run.
  a, b = _pair("p1", "spec", "A", "B")
  results = [
    grade(a, gen(THINK, weather("A"))), grade(b, gen(THINK, weather("A"))),
    grade(ex(call("get_weather", city="Z")), gen(THINK, weather("Z"))),
    grade(ex(call("answer", text="?"), category="traps"), gen(THINK, "malformed")),
  ]
  report = D.build_report(
    results=results,
    train_examples=[ex(call("get_weather", city="A"))] * 10,
    validated=[a, b], kept=[a, b],
    train_results=results,
    meta={"checkpoint": "step_0000899"},
  )
  text = json.dumps(report)
  assert json.loads(text)["meta"]["checkpoint"] == "step_0000899"
  for key in ("cells", "memorisation", "volume_vs_accuracy", "default_call",
        "confusion", "think_call_coupling", "pairs", "length", "failures", "dedup"):
    assert key in report, key


def test_build_report_omits_dedup_when_the_corpora_are_not_supplied():
  r = [grade(ex(call("get_weather", city="A")), gen(THINK, weather("A")))]
  assert "dedup" not in D.build_report(results=r, train_examples=[])
