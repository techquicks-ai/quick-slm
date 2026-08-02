"""End-to-end checks that the assembled report reaches the right verdict.

`test_diagnose.py` tests each statistic against synthetic inputs. This file
tests the thing that will actually be read: a whole `build_report` over a whole
synthetic corpus, driven by a simulated model whose failure mode is known in
advance. If the report cannot name a failure that was deliberately planted, it
cannot be trusted to name one that was not.

Each scenario plants exactly one diagnosis and asserts the report finds it *and*
does not report the others.
"""

from __future__ import annotations

import json

from quick_slm_trainer.sft import diagnose as D
from quick_slm_trainer.sft.grade import grade
from quick_slm_trainer.template import AssistantTurn, Example, UserTurn


def T(name, *args):
  return {"name": name, "description": name,
      "parameters": {"type": "object",
              "properties": {a: {"type": "string"} for a in args},
              "required": list(args)}}


TOOLS = [T("get_weather", "city"), T("get_capital_city", "country"), T("answer", "text")]


def make(city, *, category, subtype, i, tool="get_weather", arg="city"):
  """One example whose correct call is `tool(arg=city)`."""
  return Example(
    tools=TOOLS,
    turns=[UserTurn(f"{subtype} request number {i} about {city}"),
        AssistantTurn(think="x" * 20, calls=[{"name": tool, "arguments": {arg: city}}])],
    category=category,
    meta={"subtype": subtype},
  )


def emit(tool, arg, value):
  return (f'<think>\nI will use {tool} for this request.\n</think>\n'
      f'<response>[{{"name":"{tool}","arguments":{{"{arg}":"{value}"}}}}]</response>')


def perfect(ex):
  """A model that always emits the reference call."""
  call = ex.turns[1].calls[0]
  arg, val = next(iter(call["arguments"].items()))
  return emit(call["name"], arg, val)


def always_weather(ex):
  """A model with a prior: `get_weather` on the first token of the user turn."""
  city = ex.turns[0].text.split()[-1]
  return emit("get_weather", "city", city)


def corpus(fat_n, thin_n):
  """A corpus with one fat cell and one thin cell, disjoint in correct answer."""
  fat = [make(f"City{i}", category="single_stage", subtype="direct", i=i) for i in range(fat_n)]
  thin = [make(f"Land{i}", category="multi_stage", subtype="chain_2", i=i,
         tool="get_capital_city", arg="country") for i in range(thin_n)]
  return fat, thin


# ==========================================================================
def test_a_model_with_a_prior_is_diagnosed_as_composition_not_capacity():
  """The planted failure: the model learnt the fat cell's call and applies it
  everywhere. Held-out accuracy is mediocre in both cells' aggregate, and only
  the per-cell split plus the training control identify the cause."""
  fat, thin = corpus(600, 60)
  train = fat + thin
  val_fat = [make(f"City{i}", category="single_stage", subtype="direct", i=i) for i in range(900, 960)]
  val_thin = [make(f"Land{i}", category="multi_stage", subtype="chain_2", i=i,
           tool="get_capital_city", arg="country") for i in range(900, 960)]
  val = val_fat + val_thin

  results = [grade(e, always_weather(e)) for e in val]
  train_results = [grade(e, always_weather(e)) for e in train]

  report = D.build_report(results=results, train_examples=train,
              train_results=train_results, validated=train, kept=train)

  cells = {(c["category"], c["subtype"]): c for c in report["cells"]}
  assert cells[("single_stage", "direct")]["correct_rate"] == 1.0
  assert cells[("multi_stage", "chain_2")]["correct_rate"] == 0.0

  # The prior is named, with the call it fell back on.
  d = report["default_call"]
  assert d["top_call"] == "get_weather" and d["top_share"] == 1.0

  # And the confusion matrix shows what it substituted for what.
  swap = next(r for r in report["confusion"]
        if r["expected"] == "get_capital_city" and r["got"] == "get_weather")
  assert swap["n"] == 60

  # Not memorisation: training accuracy is no better than held-out. But not
  # "generalising" either -- a whole cell is at zero, and a verdict that read
  # only the train/val gap would have called this healthy. It is the level,
  # not the gap, that is wrong here.
  m = report["memorisation"]
  assert m["verdict"].startswith("uniform"), m["verdict"]
  assert m["median_gap"] == 0.0
  assert [(d["category"], d["subtype"]) for d in m["dead_cells"]] == [("multi_stage", "chain_2")]


def test_a_model_that_cannot_fit_its_training_data_is_diagnosed_as_capacity():
  """The planted failure: uniformly bad, including on data it was fit to.
  This must NOT be reported as a composition problem, because rebalancing a
  corpus would not help it."""
  fat, thin = corpus(600, 60)
  train = fat + thin
  val = [make(f"City{i}", category="single_stage", subtype="direct", i=i) for i in range(900, 960)]
  val += [make(f"Land{i}", category="multi_stage", subtype="chain_2", i=i,
         tool="get_capital_city", arg="country") for i in range(900, 960)]

  def hopeless(ex):
    return emit("get_weather", "city", "Nowhere")

  report = D.build_report(
    results=[grade(e, hopeless(e)) for e in val],
    train_examples=train,
    train_results=[grade(e, hopeless(e)) for e in train],
  )
  assert report["memorisation"]["verdict"].startswith("capacity")
  assert report["memorisation"]["train_mean"] < 0.5


def test_a_memorising_model_is_diagnosed_as_memorisation():
  """The planted failure: right on every training example, wrong on every
  held-out one. Volume is not the issue and the report must not blame it."""
  fat, thin = corpus(300, 300)
  train = fat + thin
  val = [make(f"Other{i}", category="single_stage", subtype="direct", i=i) for i in range(60)]
  val += [make(f"Else{i}", category="multi_stage", subtype="chain_2", i=i,
         tool="get_capital_city", arg="country") for i in range(60)]

  report = D.build_report(
    results=[grade(e, emit("get_weather", "city", "Wrong")) for e in val],
    train_examples=train,
    train_results=[grade(e, perfect(e)) for e in train],
  )
  m = report["memorisation"]
  assert m["verdict"].startswith("memorisation")
  assert m["train_mean"] == 1.0 and m["val_mean"] == 0.0


def test_a_good_model_is_not_reported_as_broken():
  """The control. A diagnostic that finds a problem in a healthy run is worse
  than none, because it will be believed."""
  fat, thin = corpus(300, 300)
  train = fat + thin
  val = [make(f"V{i}", category="single_stage", subtype="direct", i=i) for i in range(60)]
  val += [make(f"W{i}", category="multi_stage", subtype="chain_2", i=i,
         tool="get_capital_city", arg="country") for i in range(60)]

  report = D.build_report(
    results=[grade(e, perfect(e)) for e in val],
    train_examples=train,
    train_results=[grade(e, perfect(e)) for e in train],
  )
  assert report["memorisation"]["verdict"].startswith("generalising")
  assert report["default_call"]["n_wrong"] == 0
  assert report["think_call_coupling"]["decoupled"] == 0
  assert report["failures"] == []
  assert all(c["correct_rate"] == 1.0 for c in report["cells"] if c["n_val"])


def test_the_dedup_forensics_explain_a_cell_that_collapsed():
  """The planted failure: one cell where the teacher wrote the same thing 50
  times. The report must attribute the loss to repetition, not to volume."""
  repeated = [make("Same", category="single_stage", subtype="direct", i=0) for _ in range(50)]
  varied = [make(f"V{i}", category="single_stage", subtype="paraphrased", i=i) for i in range(50)]
  validated = repeated + varied
  kept = [repeated[0]] + varied

  report = D.build_report(results=[], train_examples=[], validated=validated, kept=kept)
  rows = {r["subtype"]: r for r in report["dedup"]}

  assert rows["direct"]["largest_cluster"] == 50
  assert rows["direct"]["distinct_fingerprints"] == 1
  assert rows["direct"]["kept_rate"] == 0.02
  assert rows["paraphrased"]["largest_cluster"] == 1
  assert rows["paraphrased"]["kept_rate"] == 1.0
  assert report["dedup"][0]["subtype"] == "direct", "worst cell must sort first"


def test_the_report_survives_a_round_trip_through_json():
  fat, thin = corpus(40, 40)
  train = fat + thin
  report = D.build_report(
    results=[grade(e, always_weather(e)) for e in train],
    train_examples=train,
    train_results=[grade(e, always_weather(e)) for e in train],
    validated=train, kept=train,
    meta={"checkpoint": "step_0000899"},
  )
  back = json.loads(json.dumps(report, default=str))
  assert back["meta"]["checkpoint"] == "step_0000899"
  assert back["cells"] and back["dedup"] and back["confusion"]
