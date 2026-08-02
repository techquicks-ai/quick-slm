"""The keys the evaluation notebooks read out of `corpus_stats.json`.

`07_sft_eval` and `08_sft_diagnose` assert that the split they rebuild is the
split that was packed, and they do it by reading counts back out of the stats
file. Those reads are string keys into a JSON blob, so nothing connects them to
`PackStats.to_dict` -- renaming a field there leaves the notebooks importable,
passing every other test, and failing with a KeyError partway through a GPU
session, after the dedup pass has already run.

That is exactly what happened: the notebooks were written against `examples`,
`PackStats` calls it `examples_in`, and the failure surfaced two minutes into a
run. This file is the missing connection.
"""

from __future__ import annotations

import json

import pytest

from quick_slm_trainer.sft.pack import PackStats, write_stats
from quick_slm_trainer.paths import Layout

#: Read by the assertion in section 4 of both evaluation notebooks.
REQUIRED_PACK_KEYS = ("examples_in", "examples_packed", "windows",
           "total_tokens", "real_tokens", "scored_tokens")


@pytest.mark.parametrize("key", REQUIRED_PACK_KEYS)
def test_packstats_exposes_the_keys_the_notebooks_read(key):
  assert key in PackStats().to_dict(), (
    f"07_sft_eval and 08_sft_diagnose read pack.<split>.{key}; renaming it "
    "breaks them at runtime, not at import"
  )


def test_the_notebook_split_check_resolves_against_a_real_stats_file(tmp_path):
  """Walk the exact key path the notebooks walk, over a file `write_stats` wrote."""
  # Both roots, not just the drive one: `mkdirs_sft` also creates
  # `local_sft_raw_dir`, which otherwise defaults under /content.
  layout = Layout(drive_root=tmp_path / "drive", local_root=tmp_path / "local")
  layout.mkdirs_sft()
  train, val = PackStats(), PackStats()
  train.examples_in, val.examples_in = 26_338, 1_386

  write_stats(layout, {
    "after_dedup": {"examples": 27_724},
    "after_rebalance": {"examples": 27_724},
    "pack": {"train": train.to_dict(), "val": val.to_dict()},
  })
  recorded = json.loads(layout.sft_stats_path.read_text())

  def rec(*path):
    node = recorded
    for k in path:
      if not isinstance(node, dict) or k not in node:
        return None
      node = node[k]
    return node

  assert rec("after_rebalance", "examples") == 27_724
  assert rec("pack", "train", "examples_in") == 26_338
  assert rec("pack", "val", "examples_in") == 1_386
  # The defensive walk returns None rather than raising on an absent path.
  assert rec("pack", "val", "examples") is None
  assert rec("nope", "nope") is None


def test_every_key_path_the_notebooks_actually_read_resolves(tmp_path):
  """The direct guard: read the paths out of the notebook source itself.

  The two tests above check that `PackStats` has the right fields and that a
  hand-written path resolves. Neither would have caught the original bug,
  which was a *notebook* asking for a key that never existed. This one lifts
  the literal key paths out of the committed notebooks and resolves each
  against a stats file written by `write_stats`.
  """
  import json as _json
  import re
  from pathlib import Path as _Path

  layout = Layout(drive_root=tmp_path / "drive", local_root=tmp_path / "local")
  layout.mkdirs_sft()
  train, val = PackStats(), PackStats()
  train.examples_in, val.examples_in = 26_338, 1_386
  write_stats(layout, {
    "after_dedup": {"examples": 27_724},
    "after_rebalance": {"examples": 27_724},
    "pack": {"train": train.to_dict(), "val": val.to_dict()},
  })
  recorded = _json.loads(layout.sft_stats_path.read_text())

  nb_dir = _Path(__file__).resolve().parents[2] / "training" / "" / "notebooks"
  notebooks = sorted(nb_dir.glob("0[78]_*.ipynb"))
  assert notebooks, f"no evaluation notebooks found under {nb_dir}"

  checked = 0
  for nb_path in notebooks:
    nb = _json.loads(nb_path.read_text())
    source = "".join(
      "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    for call in re.findall(r"_rec\(([^)]*)\)", source):
      path = re.findall(r"['\"]([^'\"]+)['\"]", call)
      if not path:
        continue
      node = recorded
      for key in path:
        assert isinstance(node, dict) and key in node, (
          f"{nb_path.name} reads corpus_stats path {path}, but "
          f"{key!r} is not there. Available: "
          f"{sorted(node) if isinstance(node, dict) else type(node).__name__}"
        )
        node = node[key]
      checked += 1

  assert checked >= 6, f"expected both notebooks' key paths, only saw {checked}"


def test_the_val_count_alone_cannot_detect_a_reshuffled_split():
  """Why the check reads three counts and not one.

  `split_examples` targets `round(N * val_fraction)`, so the corpus can change
  by a few examples without moving the val count at all -- while the split
  itself is completely different, because ungrouped examples are keyed by
  enumeration index and every later key shifts.
  """
  from quick_slm_trainer.sft.pack import split_examples
  from quick_slm_trainer.template import AssistantTurn, Example, UserTurn

  def corpus(n):
    return [
      Example(tools=[], turns=[UserTurn(f"q{i}"),
                   AssistantTurn(think="t" * 20, calls=[{"name": "a", "arguments": {}}])],
          category="single_stage", meta={})
      for i in range(n)
    ]

  before_train, before_val = split_examples(corpus(1000), val_fraction=0.05)
  after_train, after_val = split_examples(corpus(1008), val_fraction=0.05)

  # The val count barely moves...
  assert len(before_val) == 50 and len(after_val) == 50
  # ...but the train count does, which is what makes the check discriminating.
  assert len(before_train) == 950 and len(after_train) == 958

  # And the membership genuinely changed, so a count-only check on val would
  # have waved through a split the model had partly trained on.
  def ids(exs):
    return {e.turns[0].text for e in exs}

  assert ids(before_val) != ids(after_val), (
    "if this ever holds, the index-keying changed and the corpus-size check "
    "is the only remaining guard"
  )
