""" is a finished run. The framework must not reach backwards into it.

The framework is shared by every training version and will keep growing: wants
new tools, new sub-types, new presets. Additive change is safe. What is not safe
is change that alters what `pretrain_config()` and `sft_config()` produce, because the
checkpoints already exist and the paper already reports numbers measured from
them. A retuned default or an edited preset changes a published run without
touching a single file under `training/`.

These tests are the enforcement. `framework/frozen_config.json` records what
the framework resolved its presets to at freeze time, and a diff fails here at
commit time rather than on Colab, hours into a re-run.

Regenerate with `python scripts/freeze_config.py`, and only when the
difference is understood and deliberate. A failure here usually means the
framework change should have been additive instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quick_slm_trainer.config import pretrain_config, sft_config
from quick_slm_trainer.sft.generate import plan_requests
from quick_slm_trainer.support import load_window

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
FROZEN = REPO / "training" / "" / "frozen_config.json"

REGENERATE = "run `python scripts/freeze_config.py` only if the change is deliberate"

# The freeze binds the framework only while is supported. Retiring is an
# edit to its framework.json, and this is the line that reads it: once the status
# is end-of-life the framework is free to change what its presets produce, and
# these tests stand down rather than being deleted, so the record of what was
# frozen survives in the file and in git.
_WINDOW = load_window("", REPO)
pytestmark = pytest.mark.skipif(
  _WINDOW.is_end_of_life,
  reason=(
    f"training is end-of-life (support ended {_WINDOW.ended}, last framework "
    f"{_WINDOW.last_supported_framework}, archived at {_WINDOW.archive_tag}). "
    "The framework is no longer bound by its frozen config; see SUPPORT.md."
  ),
)


@pytest.fixture(scope="module")
def frozen() -> dict:
  assert FROZEN.is_file(), f"{FROZEN} is missing; {REGENERATE}"
  return json.loads(FROZEN.read_text())


def _differences(want: dict, got: dict, path: str = "") -> list[str]:
  """Every leaf that differs, named by its full path.

  A whole-dict assertion prints two 1,900-character blobs and leaves the reader
  to diff them by eye. The point of this test is to say which knob moved.
  """
  out = []
  for key in sorted(set(want) | set(got)):
    here = f"{path}.{key}" if path else key
    if key.startswith("_"):
      continue
    if key not in want:
      out.append(f"{here}: added, now {got[key]!r}")
    elif key not in got:
      out.append(f"{here}: removed, was {want[key]!r}")
    elif isinstance(want[key], dict) and isinstance(got[key], dict):
      out.extend(_differences(want[key], got[key], here))
    elif want[key] != got[key]:
      out.append(f"{here}: was {want[key]!r}, now {got[key]!r}")
  return out


def test_pretrain_config_still_resolves_to_what_it_did(frozen):
  diffs = _differences(frozen["pretrain_config"], pretrain_config().to_dict())
  assert not diffs, "the framework changed its pretraining config:\n " + "\n ".join(diffs)


def test_sft_config_still_resolves_to_what_it_did(frozen):
  diffs = _differences(frozen["sft_config"], sft_config().to_dict())
  assert not diffs, "the framework changed its SFT config:\n " + "\n ".join(diffs)


def test_the_v1_plan_is_still_the_same_size(frozen):
  # Config equality does not catch everything. The plan also depends on the seed
  # pools, the category specs and the two capacity caps, none of which live in
  # the config dict. Widening a seed pool for moves this number for .
  want = frozen["derived"]["planned_sft_requests"]
  got = len(plan_requests(sft_config().sft))
  assert got == want, (
    f"its SFT plan changed size: was {want:,} requests, now {got:,}. "
    "Something outside the config dict moved: a seed pool, a category spec, "
    f"or a capacity cap. {REGENERATE}"
  )


def test_the_frozen_file_is_what_the_freeze_script_writes():
  # Guards the guard: a hand-edited frozen_config.json would silently weaken
  # every test above.
  import importlib.util

  script = REPO / "training" / "" / "freeze_config.py"
  spec = importlib.util.spec_from_file_location("freeze_config", script)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  expected = json.dumps(module.snapshot(), indent=2, sort_keys=True) + "\n"
  assert FROZEN.read_text() == expected, (
    "frozen_config.json does not match what freeze_config.py writes; "
    "it was edited by hand, or the script changed"
  )
