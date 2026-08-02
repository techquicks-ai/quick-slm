"""The support matrix has to be true, not merely written down.

`SUPPORT.md` promises that `1.x` runs `training` and that dropping a training
version takes a major bump. A promise in prose decays: the framework version
moves, a window is edited, and the table still reads the way it did last year.
These tests make the prose and the `framework.json` files agree, and fail the
build when the installed framework falls outside a window it claims to support.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quick_slm_trainer import __version__
from quick_slm_trainer.support import (
  SupportError,
  check_framework,
  load_window,
  parse_version,
  satisfies,
)

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SUPPORT_MD = REPO / "SUPPORT.md"
WINDOWS = sorted((REPO / "training").glob("v*/framework.json"))


@pytest.fixture(scope="module")
def windows() -> list:
  return [load_window(p.parent.name, REPO) for p in WINDOWS]


# --------------------------------------------------------------------------
# The comparator
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
  "version,spec,expected",
  [
    ("1.0.0", ">=1.0,<2.0", True),
    ("1.4.5", ">=1.0,<2.0", True),
    ("2.0.0", ">=1.0,<2.0", False),
    ("0.9.9", ">=1.0,<2.0", False),
    ("1.4.5", ">=1.0,<=1.4.5", True),
    ("1.4.6", ">=1.0,<=1.4.5", False),
    # A shorter version is padded, not treated as smaller.
    ("1.0", ">=1.0.0", True),
    ("1.0.0", ">=1.0", True),
    # A release candidate for the major that drops must already fail its
    # window. Comparing equal to the release is the safe direction.
    ("2.0.0rc1", "<2.0", False),
    ("", ">=1.0", None), # unreadable, must raise
  ],
)
def test_the_comparator_agrees_with_the_windows_it_will_be_asked_about(version, spec, expected):
  if expected is None:
    with pytest.raises(ValueError):
      satisfies(version, spec)
  else:
    assert satisfies(version, spec) is expected


def test_an_unreadable_constraint_is_an_error_not_a_pass():
  # Silently accepting a typo'd constraint would make every window vacuous.
  with pytest.raises(ValueError):
    satisfies("1.0.0", "~=1.0")
  with pytest.raises(ValueError):
    satisfies("1.0.0", "1.0")


# --------------------------------------------------------------------------
# The windows themselves
# --------------------------------------------------------------------------
def test_there_is_at_least_one_training_version_declared():
  # Guards the guard: a moved directory would empty the glob and silently
  # dissolve every parametrised test below.
  assert WINDOWS, "no training/v*/framework.json found"


def test_every_window_is_well_formed(windows):
  for w in windows:
    assert w.status in {"supported", "end-of-life"}, (w.training_version, w.status)
    assert w.requires, w.training_version
    satisfies("1.0.0", w.requires) # raises if the spec is unreadable
    if w.is_end_of_life:
      assert w.archive_tag, f"{w.training_version} is end-of-life with no archive tag"
      assert w.last_supported_framework, w.training_version
      assert w.ended, f"{w.training_version} is end-of-life with no end date"
    else:
      assert w.archive_tag is None, (
        f"{w.training_version} is supported but names an archive tag; "
        "the tag is for retired versions"
      )


def test_the_installed_framework_satisfies_every_supported_window(windows):
  # The core rule. A minor or patch release that falls outside a supported
  # version's window is a major release that forgot to say so.
  for w in windows:
    if w.is_end_of_life:
      continue
    assert w.accepts(__version__), (
      f"quick-slm-trainer {__version__} is outside the window training "
      f"{w.training_version} declares ({w.requires}). Either this release "
      "should be a major bump that retires it, or the window is wrong. "
      "See SUPPORT.md."
    )


def test_check_framework_accepts_the_current_build_and_names_the_fix_when_it_does_not():
  window = check_framework("", REPO)
  assert window.training_version == ""

  with pytest.raises(SupportError) as excinfo:
    check_framework("", REPO, version="99.0.0")
  message = str(excinfo.value)
  assert "99.0.0" in message and window.requires in message
  # The message has to carry the way out, not just the diagnosis.
  assert "pip install" in message or "check out" in message


def test_a_missing_window_is_an_error_rather_than_a_pass():
  with pytest.raises(SupportError, match="declares no support window"):
    load_window("v-does-not-exist", REPO)


# --------------------------------------------------------------------------
# SUPPORT.md against the windows
# --------------------------------------------------------------------------
def test_support_md_lists_every_training_version(windows):
  text = SUPPORT_MD.read_text()
  for w in windows:
    assert f"training/{w.training_version}" in text or f"[{w.training_version}]" in text, (
      f"{w.training_version} has a framework.json but no row in SUPPORT.md"
    )


def test_support_md_states_the_current_framework_version():
  text = SUPPORT_MD.read_text()
  assert f"**{__version__}**" in text, (
    f"SUPPORT.md does not name the current framework version {__version__}; "
    "the matrix has drifted from the code"
  )


def test_support_md_agrees_with_each_window_on_status_and_range(windows):
  rows = {
    m.group(1): m.group(0)
    for m in re.finditer(r"^\| \[?(v\d+)\]?[^\n]*$", SUPPORT_MD.read_text(), re.M)
  }
  for w in windows:
    assert w.training_version in rows, f"no matrix row for {w.training_version}"
    row = rows[w.training_version]
    assert w.status in row, (
      f"SUPPORT.md row for {w.training_version} does not say {w.status!r}: {row}"
    )
    assert w.requires in row, (
      f"SUPPORT.md row for {w.training_version} does not carry its range "
      f"{w.requires!r}: {row}"
    )


def test_the_major_version_rule_holds(windows):
  # SUPPORT.md: "a major bump means a training version lost support". So every
  # still-supported version must admit the current major, and the framework's
  # major must not have moved past a supported window's ceiling.
  major = parse_version(__version__)[0]
  for w in windows:
    if w.is_end_of_life:
      continue
    assert satisfies(f"{major}.0.0", w.requires) or w.accepts(__version__), (
      f"training {w.training_version} is marked supported but its window "
      f"{w.requires} excludes the current major {major}.x. Retiring it "
      "requires setting status to end-of-life, not leaving it stranded."
    )


def test_pyproject_and_package_agree_on_the_version():
  declared = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
  assert declared, "pyproject.toml declares no version"
  assert declared.group(1) == __version__, (
    f"pyproject says {declared.group(1)}, package says {__version__}. "
    "A wheel built from this tree would claim a version the code denies."
  )
