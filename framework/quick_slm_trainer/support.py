"""Which framework versions a training version is allowed to run against.

A training version is a finished experiment. Its checkpoints exist and its paper
reports numbers measured from them, so the framework must not change what it
produces. That cannot hold forever: eventually the framework needs a change 
cannot absorb, and the honest move is to end its support at a named version
rather than to carry it silently or break it quietly.

Each `training/vN/` declares its window in `framework.json`:

  {"status": "supported", "framework_requires": ">=1.0,<2.0"}

and, once retired:

  {"status": "end-of-life",
   "framework_requires": ">=1.0,<=1.4.5",
   "archive_tag": "framework-1.4.5"}

That one file drives everything: what a notebook refuses to run against, which
frozen-config tests still apply, and what `SUPPORT.md` claims. Retiring a version
is a one-line edit to it.

**This module is permanent API.** It is the one thing that must keep working
across every major version, because it is what an old notebook calls to discover
that the framework it just installed is too new for it. Removing or renaming
`require_framework` would break exactly the check that exists to prevent silent
breakage. Add to it; do not reshape it.

No dependency on `packaging`: the comparator below is twenty lines, and the
alternative is asking a frozen notebook to install a package to find out that it
should not be running.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
  "SupportError",
  "VersionWindow",
  "check_framework",
  "load_window",
  "parse_version",
  "require_framework",
  "satisfies",
]


class SupportError(RuntimeError):
  """The installed framework is outside a training version's window."""


def parse_version(text: str) -> tuple[int, ...]:
  """"1.4.5" -> (1, 4, 5). Trailing non-numeric parts are dropped.

  Enough for the versions this project issues. A pre-release suffix such as
  "2.0.0rc1" compares equal to "2.0.0", which is the safe direction: a release
  candidate for the major that drops should already fail its window.
  """
  parts: list[int] = []
  for chunk in text.strip().split("."):
    digits = ""
    for ch in chunk:
      if not ch.isdigit():
        break
      digits += ch
    if not digits:
      break
    parts.append(int(digits))
  if not parts:
    raise ValueError(f"cannot read a version from {text!r}")
  return tuple(parts)


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Compare 1.0 against 1.0.0 as equal rather than as less-than."""
  n = max(len(a), len(b))
  return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


_OPS = {
  ">=": lambda a, b: a >= b,
  "<=": lambda a, b: a <= b,
  "==": lambda a, b: a == b,
  "!=": lambda a, b: a != b,
  ">": lambda a, b: a > b,
  "<": lambda a, b: a < b,
}


def satisfies(version: str, spec: str) -> bool:
  """Whether `version` meets every comma-separated clause of `spec`.

  `satisfies("1.4.5", ">=1.0,<2.0")` is True. An empty spec accepts anything.
  """
  got = parse_version(version)
  for clause in (c.strip() for c in spec.split(",")):
    if not clause:
      continue
    # Two-character operators first: ">=1.0" must not read as ">" then "=1.0".
    for op in ("!=", ">=", "<=", "==", ">", "<"):
      if clause.startswith(op):
        left, right = _pad(got, parse_version(clause[len(op):]))
        if not _OPS[op](left, right):
          return False
        break
    else:
      raise ValueError(f"cannot read a version constraint from {clause!r}")
  return True


class VersionWindow:
  """One training version's declared support window."""

  def __init__(self, data: dict, source: Path | None = None) -> None:
    self.data = data
    self.source = source
    self.training_version: str = data["training_version"]
    self.status: str = data["status"]
    self.requires: str = data["framework_requires"]
    self.tested: str | None = data.get("framework_tested")
    self.archive_tag: str | None = data.get("archive_tag")
    # Named for the JSON key it mirrors. A shorter attribute name here is how
    # the end-of-life branch of every consumer went untested until a version
    # was actually retired.
    self.last_supported_framework: str | None = data.get("last_supported_framework")
    self.ended: str | None = data.get("ended")

  @property
  def is_end_of_life(self) -> bool:
    return self.status == "end-of-life"

  def accepts(self, version: str) -> bool:
    return satisfies(version, self.requires)

  def install_hint(self, repo_url: str = "https://github.com/goldenmagicwizard/quick-slm") -> str:
    """How to get a framework this version will accept."""
    if self.archive_tag:
      return f'pip install "git+{repo_url}@{self.archive_tag}"'
    return f"check out a commit whose framework version satisfies {self.requires}"

  def __repr__(self) -> str: # pragma: no cover - debugging aid
    return f"<VersionWindow {self.training_version} {self.status} {self.requires}>"


def load_window(training_version: str, repo: str | Path) -> VersionWindow:
  """Read `<repo>/training/<training_version>/framework.json`."""
  path = Path(repo) / "training" / training_version / "framework.json"
  if not path.is_file():
    raise SupportError(
      f"{training_version} declares no support window: {path} is missing. "
      "Every training version needs a framework.json; see SUPPORT.md."
    )
  return VersionWindow(json.loads(path.read_text()), source=path)


def check_framework(training_version: str, repo: str | Path, version: str | None = None) -> VersionWindow:
  """Raise `SupportError` unless the installed framework is inside the window.

  Returns the window on success so a caller can report what it matched.
  """
  from . import __version__

  got = __version__ if version is None else version
  window = load_window(training_version, repo)
  if window.accepts(got):
    return window

  ended = f" (support ended {window.ended})" if window.ended else ""
  raise SupportError(
    f"quick-slm-trainer {got} is outside the support window for training "
    f"{training_version}{ended}.\n"
    f" {training_version} requires: {window.requires}\n"
    f" installed:      {got}\n"
    f"\n"
    f"This notebook is frozen against a finished run. Running it on a framework "
    f"outside its window would not fail loudly, it would build a different "
    f"corpus. Install the archived build instead:\n"
    f"\n"
    f"  {window.install_hint()}\n"
  )


def require_framework(training_version: str, repo: str | Path, *, quiet: bool = False) -> VersionWindow:
  """`check_framework`, and print what matched. What notebooks call."""
  window = check_framework(training_version, repo)
  if not quiet:
    from . import __version__

    state = "end-of-life" if window.is_end_of_life else "supported"
    print(
      f"quick-slm-trainer {__version__} satisfies training {training_version} "
      f"({state}, requires {window.requires})"
    )
  return window
