"""RFC 6901 JSON pointers, plus a structural diff.

The counterfactual state swap needs three operations and nothing else: read the
value at `decisive_path`, produce a copy of the state with that value replaced,
and prove that the two copies differ *nowhere else*.

That last one is the reason this module exists rather than a dict lookup. The
whole construction rests on the two branches of a pair being identical outside
`decisive_path`. If a building id, a timestamp, or a reordered key differs, the
model can predict the branch from the incidental cue and the pair stops measuring
state-grounding. `diff_paths` is what turns that from an intention into an
assertion.
"""

from __future__ import annotations

import copy
from typing import Any

#: `default=_MISSING` is how `get` is told to raise. `exists` therefore cannot
#: pass it as a probe, and `None` will not do either: `research_start` flips its
#: decisive field between `None` and a tech id, so a resolved `None` and an
#: unresolved pointer must stay distinguishable.
_MISSING = object()
_PROBE = object()


class PointerError(KeyError):
  """The pointer does not resolve against this document."""


def escape(token: str) -> str:
  return token.replace("~", "~0").replace("/", "~1")


def unescape(token: str) -> str:
  # `~1` before `~0`, per RFC 6901. The other order turns `~01` into `/`.
  return token.replace("~1", "/").replace("~0", "~")


def parse(pointer: str) -> list[str]:
  if pointer == "":
    return []
  if not pointer.startswith("/"):
    raise PointerError(f"a JSON pointer must be empty or start with '/': {pointer!r}")
  return [unescape(t) for t in pointer[1:].split("/")]


def _descend(node: Any, token: str) -> Any:
  if isinstance(node, list):
    try:
      index = int(token)
    except ValueError:
      return _MISSING
    return node[index] if 0 <= index < len(node) else _MISSING
  if isinstance(node, dict):
    return node.get(token, _MISSING)
  return _MISSING


def get(document: Any, pointer: str, default: Any = _MISSING) -> Any:
  node = document
  for token in parse(pointer):
    node = _descend(node, token)
    if node is _MISSING:
      if default is _MISSING:
        raise PointerError(f"{pointer!r} does not resolve")
      return default
  return node


def exists(document: Any, pointer: str) -> bool:
  return get(document, pointer, default=_PROBE) is not _PROBE


def with_value(document: Any, pointer: str, value: Any) -> Any:
  """A deep copy of `document` with `pointer` set to `value`.

  The target must already exist. A `decisive_path` with a typo would otherwise
  quietly add a new key, produce two states that differ at a field no oracle
  reads, and yield a pair whose branches have the same correct call. That pair
  would then be rejected downstream for the wrong reason, and the typo would
  survive.
  """
  tokens = parse(pointer)
  if not tokens:
    raise PointerError("cannot replace the document root")

  out = copy.deepcopy(document)
  node = out
  for token in tokens[:-1]:
    node = _descend(node, token)
    if node is _MISSING:
      raise PointerError(f"{pointer!r} does not resolve")

  last = tokens[-1]
  if isinstance(node, list):
    try:
      index = int(last)
    except ValueError as e:
      raise PointerError(f"{pointer!r} indexes a list with {last!r}") from e
    if not 0 <= index < len(node):
      raise PointerError(f"{pointer!r} is out of range")
    node[index] = value
  elif isinstance(node, dict):
    if last not in node:
      raise PointerError(f"{pointer!r} does not resolve; refusing to create {last!r}")
    node[last] = value
  else:
    raise PointerError(f"{pointer!r} does not resolve")
  return out


def _scalar_eq(a: Any, b: Any) -> bool:
  # `True == 1` and `False == 0` in Python. A decisive path holding a boolean
  # would then read as unchanged when flipped against an integer variant, and
  # the pair would be rejected for the wrong reason.
  if isinstance(a, bool) != isinstance(b, bool):
    return False
  return a == b


def diff_paths(a: Any, b: Any, prefix: str = "") -> set[str]:
  """Every pointer at which `a` and `b` disagree.

  Descends into dicts and lists and reports the shallowest differing pointer.
  A key present on one side only is reported at that key. Two documents are
  identical exactly when this returns the empty set.
  """
  if isinstance(a, dict) and isinstance(b, dict):
    out: set[str] = set()
    for key in set(a) | set(b):
      child = f"{prefix}/{escape(str(key))}"
      if key not in a or key not in b:
        out.add(child)
      else:
        out |= diff_paths(a[key], b[key], child)
    return out

  if isinstance(a, list) and isinstance(b, list):
    if len(a) != len(b):
      return {prefix}
    out = set()
    for i, (x, y) in enumerate(zip(a, b)):
      out |= diff_paths(x, y, f"{prefix}/{i}")
    return out

  return set() if _scalar_eq(a, b) else {prefix}
