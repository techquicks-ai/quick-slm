"""Structural validation of teacher output.

Every check here rejects rather than repairs. A repaired example is one the
teacher got wrong and the pipeline guessed at, and a 103M student learns the
guess as readily as the intent. `docs/sft_readme.md` budgets a 25 to 35
percent total reject rate; generation is sized for it.

The reject reasons are strings rather than an enum so that the counter can be
serialised straight into `corpus_stats.json` and quoted in the paper's data
section without a lookup table.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Sequence

from ..template import AssistantTurn, Example, ToolTurn, UserTurn, format_memory
from .specs import CategorySpec
from .tools import Tool, by_name, properties_of, required_args

# --------------------------------------------------------------------------
# Reject reasons
# --------------------------------------------------------------------------
NOT_JSON = "not_json"
BAD_SHAPE = "bad_shape"
EMPTY_USER = "empty_user"
BAD_TURN_ORDER = "bad_turn_order"
TURN_COUNT_OUT_OF_BOUNDS = "turn_count_out_of_bounds"
EMPTY_THINK = "empty_think"
THINK_LEAKED_MARKUP = "think_leaked_markup"
NO_CALLS = "no_calls"
UNDEFINED_TOOL = "undefined_tool"
MISSING_REQUIRED_ARG = "missing_required_arg"
UNKNOWN_ARG = "unknown_arg"
BAD_ARG_TYPE = "bad_arg_type"
FINAL_CALL_NOT_ANSWER = "final_call_not_answer"
STATE_MISSING = "state_missing"
STATE_UNEXPECTED = "state_unexpected"
MEMORY_MISSING = "memory_missing"
MEMORY_UNEXPECTED = "memory_unexpected"

# --- Counterfactual state swap, category 3. See `conflict.py`. -------------
#: The teacher's call does not match the oracle for its branch.
SWAP_CALL_MISMATCH = "swap_call_mismatch"
#: The other branch of this pair failed, so this one is dropped with it.
SWAP_PARTNER_FAILED = "swap_partner_failed"
#: Only one branch of the pair reached the shard.
SWAP_MISSING_BRANCH = "swap_missing_branch"
#: `<memory>` is not byte-identical across the branches, so the model could win
#: by reading memory and the pair stops testing state-grounding.
SWAP_MEMORY_DIVERGED = "swap_memory_diverged"
#: The two states differ somewhere other than the decisive path: a spurious cue.
SWAP_STATE_DIVERGED = "swap_state_diverged"
#: The user turn is not identical across the branches.
SWAP_REQUEST_DIVERGED = "swap_request_diverged"
#: Flipping the decisive field did not change the correct call. A spec fault.
SWAP_ORACLES_AGREE = "swap_oracles_agree"
#: More than one call, which the single-call oracle cannot be compared against.
SWAP_MULTIPLE_CALLS = "swap_multiple_calls"

MIN_THINK_CHARS = 15
# `<think>` and `<response>` in the reasoning field mean the teacher started
# writing the template instead of reasoning inside it.
#
# `state` and `memory` are deliberately NOT in this list, though they once were.
# They are content blocks the teacher is *instructed* to consult: the paired
# prompt's rule 4 reads "Read the values in <state> before choosing. <state> was
# sampled this turn and outranks <memory> wherever the two disagree." A teacher
# that then reasons "the <state> block shows b4 is paused" is following that
# instruction, not leaking the envelope, and rejecting it cost 481 records on the
# run. Because the paired category is accepted or rejected as a pair, each of
# those killed a second, blameless branch with it.
#
# The tags that remain are the envelope's own: they cannot appear unless the
# teacher stopped writing content and started writing the template around it.
_MARKUP = re.compile(r"<\s*/?\s*(think|response|tool|\|im_)", re.IGNORECASE)


class ValidationStats:
  """Reject tally, and the accepted count."""

  def __init__(self) -> None:
    self.accepted = 0
    self.rejected: Counter[str] = Counter()

  def record(self, reason: str | None) -> bool:
    if reason is None:
      self.accepted += 1
      return True
    self.rejected[reason] += 1
    return False

  @property
  def total(self) -> int:
    return self.accepted + sum(self.rejected.values())

  def to_dict(self) -> dict:
    return {
      "seen": self.total,
      "accepted": self.accepted,
      "reject_rate": (sum(self.rejected.values()) / self.total) if self.total else 0.0,
      "rejected": dict(self.rejected.most_common()),
    }

  def report(self) -> str:
    lines = [f"seen {self.total:,} accepted {self.accepted:,}"]
    if self.total:
      rate = 100.0 * sum(self.rejected.values()) / self.total
      lines[0] += f" reject rate {rate:.1f}%"
    for reason, n in self.rejected.most_common():
      lines.append(f" {reason:<28s} {n:>7,} ({100.0 * n / max(1, self.total):.1f}%)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
  """The first complete JSON object in the teacher's reply.

  Teachers wrap output in markdown fences and prepend "Here is the example:"
  despite being told not to, at a low but nonzero rate. Scanning for a balanced
  brace span recovers those instead of discarding them. String literals are
  tracked so a brace inside `"text"` does not close the object early.
  """
  if not text:
    return None
  start = text.find("{")
  if start == -1:
    return None

  depth, in_str, escaped = 0, False, False
  for i in range(start, len(text)):
    c = text[i]
    if in_str:
      if escaped:
        escaped = False
      elif c == "\\":
        escaped = True
      elif c == '"':
        in_str = False
      continue
    if c == '"':
      in_str = True
    elif c == "{":
      depth += 1
    elif c == "}":
      depth -= 1
      if depth == 0:
        try:
          obj = json.loads(text[start : i + 1])
        except (json.JSONDecodeError, ValueError):
          return None
        return obj if isinstance(obj, dict) else None
  return None


# --------------------------------------------------------------------------
# Argument type checking
# --------------------------------------------------------------------------
def _type_ok(value: Any, declared: dict) -> bool:
  t = declared.get("type")
  if t is None:
    return True
  if t == "boolean":
    return isinstance(value, bool)
  # `bool` is a subclass of `int` in Python, so True would pass an integer
  # check and bind as 1. Every tool that takes a quantity would accept it.
  if t == "integer":
    return isinstance(value, int) and not isinstance(value, bool)
  if t == "number":
    return isinstance(value, (int, float)) and not isinstance(value, bool)
  if t == "string":
    return isinstance(value, str)
  if t == "object":
    return isinstance(value, dict)
  if t == "array":
    return isinstance(value, list)
  return True


def check_think(text: Any) -> str | None:
  """`<think>` must be non-empty prose. Nothing about its wording is checked.

  Category 3 used to require the words "state" and "memory" to appear here.
  That filter selected for narration rather than for behaviour, and it is gone:
  see the module docstring of `conflict.py`. The reasoning field is now checked
  for exactly two things, in every category alike, and this is both of them.
  """
  if not isinstance(text, str) or len(text.strip()) < MIN_THINK_CHARS:
    return EMPTY_THINK
  if _MARKUP.search(text):
    return THINK_LEAKED_MARKUP
  return None


def check_call(call: Any, allowed: dict[str, Tool]) -> str | None:
  """Structural check of one call against the tools it was offered."""
  if not isinstance(call, dict) or "name" not in call:
    return BAD_SHAPE
  name = call.get("name")
  args = call.get("arguments", {})
  if not isinstance(name, str) or not isinstance(args, dict):
    return BAD_SHAPE
  if name not in allowed:
    return UNDEFINED_TOOL

  tool = allowed[name]
  props = properties_of(tool)
  for req in required_args(tool):
    if req not in args:
      return MISSING_REQUIRED_ARG
  for key, value in args.items():
    if key not in props:
      return UNKNOWN_ARG
    if not _type_ok(value, props[key]):
      return BAD_ARG_TYPE
  return None


# --------------------------------------------------------------------------
# Record validation
# --------------------------------------------------------------------------
def validate_record(rec: Any, spec: CategorySpec, tools: Sequence[Tool]) -> str | None:
  """`None` when the record is usable, otherwise the reject reason.

  Paired categories never come through here. Their accept decision is taken
  over both branches at once, and a per-sample verdict on one of them is
  exactly the thing the pair construction exists to prevent.
  """
  if spec.paired:
    raise ValueError(
      f"category {spec.key!r} is paired; use conflict.validate_pair, which accepts or "
      "rejects both branches together"
    )
  if not isinstance(rec, dict):
    return NOT_JSON

  user = rec.get("user")
  if not isinstance(user, str) or not user.strip():
    return EMPTY_USER

  turns = rec.get("turns")
  if not isinstance(turns, list) or not turns:
    return BAD_SHAPE

  # Assistant first, assistant last, strict alternation between.
  roles = [t.get("role") if isinstance(t, dict) else None for t in turns]
  if roles[0] != "assistant" or roles[-1] != "assistant":
    return BAD_TURN_ORDER
  expected = ["assistant" if i % 2 == 0 else "tool" for i in range(len(roles))]
  if roles != expected:
    return BAD_TURN_ORDER

  assistants = [t for t in turns if t.get("role") == "assistant"]
  lo, hi = spec.n_assistant_turns
  if not lo <= len(assistants) <= hi:
    return TURN_COUNT_OUT_OF_BOUNDS

  allowed = by_name(tools)
  for turn in assistants:
    reason = check_think(turn.get("think"))
    if reason is not None:
      return reason

    calls = turn.get("calls")
    if not isinstance(calls, list) or not calls:
      return NO_CALLS
    for call in calls:
      reason = check_call(call, allowed)
      if reason is not None:
        return reason

  if spec.final_call_must_be_answer:
    final = assistants[-1]["calls"]
    if len(final) != 1 or final[0].get("name") != "answer":
      return FINAL_CALL_NOT_ANSWER

  state = rec.get("state")
  if spec.needs_state and state is None:
    return STATE_MISSING
  if not spec.needs_state and state is not None:
    return STATE_UNEXPECTED

  memory = rec.get("memory")
  if spec.needs_memory and not _memory_populated(memory):
    return MEMORY_MISSING
  if not spec.needs_memory and _memory_populated(memory):
    return MEMORY_UNEXPECTED

  return None


def _memory_populated(memory: Any) -> bool:
  if not isinstance(memory, dict):
    return False
  return bool(memory.get("recent")) or bool(memory.get("last_results"))


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def record_to_example(
  rec: dict,
  spec: CategorySpec,
  tools: Sequence[Tool],
  *,
  system: str,
  subtype: str = "",
) -> Example:
  """Build the `Example` the packer will render. Assumes `validate_record` passed."""
  turns: list = [UserTurn(rec["user"].strip())]
  for turn in rec["turns"]:
    if turn["role"] == "assistant":
      turns.append(AssistantTurn(think=turn["think"].strip(), calls=turn["calls"]))
    else:
      turns.append(ToolTurn(result=turn["result"]))

  memory = None
  if _memory_populated(rec.get("memory")):
    mem = rec["memory"]
    memory = format_memory(
      recent=mem.get("recent", []) or [],
      last_results=mem.get("last_results", []) or [],
    )

  return Example(
    tools=list(tools),
    turns=turns,
    state=rec.get("state"),
    memory=memory,
    system=system,
    category=spec.key,
    meta={"subtype": subtype},
  )


def validate_and_convert(
  rec: Any,
  spec: CategorySpec,
  tools: Sequence[Tool],
  *,
  system: str,
  subtype: str = "",
  stats: ValidationStats | None = None,
) -> Example | None:
  reason = validate_record(rec, spec, tools)
  if stats is not None:
    stats.record(reason)
  if reason is not None:
    return None
  return record_to_example(rec, spec, tools, system=system, subtype=subtype)
