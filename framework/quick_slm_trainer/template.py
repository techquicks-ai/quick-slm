"""Prompt templates. The single source of truth for both training stages.

Two templates live here:

`Pretraining` (flat)
  What the xLAM slice of the pretraining corpus was rendered into, and what
  the capability probes must therefore use when they test tool-call shape.
  A probe that reformats the prompt is not measuring the model, it is
  measuring the mismatch.

`ChatML` (SFT)
  The envelope the supervised stage teaches. `<state>` and `<memory>` sit
  outside the `<|im_start|>` turns on purpose: they are injected by the
  inference server, not authored by any role, and the model should treat them
  as ambient context in the same way it treats a system prompt.

Before this module existed the flat template was written out by hand in three
places (the data-prep notebook's xLAM renderer, the checkpoint-test notebook's
probe builder, and the design doc). A 103M model conditions on exact surface
form, so a stray space between `calls:` and the JSON array is not cosmetic. One
renderer, imported everywhere, removes the possibility.

Note on `<call>` / `</call>`: the tokenizer reserves the pair and the README's
token table claims SFT uses it, but the canonical SFT template in
`docs/sft_readme.md` does not. It is unused. The table is wrong, not this
file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence, Union

# --------------------------------------------------------------------------
# Token constants. `tokenizer.SPECIAL_TOKENS` is assembled from these.
# --------------------------------------------------------------------------
TOOL_OPEN, TOOL_CLOSE = "<tool>", "</tool>"
RESPONSE_OPEN, RESPONSE_CLOSE = "<response>", "</response>"
CALL_OPEN, CALL_CLOSE = "<call>", "</call>" # reserved, currently unused
STATE_OPEN, STATE_CLOSE = "<state>", "</state>"
MEMORY_OPEN, MEMORY_CLOSE = "<memory>", "</memory>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

DEFAULT_SYSTEM = "You are a helpful assistant with access to these tools:"

#: The label value cross-entropy skips. Defined here rather than beside the
#: dataset because the mask is a property of the template, not of the storage
#: format, and `encode` below is what produces it.
IGNORE_INDEX = -100

Role = Literal["system", "user", "assistant", "tool"]


def dumps_compact(obj: Any) -> str:
  """JSON with no incidental whitespace. Used by the ChatML template."""
  return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _as_json_str(obj: Any) -> str:
  """Pass strings through untouched, dump everything else.

  xLAM stores `tools` and `answers` as pre-serialised JSON strings. Round
  tripping them through `json.loads`/`json.dumps` would silently renormalise
  their whitespace, so the pretraining renderer never does.
  """
  return obj if isinstance(obj, str) else json.dumps(obj)


# ==========================================================================
# Pretraining template (flat). Frozen: the base checkpoint saw exactly this.
# ==========================================================================
def render_pretrain_prompt(query: str, tools: Any) -> str:
  """Everything up to and including the point the model starts generating.

  Ends with a trailing space after `calls:`, which is not a typo. The corpus
  was written that way and the model's first generated token follows it.
  """
  return (
    f"User request: {query}\n"
    f"{TOOL_OPEN} Available tools: {_as_json_str(tools)} {TOOL_CLOSE}\n"
    f"{RESPONSE_OPEN} Valid function calls: "
  )


def render_pretrain_example(query: str, tools: Any, answers: Any) -> str:
  """A complete training document: prompt plus the target completion."""
  return f"{render_pretrain_prompt(query, tools)}{_as_json_str(answers)} {RESPONSE_CLOSE}"


# ==========================================================================
# ChatML template (SFT)
# ==========================================================================
@dataclass
class UserTurn:
  text: str


@dataclass
class ToolTurn:
  """A tool result fed back to the model. Never scored."""

  result: Any # dict, list, or a pre-rendered string


@dataclass
class AssistantTurn:
  """The only kind of turn that carries loss."""

  think: str
  calls: Sequence[dict]


#: `typing.Union` rather than `X | Y`, which is evaluated at class-creation time
#: and would make this module, and therefore the whole package, unimportable on
#: Python 3.9. Every other union here sits inside a deferred annotation.
Turn = Union[UserTurn, ToolTurn, AssistantTurn]


@dataclass
class Segment:
  """A slice of the rendered example, tagged with whether loss applies.

  Segment boundaries always fall immediately next to a special token. That is
  load-bearing: a HuggingFace fast tokenizer splits its input on added tokens
  and encodes each chunk independently, so encoding the segments separately
  and concatenating produces exactly the ids that encoding the whole string
  would. Introduce a boundary in the middle of ordinary prose and that
  guarantee is gone, because a BPE merge could have spanned it.
  """

  text: str
  scored: bool


@dataclass
class Example:
  """One SFT training example."""

  tools: Sequence[dict]
  turns: Sequence[Turn]
  state: Any | None = None
  memory: str | None = None
  system: str = DEFAULT_SYSTEM
  category: str = ""
  meta: dict = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not any(isinstance(t, AssistantTurn) for t in self.turns):
      raise ValueError("an example with no assistant turn carries no loss")
    if not isinstance(self.turns[0], UserTurn):
      raise ValueError("the first turn must be a UserTurn")


def group_of(ex: Example) -> str | None:
  """Examples that must move together through dedup and the train/val split.

  The two branches of a counterfactual pair share a group. Splitting them apart
  would leave a lone branch in the corpus, which is indistinguishable from a
  sample the teacher got right by pattern-matching the user turn, and letting
  one cross into the validation split would leak: the branches differ in one
  field of `<state>` and in nothing else.

  `None` means the example is its own group.
  """
  return ex.meta.get("group")


def format_memory(recent: Iterable[str] = (), last_results: Iterable[str] = ()) -> str:
  """Render the session-memory body.

  `<recent>` and `<last_results>` are ordinary text, not reserved tokens. They
  cost a few tokens each and buy the `<think>` block something to point at
  when it reconciles memory against state.
  """
  lines: list[str] = []
  recent = list(recent)
  last_results = list(last_results)
  if recent:
    lines.append("<recent>")
    lines.extend(f" {i}. {r}" for i, r in enumerate(recent, 1))
    lines.append("</recent>")
  if last_results:
    lines.append("<last_results>")
    lines.extend(f" {r}" for r in last_results)
    lines.append("</last_results>")
  return "\n".join(lines)


def _segments(ex: Example) -> list[Segment]:
  segs: list[Segment] = []
  prompt: list[str] = []

  prompt.append(
    f"{IM_START}system\n{ex.system}\n"
    f"{TOOL_OPEN}{dumps_compact(list(ex.tools))}{TOOL_CLOSE}\n{IM_END}\n"
  )
  if ex.state is not None:
    body = ex.state if isinstance(ex.state, str) else dumps_compact(ex.state)
    prompt.append(f"{STATE_OPEN}{body}{STATE_CLOSE}\n")
  if ex.memory is not None:
    prompt.append(f"{MEMORY_OPEN}\n{ex.memory}\n{MEMORY_CLOSE}\n")

  pending = "".join(prompt)

  for turn in ex.turns:
    if isinstance(turn, UserTurn):
      pending += f"{IM_START}user\n{turn.text}\n{IM_END}\n"
    elif isinstance(turn, ToolTurn):
      body = turn.result if isinstance(turn.result, str) else dumps_compact(turn.result)
      pending += f"{IM_START}tool\n{body}\n{IM_END}\n"
    elif isinstance(turn, AssistantTurn):
      # The header cues the model that its turn has begun; it is context,
      # not target. Loss starts at `<think>` and runs through `<|im_end|>`
      # so the model learns to stop rather than run on into the next turn.
      pending += f"{IM_START}assistant\n"
      segs.append(Segment(pending, scored=False))
      segs.append(
        Segment(
          f"{THINK_OPEN}\n{turn.think}\n{THINK_CLOSE}\n"
          f"{RESPONSE_OPEN}{dumps_compact(list(turn.calls))}{RESPONSE_CLOSE}\n"
          f"{IM_END}",
          scored=True,
        )
      )
      pending = "\n"
    else: # pragma: no cover
      raise TypeError(f"unknown turn type: {type(turn).__name__}")

  if pending.strip():
    segs.append(Segment(pending, scored=False))
  return segs


def render(ex: Example) -> str:
  """The example as the model sees it. For eyeballing and for prompt-building."""
  return "".join(s.text for s in _segments(ex))


def render_prompt(ex: Example) -> str:
  """Everything up to the first assistant turn. What inference conditions on."""
  out: list[str] = []
  for s in _segments(ex):
    if s.scored:
      break
    out.append(s.text)
  return "".join(out)


def encode(ex: Example, tok, ignore_index: int = IGNORE_INDEX) -> tuple[list[int], list[int]]:
  """Tokenize and build the label mask.

  Returns `(input_ids, labels)`, where `labels[i]` is `input_ids[i]` inside an
  assistant turn and `ignore_index` everywhere else. The system, user, and
  tool turns, and the `<state>` / `<memory>` blocks, are context: the model
  conditions on them but is never graded on reproducing them. Grading them
  would be pure waste, since the server injects them at inference time.
  """
  ids: list[int] = []
  labels: list[int] = []
  for seg in _segments(ex):
    seg_ids = tok(seg.text, add_special_tokens=False)["input_ids"]
    ids.extend(seg_ids)
    labels.extend(seg_ids if seg.scored else [ignore_index] * len(seg_ids))
  return ids, labels


# --------------------------------------------------------------------------
# Parsing, for validation filters and for the inference server
# --------------------------------------------------------------------------
def _between(text: str, open_tag: str, close_tag: str) -> str | None:
  i = text.find(open_tag)
  if i == -1:
    return None
  j = text.find(close_tag, i + len(open_tag))
  if j == -1:
    return None
  return text[i + len(open_tag) : j]


def parse_think(text: str) -> str | None:
  body = _between(text, THINK_OPEN, THINK_CLOSE)
  return body.strip() if body is not None else None


def parse_calls(text: str) -> list[dict] | None:
  """Extract and parse the `<response>` array. `None` if absent or malformed.

  Returning `None` rather than raising is deliberate: the validation filter's
  single largest reject bucket is a closing-bracket mismatch from the teacher,
  and that is an expected outcome, not an exception.
  """
  body = _between(text, RESPONSE_OPEN, RESPONSE_CLOSE)
  if body is None:
    return None
  try:
    calls = json.loads(body.strip())
  except (json.JSONDecodeError, ValueError):
    return None
  if isinstance(calls, dict):
    calls = [calls]
  if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
    return None
  return calls
