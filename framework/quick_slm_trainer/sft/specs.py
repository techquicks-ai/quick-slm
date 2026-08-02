"""The five SFT categories, as data.

Each spec carries what the prompt builder needs to ask the teacher for one
example, and what the validator needs to decide whether it got one. Keeping both
in one object is what stops the two from disagreeing: a category whose teacher
prompt says "call exactly one tool" and whose validator permits three is a
category that silently teaches the wrong thing.

Shares and the category set are from `docs/sft_readme.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..config import SFTConfig


@dataclass(frozen=True)
class CategorySpec:
  key: str
  share: float
  domains: tuple[str, ...]
  subtypes: tuple[str, ...]

  #: Inclusive bounds on how many domain tools appear in `<tool>`, before
  #: `answer` is appended.
  n_tools: tuple[int, int]
  #: Inclusive bounds on the number of assistant turns. Two or more means the
  #: teacher must also invent the intervening tool results.
  n_assistant_turns: tuple[int, int]

  needs_state: bool = False
  needs_memory: bool = False

  #: The final assistant turn must call `answer` and nothing else. True for
  #: traps and refusals: a trap resolved by calling a real tool is not a trap,
  #: it is the teacher taking the easy way out, and the filter rejects it. True
  #: for multi-stage too, where the last turn delivers the chain's result in
  #: natural language rather than making a sixth call.
  final_call_must_be_answer: bool = False

  #: Samples are generated and accepted two at a time, as counterfactual pairs.
  #: Category 3 only. Paired categories bypass `validate.validate_record`
  #: entirely: their planning lives in `conflict.plan_pairs` and their accept
  #: decision in `conflict.validate_pair`, which keeps or drops both branches.
  #:
  #: There is deliberately no `think_must_mention` field. It once carried
  #: `("state", "memory")` here, and selecting on the wording of a reasoning
  #: block rewards narration over behaviour. Emptying the tuple would have left
  #: a filter that silently passes everything; deleting the field means a spec
  #: that tries to set it fails loudly.
  paired: bool = False

  guidance: str = ""
  subtype_guidance: dict[str, str] = field(default_factory=dict)

  #: Sub-types valid only in some domains. A `search_fallback` chain asked for
  #: in the factory domain, which has no `web_search`, is a request the teacher
  #: cannot satisfy and the validator will reject. Domains absent from this map
  #: get the full `subtypes` tuple.
  domain_subtypes: dict[str, tuple[str, ...]] = field(default_factory=dict)

  def subtypes_for(self, domain: str) -> tuple[str, ...]:
    return self.domain_subtypes.get(domain, self.subtypes)


SINGLE_STAGE = CategorySpec(
  key="single_stage",
  share=0.40,
  domains=("world", "factory"),
  subtypes=("direct", "paraphrased", "irrelevant_context", "optional_args"),
  n_tools=(3, 6),
  n_assistant_turns=(1, 1),
  guidance=(
    "The user asks for one thing. Exactly one assistant turn, calling exactly one tool, "
    "with correct arguments. The <think> block is one or two sentences: what the user wants, "
    "which tool fits, what arguments to bind."
  ),
  subtype_guidance={
    "direct": "Phrase the request plainly and literally, naming the thing the tool needs.",
    "paraphrased": (
      "Phrase the request indirectly. Do not use the tool's own words. "
      'Say "get me a dozen iron units" rather than "buy 12 iron ore".'
    ),
    "irrelevant_context": (
      "Open with a sentence of small talk or unrelated context before the real request. "
      "The <think> block should ignore the noise rather than comment on it at length."
    ),
    "optional_args": (
      "Choose a tool with optional parameters and have the user supply one of them. "
      "The call must bind the optional argument the user mentioned and omit the rest."
    ),
  },
)

MULTI_STAGE = CategorySpec(
  key="multi_stage",
  share=0.25,
  domains=("world", "factory"),
  subtypes=("chain_2", "chain_3", "chain_4", "verify_not_guess", "search_fallback"),
  n_tools=(4, 8),
  # A chain of N tool steps needs N assistant turns to make the calls plus one
  # more to deliver the result, so `chain_2` through `chain_4` span three to
  # five assistant turns. The bounds and the sub-type text have to agree, or
  # the validator rejects most of what the prompt asked for.
  n_assistant_turns=(3, 5),
  # The guidance below has always said "the final assistant turn calls `answer`",
  # and until this flag was set nothing checked it. This module's own docstring
  # names that failure: a category whose teacher prompt says one thing and whose
  # validator permits another is a category that silently teaches the wrong
  # thing. A chain that ends on its last tool call never learns to report.
  final_call_must_be_answer=True,
  guidance=(
    "The request needs several tools in sequence, where step N's arguments come from step "
    "N-1's result. Write the intervening tool turns yourself, as short realistic JSON results. "
    "Each <think> block must read the previous tool result before choosing the next call. "
    "The final assistant turn calls `answer` to deliver the result in natural language."
  ),
  subtype_guidance={
    "chain_2": "Two tool-calling assistant turns, then a final `answer` turn. Three in total.",
    "chain_3": "Three tool-calling assistant turns, then a final `answer` turn. Four in total.",
    "chain_4": "Four tool-calling assistant turns, then a final `answer` turn. Five in total.",
    "verify_not_guess": (
      "Two or three tool-calling assistant turns, then a final `answer` turn. The model "
      "plausibly knows the intermediate fact already and must look it up anyway. One <think> "
      "block says so explicitly: it will not guess when a tool can confirm."
    ),
    "search_fallback": (
      "Two or three tool-calling assistant turns, then a final `answer` turn. No specialised "
      "tool covers one of the steps, so `web_search` is used for it and its result feeds the "
      "next call."
    ),
  },
  # `web_search` is a world tool. Asking for a search fallback inside the
  # factory simulation asks for a tool that is not there.
  domain_subtypes={"factory": ("chain_2", "chain_3", "chain_4", "verify_not_guess")},
)

STATE_MEMORY_CONFLICT = CategorySpec(
  key="state_memory_conflict",
  share=0.20,
  domains=("factory",),
  subtypes=("stale_memory", "stale_state", "memory_missing_context"),
  # The scenario, the tools, the state, and the memory all come from a
  # `conflict.SwapSpec`, not from these bounds. They are kept because the
  # counting and reporting code reads them, and because a paired category that
  # emitted more than one assistant turn could not be compared to a
  # single-call oracle.
  n_tools=(3, 5),
  n_assistant_turns=(1, 1),
  needs_state=True,
  needs_memory=True,
  paired=True,
  guidance=(
    "The <state> block is the ground truth, sampled this turn. The <memory> block records what "
    "happened recently and may be out of date. When they disagree, the call must follow state. "
    "The teaching signal is behavioural: which call gets made, not whether the reasoning says "
    "so out loud."
  ),
  subtype_guidance={
    "stale_memory": (
      "The action in memory succeeded, but the world has moved on since: something else "
      "mutated the state."
    ),
    "stale_state": (
      "Memory records an action whose effect has not yet appeared in state. State still wins."
    ),
    "memory_missing_context": (
      "Memory does not contain what the user is asking about, while state does."
    ),
  },
)

TRAPS = CategorySpec(
  key="traps",
  share=0.10,
  domains=("world", "factory"),
  subtypes=("missing_argument", "no_relevant_tool", "tool_error"),
  n_tools=(2, 5),
  n_assistant_turns=(1, 2),
  final_call_must_be_answer=True,
  guidance=(
    "The situation is invalid and the model must notice rather than confabulate. It must never "
    "invent a tool that is not in the list, never fabricate an argument the user did not give, "
    "and never retry a failing call unchanged. It resolves the turn with `answer`."
  ),
  subtype_guidance={
    "missing_argument": (
      "A tool that fits exists, but the user omitted a required argument. Ask for it. "
      "Do not guess a plausible default."
    ),
    "no_relevant_tool": (
      "The request is coherent, but no tool in the list can serve it. Say what is available "
      "and decline the rest. One assistant turn."
    ),
    "tool_error": (
      "The first assistant turn makes a reasonable call. The tool turn returns an error or an "
      "empty result. The second assistant turn reports the problem instead of looping."
    ),
  },
)

REFUSALS = CategorySpec(
  key="refusals",
  share=0.05,
  domains=("factory", "world"),
  subtypes=("out_of_scope", "unsafe", "privacy_impersonation"),
  n_tools=(3, 6),
  n_assistant_turns=(1, 1),
  final_call_must_be_answer=True,
  guidance=(
    "The model could act, and should not. It declines in one short assistant turn. The <think> "
    "block names the reason in one sentence. The refusal itself is brief and does not lecture: "
    "a long moralising reply is itself a failure mode."
  ),
  subtype_guidance={
    "out_of_scope": "Coherent request, tools exist, but this deployment was not built for it.",
    "unsafe": (
      "Acting would cause harm. Decline before reasoning about how the thing would be done."
    ),
    "privacy_impersonation": (
      "The request asks the model to claim to be a person, give professional advice it is not "
      "qualified to give, or reveal what it was trained on."
    ),
  },
)


CATEGORIES: dict[str, CategorySpec] = {
  s.key: s
  for s in (SINGLE_STAGE, MULTI_STAGE, STATE_MEMORY_CONFLICT, TRAPS, REFUSALS)
}


def plan_counts(cfg: SFTConfig) -> dict[str, int]:
  """How many raw examples to request per category.

  The shares come from the config rather than from `CategorySpec.share`, so a
  targeted regeneration run can reweight without editing this module. They are
  checked against each other, because a config whose shares do not sum to one
  produces a corpus whose composition nobody intended.
  """
  unknown = set(cfg.category_shares) - set(CATEGORIES)
  if unknown:
    raise KeyError(f"config names categories that do not exist: {sorted(unknown)}")
  missing = set(CATEGORIES) - set(cfg.category_shares)
  if missing:
    raise KeyError(f"config omits categories: {sorted(missing)}")

  total = sum(cfg.category_shares.values())
  if abs(total - 1.0) > 1e-6:
    raise ValueError(f"category_shares sum to {total}, not 1.0")

  counts = {k: int(round(cfg.target_examples * v)) for k, v in cfg.category_shares.items()}

  # A paired category is counted in examples, and each counterfactual pair
  # contributes two. An odd count would plan half a pair, whose surviving
  # branch is exactly the isolated sample the construction exists to exclude.
  for key, spec in CATEGORIES.items():
    if spec.paired and counts[key] % 2:
      counts[key] += 1

  # Push the rounding residue onto the largest unpaired category, where it is a
  # rounding error rather than a visible share change. Correcting it against a
  # paired category would make the count odd again.
  residue = cfg.target_examples - sum(counts.values())
  if residue:
    unpaired = [k for k in counts if not CATEGORIES[k].paired]
    biggest = max(unpaired, key=lambda k: counts[k])
    counts[biggest] += residue
  return counts


def cycle(items: Sequence[str], i: int) -> str:
  """Deterministic round-robin over `items`.

  Each item takes its exact share only when `i` advances by one per draw *from
  this sequence*. Two cycles keyed on the same `i` do not compose that way: when
  their lengths share a factor the pairs they produce walk one common orbit, and
  every pair off it is unreachable. Keying a two-domain cycle and a four-subtype
  cycle on the same index reaches four of the eight combinations, not eight.

  `generate.plan_requests` therefore divides the outer length out of `i` before
  it indexes the inner sequence.
  """
  return items[i % len(items)]
