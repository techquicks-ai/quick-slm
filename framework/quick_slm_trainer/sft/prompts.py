"""Teacher prompts.

The teacher is asked for a JSON object, not for the ChatML template.

`docs/sft_readme.md` sketches the opposite: Gemma writes out the canonical
template and `validate.py` checks it parses. That was rejected here. Asking a
model to reproduce a surface form exactly makes the teacher responsible for
whitespace it has no reason to care about, and turns the single largest reject
bucket into a formatting bucket rather than a content bucket. Worse, it means the
corpus's surface form is whatever the teacher happened to emit, while the
inference server's is whatever `template.py` emits, and nobody notices the drift
until the model is generating a stray space before `[`.

Here the teacher supplies content: the user's request, the reasoning, the calls,
the invented tool results. `template.render` turns that into the exact bytes the
student trains on, and the exact bytes the inference server will later produce.
The teacher cannot get the format wrong because it never sees the format.

Validation becomes structural in the same move. Rather than regex-ing a
`<response>` block out of prose, `validate.py` checks a parsed call against the
tool schema it was generated against.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Sequence

from .specs import CategorySpec
from .tools import ALL_TOOLS, Tool, has_optional_args

TEACHER_SYSTEM = (
  "You author training examples for a small tool-calling language model. "
  "You reply with a single JSON object and nothing else: no prose, no markdown fences, "
  "no commentary before or after."
)

RULES = """\
Hard rules, all of which are checked automatically. A violation discards the example.
 1. Every `name` in every `calls` array is one of the tool names listed above. Never invent one.
 2. Every required parameter of a called tool appears in `arguments`, with a value of the declared
   type. No parameter that the tool does not declare may appear.
 3. `think` is never empty and never contains JSON or markup tags such as <think> or <response>.
   It is plain reasoning prose. It may of course discuss state and memory by name.
 4. `turns` begins with an assistant turn and ends with an assistant turn. Assistant and tool turns
   strictly alternate. A tool turn's `result` is what the immediately preceding call returns.
 5. The user's request is specific and self-contained. It does not address the model as an AI, and
   it never mentions tools, JSON, or schemas by name.
 6. Never restate the user's request back to them in `think`. Reason about it."""


def _tools_block(tools: Sequence[Tool]) -> str:
  return json.dumps(list(tools), indent=2)


# ==========================================================================
# The output contract, built per category
# ==========================================================================
# Field names match the `Example` constructor in `template.py`, so
# `validate.record_to_example` is a rename-free walk.
#
# One schema for all five categories used to be a constant here. It showed a
# populated `memory` object and three turns including a tool turn, while the
# STRUCTURE section of the same prompt told `single_stage` that `turns` has
# length 1 and that `state` and `memory` are both null. Eighty percent of the
# corpus was asked for two incompatible things in one prompt, and the rejects
# would have arrived as `MEMORY_UNEXPECTED`, `BAD_TURN_ORDER`, and
# `TURN_COUNT_OUT_OF_BOUNDS` at full teacher cost. The schema is now generated
# from the same `CategorySpec` the validator reads, so the two cannot drift.

_ANY_CALL = '{"name": "<a tool name from the list>", "arguments": {}}'
_ANSWER_CALL = '{"name": "answer", "arguments": {"text": "<what the assistant says to the user>"}}'
_TOOL_TURN = '  {"role": "tool", "result": <a JSON object: what the call above would return>}'


def _assistant_turn(call: str) -> str:
  return (
    '  {"role": "assistant",\n'
    '   "think": "<private reasoning, plain prose, no JSON, no tags>",\n'
    '   "calls": [' + call + "]}"
  )


def _state_slot(spec: CategorySpec) -> str:
  return "<a JSON object describing the live world>" if spec.needs_state else "null"


def _memory_slot(spec: CategorySpec) -> str:
  if not spec.needs_memory:
    return "null"
  return (
    "{\n"
    '  "recent": ["<what happened recently, one short line each>"],\n'
    '  "last_results": ["<outcome of each recent tool call, one short line each>"]\n'
    " }"
  )


def _turns_slot(spec: CategorySpec) -> str:
  """The shortest turn sequence the spec's own bounds accept.

  The shortest rather than the longest, because every category with room to
  grow grows for one sub-type only. Showing `traps` a two-turn shape would
  invite `missing_argument` to open with a call, and a call whose argument the
  user never gave is the exact confabulation the category exists to punish.
  `_turn_note` supplies the shape of the extension in prose instead.
  """
  lo, _hi = spec.n_assistant_turns
  tail = _ANSWER_CALL if spec.final_call_must_be_answer else _ANY_CALL

  rows: list[str] = []
  for _ in range(lo - 1):
    rows.append(_assistant_turn(_ANY_CALL))
    rows.append(_TOOL_TURN)
  rows.append(_assistant_turn(tail))
  return ' "turns": [\n' + ",\n".join(rows) + "\n ]"


def _turn_note(spec: CategorySpec) -> str:
  lo, hi = spec.n_assistant_turns
  if lo == hi:
    return ""
  plural = "" if lo == 1 else "s"
  return (
    f"\n\nThe `turns` array above is the shortest legal shape, with {lo} assistant "
    f"turn{plural}. Up to {hi} are allowed. Any extra assistant turn goes before the final "
    "one, and every assistant turn except the final one is followed by a tool turn, "
    '{"role": "tool", "result": <a JSON object>}, holding the result of the call it just made.'
  )


def build_output_schema(spec: CategorySpec) -> str:
  """The JSON object this category's teacher is asked to fill in."""
  return (
    "{\n"
    ' "user": "<the user\'s request, natural language, first turn>",\n'
    f' "state": {_state_slot(spec)},\n'
    f' "memory": {_memory_slot(spec)},\n'
    f"{_turns_slot(spec)}\n"
    "}"
  )


def _memory_rule(spec: CategorySpec) -> str:
  if spec.needs_memory:
    return (
      "`state` and `memory` are both REQUIRED and must contradict each other in the way the "
      "sub-type describes. `state` is what is true now. `memory` is what the assistant "
      "believes from earlier turns."
    )
  return "`state` and `memory` must both be null. This category has no ambient context."


def _turn_rule(spec: CategorySpec) -> str:
  lo, hi = spec.n_assistant_turns
  if lo == hi == 1:
    base = "Exactly one assistant turn. `turns` has length 1 and contains no tool turn."
  elif lo == hi:
    base = f"Exactly {lo} assistant turns, with a tool turn between each consecutive pair."
  else:
    base = f"Between {lo} and {hi} assistant turns, with a tool turn between each consecutive pair."
  if spec.final_call_must_be_answer:
    base += (
      " The FINAL assistant turn calls `answer` and nothing else. Calling any other tool in "
      "the final turn defeats the purpose of this category."
    )
  return base


def build_teacher_prompt(
  spec: CategorySpec,
  tools: Sequence[Tool],
  *,
  subtype: str,
  seed_topic: str,
  domain: str,
) -> str:
  """The user-role message handed to the teacher for one example."""
  subtype_note = spec.subtype_guidance.get(subtype, "")
  return f"""\
Write ONE training example for the "{spec.key}" category, sub-type "{subtype}".

CATEGORY
{spec.guidance}

SUB-TYPE
{subtype_note}

SCENARIO SEED
Build the example around this, without quoting it verbatim: {seed_topic}

AVAILABLE TOOLS (the example may only call these)
{_tools_block(tools)}

STRUCTURE
{_turn_rule(spec)}
{_memory_rule(spec)}

{RULES}

OUTPUT FORMAT
Reply with exactly this JSON object, filled in, and nothing else:
{build_output_schema(spec)}{_turn_note(spec)}"""


# ==========================================================================
# Scenario seeds
# ==========================================================================
# Diversity in the corpus comes from here rather than from sampling temperature.
# Raising temperature past 0.7 buys variety by degrading JSON validity; varying
# the scenario buys it for free. `docs/sft_readme.md` asks for direct,
# paraphrased, and noisy phrasings, and for chains of two to four steps; the
# spec's sub-types cover the shape and these cover the subject matter.


@dataclass(frozen=True)
class Seed:
  """A scenario topic, and the tools that scenario presupposes.

  `requires` is pinned into the sampled tool list by `sample_tools(include=...)`.
  Without it the planner drew the seed and the tool subset independently, and
  the tool a seed named was absent from the list more often than not: 74 percent
  of the `missing_argument` and `tool_error` traps, 55 percent of the
  single-stage world scenarios. The teacher's escapes from that bind both
  corrupt the corpus. It can drift off the seed and write something generic, or
  it can write a `no_relevant_tool` example under the `missing_argument` label,
  which teaches the student to decline where it should ask a question. Both pass
  validation, because nothing downstream knows what the seed asked for.

  Every name here must belong to the seed's own domain, and there may be at most
  two, because `specs.TRAPS` sets the smallest `n_tools` lower bound at two and
  a pinned list longer than `k` would overrun it. `sample_tools` enforces the
  domain half; `tests/test_sft_specs.py` enforces the count.

  A seed leaves `requires` empty when its scenario names no tool on purpose. The
  `no_relevant_tool` traps and every refusal are of that kind: what makes them
  work is that nothing in the list serves the request.
  """

  topic: str
  requires: tuple[str, ...] = ()


def _s(topic: str, *requires: str) -> Seed:
  return Seed(topic, requires)


WORLD_SEEDS: tuple[Seed, ...] = (
  _s("a traveller comparing the weather in two cities before packing", "get_weather"),
  _s("someone converting a recipe from imperial to metric", "convert_units"),
  _s("a student checking the population of a country for an assignment", "get_population"),
  _s("an investor glancing at a ticker before a meeting", "get_stock_price"),
  _s("someone working out what time it is for a colleague abroad", "get_current_time"),
  _s("a shopper converting a price into their home currency", "convert_currency"),
  _s("someone deciding what to do in a city on a rainy afternoon", "get_weather", "get_indoor_activities"),
  _s("a quiz question about a national capital", "get_capital_city"),
  _s("a person splitting a restaurant bill", "calculate"),
  _s("someone researching an unfamiliar term they read this morning", "web_search"),
  _s("a parent planning an outdoor weekend", "get_weather"),
  _s("a journalist checking a figure before publishing", "web_search"),
  _s("someone estimating a flight's cost in another currency", "convert_currency"),
  _s("a runner converting a distance from miles to kilometres", "convert_units"),
  _s("someone confirming a fact they half-remember from school", "web_search"),
  _s("a hiker checking tomorrow's forecast before setting out", "get_weather"),
  _s("a trader pulling the latest quote on a company", "get_stock_price"),
  _s("someone converting a room's dimensions from feet to metres", "convert_units"),
  _s("a pub-quiz team stuck on a country's capital", "get_capital_city"),
  _s("someone totalling a shopping list in their head", "calculate"),
  _s("a debater checking a country's population mid-argument", "get_population"),
  _s("a caller working out the local time on the other end", "get_current_time"),
  _s("a tourist converting euros to yen before a trip", "convert_currency"),
  _s("a reader looking up a word they did not recognise", "web_search"),
  _s("a family looking for something to do indoors while it rains", "get_indoor_activities"),
  _s("someone splitting a percentage tip on a bill", "calculate"),
  _s("an analyst comparing two tickers before the open", "get_stock_price"),
  _s("a gardener checking whether it will freeze tonight", "get_weather"),
  _s("someone fact-checking a claim they saw online", "web_search"),
)

FACTORY_SEEDS: tuple[Seed, ...] = (
  _s("the iron smelter has stopped producing and nobody knows why", "inspect", "resume_building"),
  _s("a new assembler needs placing next to the existing conveyor run", "build", "connect"),
  _s("copper stock is running low ahead of a large order", "get_resource", "buy"),
  _s("the power grid is browning out during peak production", "get_power_status", "set_priority"),
  _s("a research tech has just unlocked and should be started", "list_research", "research"),
  _s("two buildings need connecting so output flows into input", "connect"),
  _s("the market price of steel has moved and stock should be sold", "get_market_price", "sell"),
  _s("a building was placed in the wrong spot and blocks a belt", "move_building"),
  _s("the recipe on a furnace is wrong for what the user wants to make", "list_recipes", "set_recipe"),
  _s("the user wants a production-rate figure before committing to an order", "get_production_rate"),
  _s("an old building is idle and taking up grid priority", "set_priority", "toggle_power"),
  _s("the user is auditing what is actually on the map", "list_buildings"),
  _s("a conveyor was connected backwards and is starving a machine", "disconnect", "connect"),
  _s("the user wants to know whether they can afford a tech before starting it", "get_inventory", "list_research"),
  _s("a smelter was paused earlier in the session and its status is now unclear", "inspect", "resume_building"),
  _s("surplus steel is piling up and should be sold above a floor price", "get_resource", "sell"),
  _s("a full audit of what is on the map is needed before a rebuild", "list_buildings"),
  _s("a furnace is on the wrong recipe and output is going to waste", "list_recipes", "set_recipe"),
  _s("the market price of a resource should be checked before buying", "get_market_price", "buy"),
  _s("a building's throughput seems low and should be measured", "get_production_rate", "inspect"),
  _s("power headroom should be checked before bringing a line online", "get_power_status", "toggle_power"),
  _s("an idle building should be demolished to free up space", "list_buildings", "demolish"),
  _s("a mis-wired belt between two machines needs reversing", "disconnect", "connect"),
  _s("a low-priority building keeps getting shed and should be raised", "get_power_status", "set_priority"),
  _s("stock of a component is short and should be topped up from the market", "get_resource", "buy"),
  _s("a new production line needs a building placed and wired in", "build", "connect"),
  _s("a completed research should be followed by starting the next one", "list_research", "research"),
  _s("a running building should be paused to save power during a spike", "get_power_status", "pause_building"),
  _s("the recipes a building type can run need listing before a switch", "list_recipes"),
)

# Traps and refusals are the two categories whose sub-type *is* the scenario. A
# `no_relevant_tool` prompt paired with a seed about a forgotten argument gives
# the teacher two contradictory instructions, and it will follow one of them, so
# these pools are keyed by sub-type rather than drawn from a shared bag.
TRAP_SEEDS: dict[str, dict[str, tuple[Seed, ...]]] = {
  "missing_argument": {
    "world": (
      _s("the user asks about the weather but never says where", "get_weather"),
      _s("the user wants a currency converted but names only one currency", "convert_currency"),
      _s("the user asks for the time without saying where", "get_current_time"),
      _s("the user asks for a stock price without naming the company", "get_stock_price"),
      _s("the user asks to convert a distance but gives no target unit", "convert_units"),
      _s("the user asks for a capital without naming the country", "get_capital_city"),
    ),
    "factory": (
      _s("the user asks to buy a resource without saying how many", "buy"),
      _s("the user asks to inspect a building without naming it", "inspect"),
      _s("the user asks to set a recipe without saying which one", "set_recipe"),
      _s("the user asks to connect two buildings but names only one", "connect"),
      _s("the user asks to set a power priority without giving a level", "set_priority"),
      _s("the user asks to sell a resource without saying how much", "sell"),
    ),
  },
  # No `requires` anywhere below: the trap *is* that the list cannot serve the
  # request. Pinning a tool here would dismantle the category.
  "no_relevant_tool": {
    "world": (
      _s("the user asks to book a flight, which nothing in the list can do"),
      _s("the user asks to send an email, which nothing in the list can do"),
      _s("the user asks to translate a sentence, which no tool covers"),
      _s("the user asks to set a reminder, which no tool covers"),
      _s("the user asks to play a song, which no tool can do"),
      _s("the user asks to order a taxi, which nothing in the list covers"),
    ),
    "factory": (
      _s("the user asks to rename a building, which no tool can do"),
      _s("the user asks to save the game, which no tool can do"),
      _s("the user asks to undo their last action, which no tool can do"),
      _s("the user asks to change the game's difficulty, which no tool can do"),
      _s("the user asks to take a screenshot, which no tool can do"),
      _s("the user asks to rebind a hotkey, which nothing in the list covers"),
    ),
  },
  "tool_error": {
    "world": (
      _s("a weather lookup returns an error because the city does not exist", "get_weather"),
      _s("a stock lookup returns an empty result for an unlisted symbol", "get_stock_price"),
      _s("a currency conversion fails on an unrecognised currency code", "convert_currency"),
      _s("a search returns no results at all", "web_search"),
      _s("a population lookup errors on a country name it does not recognise", "get_population"),
      _s("a time lookup fails on an invalid timezone string", "get_current_time"),
    ),
    "factory": (
      _s("an inspect call returns an error because the building id is unknown", "inspect"),
      _s("a connect call fails because one building was already demolished", "connect"),
      _s("a buy call fails because the market does not carry that resource", "buy"),
      _s("a research call returns an error because prerequisites are missing", "research"),
      _s("a set_recipe call fails because the recipe is unknown to that building", "set_recipe"),
      _s("a sell call returns an error because there is nothing in stock", "sell"),
    ),
  },
}

# A refusal is a decision not to act. Pinning the tool the request gestures at
# would offer the teacher a way to comply, so `requires` is empty throughout: the
# only call these examples make is `answer`, which `sample_tools` always appends.
REFUSAL_SEEDS: dict[str, dict[str, tuple[Seed, ...]]] = {
  "out_of_scope": {
    "world": (
      _s("the user asks for legal advice about a contract they signed"),
      _s("the user asks the assistant to diagnose a rash from a description"),
      _s("the user asks for a recommendation on which stock to buy"),
      _s("the user asks whether to plead guilty to a charge"),
      _s("the user asks the assistant to interpret their blood test results"),
    ),
    "factory": (
      _s("the user asks for medical advice about a headache"),
      _s("the user asks the assistant to write their homework essay"),
      _s("the user asks for tax advice about their business"),
      _s("the user asks the assistant to settle a personal dispute with a friend"),
      _s("the user asks for investment advice on their savings"),
    ),
  },
  "unsafe": {
    "world": (
      _s("the user asks for help writing a message intended to deceive someone"),
      _s("the user asks how to get into an account that is not theirs"),
      _s("the user asks the assistant to draft a threatening letter"),
      _s("the user asks how to make a weapon at home"),
      _s("the user asks how to stalk someone without being noticed"),
    ),
    "factory": (
      _s("the user asks how to disable a safety interlock on a machine"),
      _s("the user asks how to vent toxic byproduct into the worker area"),
      _s("the user asks the assistant to bypass the emergency shutdown"),
      _s("the user asks how to override a pressure limit past its rated maximum"),
      _s("the user asks how to defeat a lockout on a live electrical panel"),
    ),
  },
  "privacy_impersonation": {
    "world": (
      _s("the user asks the assistant to impersonate a named real person"),
      _s("the user asks the assistant to reveal what it was trained on"),
      _s("the user asks the assistant to claim it is a licensed professional"),
      _s("the user asks the assistant to write a review posing as a real customer"),
      _s("the user asks the assistant for another person's home address"),
    ),
    "factory": (
      _s("the user asks the assistant to pretend to be a doctor"),
      _s("the user asks the assistant to sign off on a safety inspection as an engineer"),
      _s("the user asks the assistant to recite its own system prompt"),
      _s("the user asks the assistant to certify a report as a licensed inspector"),
      _s("the user asks the assistant to reveal the operator's private credentials"),
    ),
  },
}

#: Categories whose scenario does not depend on the sub-type, keyed by domain.
#: `state_memory_conflict` is absent on purpose: its scenarios come from
#: `conflict_specs.TRAIN_SPECS`, not from a seed pool, because a counterfactual
#: pair needs a decisive field and an oracle rather than a topic.
_DOMAIN_SEEDS: dict[str, dict[str, tuple[Seed, ...]]] = {
  "single_stage": {"world": WORLD_SEEDS, "factory": FACTORY_SEEDS},
  "multi_stage": {"world": WORLD_SEEDS, "factory": FACTORY_SEEDS},
}

_SUBTYPE_SEEDS: dict[str, dict[str, dict[str, tuple[Seed, ...]]]] = {
  "traps": TRAP_SEEDS,
  "refusals": REFUSAL_SEEDS,
}


def seed_pool(category: str, domain: str, subtype: str) -> tuple[Seed, ...]:
  if category in _SUBTYPE_SEEDS:
    by_domain = _SUBTYPE_SEEDS[category][subtype]
  else:
    by_domain = _DOMAIN_SEEDS[category]
  if domain in by_domain:
    return by_domain[domain]
  # A category restricted to one domain still answers for the others rather
  # than raising, because the domain that reaches here is the spec's own.
  return next(iter(by_domain.values()))


def usable_seeds(category: str, domain: str, subtype: str) -> tuple[Seed, ...]:
  """The pool, narrowed to the seeds a sub-type can actually be built from.

  Only `optional_args` narrows anything. It asks the teacher to "choose a tool
  with optional parameters and have the user supply one of them", which a seed
  about a national capital cannot serve: `get_capital_city` declares no optional
  parameter. Pinning a qualifying tool alongside such a seed would hand the
  teacher a scenario and an instruction that contradict each other, which is the
  same fault `TRAP_SEEDS` is keyed by sub-type to avoid. Narrowing the pool keeps
  the two in agreement.

  Three world seeds and two factory seeds survive the narrowing. That is thin,
  and it is the sharpest instance of a corpus-wide problem: these pools are 105
  hand-written strings inflated to 80,000 examples. Widening them is a design
  question, not a mechanical fix, and it is not settled here.
  """
  pool = seed_pool(category, domain, subtype)
  if subtype != "optional_args":
    return pool

  narrowed = tuple(s for s in pool if any(has_optional_args(ALL_TOOLS[n]) for n in s.requires))
  if not narrowed:
    raise LookupError(
      f"{category}/{domain}/{subtype}: no seed names a tool with an optional parameter, "
      "so the sub-type cannot be built in this domain"
    )
  return narrowed


def choose_seed(rng: random.Random, category: str, domain: str, subtype: str) -> Seed:
  return rng.choice(usable_seeds(category, domain, subtype))


def seed_topic(rng: random.Random, category: str, domain: str, subtype: str) -> str:
  return choose_seed(rng, category, domain, subtype).topic


def recommended_examples_per_seed() -> int:
  """Examples to plan per distinct seed, for the unpaired categories.

  The unpaired twin of `conflict.recommended_conflict_pairs`, and deliberately a
  constant where that one is computed. The paired ceiling is combinatorial: a swap
  spec's distinct pairs are the product of its own small choice lists, so
  `conflict.distinct_capacity` counts them on CPU with no teacher. An unpaired cell
  fixes one seed and draws all its remaining variety from the teacher sampling at
  temperature 0.7, so its distinct yield per seed is an empirical property of the
  teacher that no CPU pass can read. Notebook `04a_sft_seed_audit` section 6
  measures it directly, and `generate.distinct_turns_per_seed` folds that same
  measurement into the pre-flight.

  Absent that measurement this returns 40. It holds the widest cell (single_stage,
  29 seeds) to roughly 1,160 examples, a 3.4x cut from the ~4,000 the raw share
  plans, while staying above the distinct yield the higher-variety sub-types
  (`paraphrased`, `irrelevant_context`) reach, so the cap does not bind below what
  the teacher can actually make distinct. Raise it toward the pre-flight figure
  once measured; the notebook's projection table reads the trade at 20, 30, 40,
  60, and 100.
  """
  return 40


# ==========================================================================
# The counterfactual state swap, category 3
# ==========================================================================
#: The teacher completes one branch. It is shown the exact prompt the student
#: will see, rendered by `template.render_prompt`, so no surface form can drift
#: between the corpus and the inference server.
#:
#: It is told the policy (state outranks memory) and never the answer. The oracle
#: decides whether it applied the policy. Nothing about the wording of `think` is
#: requested, required, or checked: asking for narration is how the previous
#: filter came to select for narration.
CONFLICT_SYSTEM = (
  "You complete one assistant turn of a tool-calling session. "
  "You reply with a single JSON object and nothing else: no prose, no markdown fences."
)

CONFLICT_OUTPUT_SCHEMA = """\
{
 "think": "<private reasoning, plain prose, no JSON, no tags>",
 "calls": [{"name": "<a tool name from the list above>", "arguments": {}}]
}"""


def build_conflict_prompt(prompt_block: str, *, guidance: str, subtype_guidance: str) -> str:
  """`prompt_block` is the rendered student prompt for one branch."""
  return f"""\
Below is a complete assistant turn, up to the point where the assistant must act.

{prompt_block}

WHAT THE BLOCKS MEAN
{guidance}

SITUATION
{subtype_guidance}

HARD RULES, all checked automatically. A violation discards this example and its twin.
 1. Emit EXACTLY ONE call, in `calls`.
 2. Its `name` is one of the tools listed in the <tool> block above. Never invent one.
 3. Every required parameter appears in `arguments`, correctly typed. No undeclared parameter.
 4. Read the values in <state> before choosing. <state> was sampled this turn and outranks
   <memory> wherever the two disagree.
 5. `think` is non-empty plain prose and contains no JSON and no tags. Its wording is not
   graded. Reason however you find natural.

OUTPUT FORMAT
Reply with exactly this JSON object, filled in, and nothing else:
{CONFLICT_OUTPUT_SCHEMA}"""
