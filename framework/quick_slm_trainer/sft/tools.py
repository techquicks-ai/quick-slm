"""The tool universe the SFT corpus is generated against.

`docs/sft_readme.md` left this open ("the SFT corpus needs a fixed set of
~30 tools"). It is fixed here, in Python rather than YAML, so that a schema
error is a test failure and not a runtime surprise eleven hours into teacher
generation.

Two domains, plus one tool that belongs to both.

`answer` is the escape hatch and the single most important schema in the file.
Every trap and every refusal resolves to `answer`, so a model that has never
seen it has no way to decline: its only options are to invoke a tool it should
not, or to invent one. It is therefore appended to every tool subset, always.

The tools are stubs. No implementation exists behind them and none is needed:
SFT teaches the model to *emit* a well-formed call, and the tool turns in the
corpus are synthetic results written by the teacher. What matters is that the
schemas are internally consistent, because the validation filter checks every
generated call against them.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

Tool = dict[str, Any]


def _tool(name: str, description: str, properties: dict, required: Sequence[str]) -> Tool:
  unknown = set(required) - set(properties)
  if unknown:
    raise ValueError(f"tool {name!r} requires undeclared properties: {sorted(unknown)}")
  return {
    "name": name,
    "description": description,
    "parameters": {
      "type": "object",
      "properties": properties,
      "required": list(required),
    },
  }


_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}


# ==========================================================================
# Universal
# ==========================================================================
ANSWER = _tool(
  "answer",
  "Reply to the user in natural language. Use this when no other tool applies, "
  "when a required argument is missing, when a tool has failed, when the request "
  "is out of scope, or to deliver a final result.",
  {"text": _STR},
  ["text"],
)


# ==========================================================================
# World-knowledge stubs
# ==========================================================================
WORLD_TOOLS: list[Tool] = [
  # `units` and `max_results` are optional on purpose. Without at least a few
  # optional parameters in the universe, the `optional_args` sub-type of the
  # single-stage category has nothing to exercise, and the model never learns
  # that an unmentioned optional field should be left out rather than guessed.
  _tool(
    "get_weather",
    "Current weather for a city. `units` defaults to celsius.",
    {"city": _STR, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
    ["city"],
  ),
  _tool("get_capital_city", "Capital city of a country.", {"country": _STR}, ["country"]),
  _tool(
    "web_search",
    "Search the open web. `max_results` defaults to 5.",
    {"query": _STR, "max_results": _INT},
    ["query"],
  ),
  _tool(
    "get_current_time",
    "Current local time in an IANA timezone.",
    {"timezone": _STR},
    ["timezone"],
  ),
  _tool(
    "convert_currency",
    "Convert an amount between two ISO-4217 currency codes.",
    {"amount": _NUM, "from_currency": _STR, "to_currency": _STR},
    ["amount", "from_currency", "to_currency"],
  ),
  _tool(
    "convert_units",
    "Convert a value between two units of the same physical quantity.",
    {"value": _NUM, "from_unit": _STR, "to_unit": _STR},
    ["value", "from_unit", "to_unit"],
  ),
  _tool("get_stock_price", "Latest price for a ticker symbol.", {"symbol": _STR}, ["symbol"]),
  _tool(
    "calculate",
    "Evaluate an arithmetic expression.",
    {"expression": _STR},
    ["expression"],
  ),
  _tool(
    "get_indoor_activities",
    "Suggested indoor activities in a city.",
    {"city": _STR},
    ["city"],
  ),
  _tool("get_population", "Population of a country.", {"country": _STR}, ["country"]),
]


# ==========================================================================
# Factory simulation, the demo environment
# ==========================================================================
FACTORY_TOOLS: list[Tool] = [
  _tool(
    "list_buildings",
    "List buildings on the map, optionally filtered to one type.",
    {"building_type": _STR},
    [],
  ),
  _tool("inspect", "Read a building's current state.", {"building_id": _STR}, ["building_id"]),
  _tool(
    "build",
    "Place a new building at map coordinates.",
    {"building_type": _STR, "x": _INT, "y": _INT},
    ["building_type", "x", "y"],
  ),
  _tool("demolish", "Remove a building.", {"building_id": _STR}, ["building_id"]),
  _tool(
    "move_building",
    "Relocate a building to new coordinates.",
    {"building_id": _STR, "x": _INT, "y": _INT},
    ["building_id", "x", "y"],
  ),
  _tool(
    "set_recipe",
    "Assign a production recipe to a building.",
    {"building_id": _STR, "recipe_id": _STR},
    ["building_id", "recipe_id"],
  ),
  _tool("pause_building", "Halt production at a building.", {"building_id": _STR}, ["building_id"]),
  _tool("resume_building", "Resume a paused building.", {"building_id": _STR}, ["building_id"]),
  _tool(
    "connect",
    "Run a conveyor from one building's output to another's input.",
    {"from_id": _STR, "to_id": _STR},
    ["from_id", "to_id"],
  ),
  _tool(
    "disconnect",
    "Remove the conveyor between two buildings.",
    {"from_id": _STR, "to_id": _STR},
    ["from_id", "to_id"],
  ),
  _tool("get_inventory", "Every resource currently in storage.", {}, []),
  _tool("get_resource", "Stored quantity of one resource.", {"resource_id": _STR}, ["resource_id"]),
  _tool(
    "buy",
    "Purchase a resource from the market.",
    {"resource_id": _STR, "quantity": _INT},
    ["resource_id", "quantity"],
  ),
  _tool(
    "sell",
    "Sell a resource on the market. `min_price` sets a floor below which the sale is skipped.",
    {"resource_id": _STR, "quantity": _INT, "min_price": _NUM},
    ["resource_id", "quantity"],
  ),
  _tool("get_market_price", "Current market price of a resource.", {"resource_id": _STR}, ["resource_id"]),
  _tool("research", "Begin researching a technology.", {"tech_id": _STR}, ["tech_id"]),
  _tool("list_research", "Available and completed technologies.", {}, []),
  _tool(
    "set_priority",
    "Set a building's power priority from 1 (lowest) to 5 (highest).",
    {"building_id": _STR, "priority": _INT},
    ["building_id", "priority"],
  ),
  _tool("get_power_status", "Grid generation, consumption, and headroom.", {}, []),
  _tool(
    "toggle_power",
    "Connect or disconnect a building from the power grid.",
    {"building_id": _STR, "on": _BOOL},
    ["building_id", "on"],
  ),
  _tool(
    "get_production_rate",
    "Output rate of a building, in units per minute.",
    {"building_id": _STR},
    ["building_id"],
  ),
  _tool(
    "list_recipes",
    "Recipes a building type can run.",
    {"building_type": _STR},
    ["building_type"],
  ),
]


DOMAIN_TOOLS: dict[str, list[Tool]] = {
  "world": WORLD_TOOLS,
  "factory": FACTORY_TOOLS,
}

ALL_TOOLS: dict[str, Tool] = {t["name"]: t for t in [ANSWER, *WORLD_TOOLS, *FACTORY_TOOLS]}

#: A system prompt per domain. The factory line is what teaches the model that a
#: medical or legal question is out of scope, so refusals need it verbatim.
DOMAIN_SYSTEM: dict[str, str] = {
  "world": "You are a helpful assistant with access to these tools:",
  "factory": "You are managing a factory simulation. You have access to these tools:",
}


# --------------------------------------------------------------------------
# Accessors used by the validator
# --------------------------------------------------------------------------
def properties_of(tool: Tool) -> dict:
  return tool["parameters"]["properties"]


def required_args(tool: Tool) -> set[str]:
  return set(tool["parameters"]["required"])


def optional_args(tool: Tool) -> set[str]:
  return set(properties_of(tool)) - required_args(tool)


def has_optional_args(tool: Tool) -> bool:
  """Whether `optional_args`, the single-stage sub-type, can be built around this tool.

  Four tools in the universe qualify: `get_weather`, `web_search`,
  `list_buildings`, and `sell`. A sub-type that asks the user to supply one
  optional parameter and omit the rest needs a tool that declares some, and the
  seed the example is built around has to be about that tool. `prompts.seed_pool`
  filters on this rather than pinning an arbitrary qualifying tool into the
  list, because a seed about splitting a restaurant bill and an instruction to
  bind an optional argument are two instructions the teacher cannot both follow.
  """
  return bool(optional_args(tool))


def tool_names(tools: Iterable[Tool]) -> set[str]:
  return {t["name"] for t in tools}


def by_name(tools: Iterable[Tool]) -> dict[str, Tool]:
  return {t["name"]: t for t in tools}


def sample_tools(
  rng: random.Random,
  domain: str,
  k: int,
  *,
  include: Sequence[str] = (),
  exclude: Sequence[str] = (),
) -> list[Tool]:
  """A `k`-tool subset of `domain`, plus `answer`, in a shuffled order.

  Order is shuffled because a model shown the correct tool first every time
  learns position, not selection. `include` pins tools a category needs: a
  multi-stage chain that must be *possible* needs both of its links present,
  and a scenario seed that names a tool needs that tool on offer. `exclude` is
  how the no-relevant-tool trap is built: the tool the request obviously calls
  for is removed from the list before the teacher sees it.
  """
  if domain not in DOMAIN_TOOLS:
    raise KeyError(f"unknown domain {domain!r}; known: {sorted(DOMAIN_TOOLS)}")

  # A world tool pinned into a factory list would be offered to the teacher and
  # then rejected by the validator, which checks the call against the tools that
  # were on offer. Catching it here makes a mistyped `requires` a test failure
  # rather than a reject bucket nobody reads until the run is over. `answer` is
  # rejected too: it is appended below, and pinning it would duplicate it.
  domain_names = {t["name"] for t in DOMAIN_TOOLS[domain]}
  foreign = sorted(set(include) - domain_names)
  if foreign:
    raise KeyError(f"cannot pin {foreign} into domain {domain!r}")

  banned = set(exclude)
  pinned = [ALL_TOOLS[n] for n in include if n not in banned]
  pinned_names = tool_names(pinned)

  pool = [t for t in DOMAIN_TOOLS[domain] if t["name"] not in banned and t["name"] not in pinned_names]
  k_extra = max(0, k - len(pinned))
  chosen = pinned + rng.sample(pool, min(k_extra, len(pool)))

  rng.shuffle(chosen)
  # `answer` is appended after the shuffle rather than mixed into it. Its
  # position is then constant, which is fine: the model never has to *find*
  # it, only to decide whether to use it.
  return [*chosen, ANSWER]
