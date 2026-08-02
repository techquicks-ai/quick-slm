"""Deterministic `correct_call(state, request)` per tool family, and the
canonical form the teacher's call is compared against.

The teacher never supplies ground truth. It sees the tools, the `<state>`, the
`<memory>`, and the user turn, and it emits a call. The oracle computes what the
call should have been, from the state alone. Agreement is the accept criterion.

`request` is a dict, not a bare string. It carries `text`, the user turn, plus
whatever the scenario already knew when it wrote that turn: which building the
user meant, which recipe they named, what quantity they asked for. A real
dispatcher has exactly this, because it is what produced the utterance. Handing
it to the oracle avoids re-parsing English inside a function whose whole value is
that it is deterministic. The same `request` dict goes to both branches of a
pair, so it carries no information about which branch is which.

Canonical comparison, and why `answer` is special
-------------------------------------------------
Two calls are compared on `{name, arguments}` after canonical key ordering. The
`text` argument of `answer` is excluded, because no teacher will reproduce a
hand-written sentence verbatim and demanding it would reject every correct
refusal.

That exclusion is what gives criterion 1 its teeth. `oracle(S_a) != oracle(S_b)`
is evaluated under the *same* canonical form, so a spec whose two branches both
resolve to `answer` compares equal and the spec is rejected. A spec only survives
when flipping the decisive field changes the tool called, or changes a structured
argument. Free prose can never be the thing that differs.
"""

from __future__ import annotations

from typing import Any, Callable

Call = dict[str, Any]
Request = dict[str, Any]
Oracle = Callable[[dict, Request], Call]

#: Arguments excluded from structural comparison, per tool. Adding a tool here
#: makes that argument unverifiable, so the list stays as short as it can be.
FREE_TEXT_ARGS: dict[str, frozenset[str]] = {"answer": frozenset({"text"})}


def _canon(value: Any) -> Any:
  if isinstance(value, bool):
    # Tagged, not returned bare. `False == 0` and `True == 1` in Python, so a
    # bare boolean would compare equal to the number the next branch produces,
    # and `toggle_power(on=False)` would be satisfied by `on=0`.
    return ("bool", value)
  if isinstance(value, (int, float)):
    # 500 and 500.0 are the same quantity. A teacher that writes one where
    # the oracle wrote the other has not made an error.
    return float(value)
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, dict):
    return tuple(sorted((k, _canon(v)) for k, v in value.items()))
  if isinstance(value, (list, tuple)):
    return tuple(_canon(v) for v in value)
  return value


def canonical_call(call: Call) -> tuple:
  """A hashable, order-insensitive form of `{name, arguments}`."""
  name = call.get("name")
  free = FREE_TEXT_ARGS.get(name, frozenset())
  args = call.get("arguments") or {}
  return (name, tuple(sorted((k, _canon(v)) for k, v in args.items() if k not in free)))


def calls_agree(a: Call, b: Call) -> bool:
  return canonical_call(a) == canonical_call(b)


def _answer(text: str) -> Call:
  return {"name": "answer", "arguments": {"text": text}}


# ==========================================================================
# Factory families
# ==========================================================================
def building_run(state: dict, request: Request) -> Call:
  """`building_control`: ensure a building is running."""
  b = state["buildings"][0]
  if b["paused"]:
    return {"name": "resume_building", "arguments": {"building_id": b["id"]}}
  return _answer(f"The {b['type']} ({b['id']}) is already running.")


def building_stop(state: dict, request: Request) -> Call:
  """`building_control`: ensure a building is halted."""
  b = state["buildings"][0]
  if b["paused"]:
    return _answer(f"The {b['type']} ({b['id']}) is already paused.")
  return {"name": "pause_building", "arguments": {"building_id": b["id"]}}


def inventory_topup(state: dict, request: Request) -> Call:
  """`inventory`: buy the shortfall against a target, or say there is none.

  The decisive field is the stock on hand, and it sets the *quantity*, not only
  the tool. A model that pattern-matches the user turn can guess `buy`; it
  cannot guess how many.
  """
  line = state["inventory"][0]
  target = state["order"]["target"]
  have = line["quantity"]
  if have >= target:
    return _answer(f"{line['resource_id']} is already at {have}, at or above the target of {target}.")
  return {
    "name": "buy",
    "arguments": {"resource_id": line["resource_id"], "quantity": target - have},
  }


def production_recipe(state: dict, request: Request) -> Call:
  """`production`: set a building's recipe to the one the user named."""
  b = state["buildings"][0]
  wanted = request["recipe_id"]
  if b["recipe_id"] == wanted:
    return _answer(f"The {b['type']} ({b['id']}) is already running {wanted}.")
  return {"name": "set_recipe", "arguments": {"building_id": b["id"], "recipe_id": wanted}}


def power_shed(state: dict, request: Request) -> Call:
  """`power`: drop a building from the grid when consumption exceeds generation."""
  grid = state["grid"]
  b = state["buildings"][0]
  if grid["consumption"] > grid["generation"]:
    return {"name": "toggle_power", "arguments": {"building_id": b["id"], "on": False}}
  return _answer(
    f"The grid draws {grid['consumption']} against {grid['generation']} generated, so nothing needs shedding."
  )


def power_priority(state: dict, request: Request) -> Call:
  """`power`: raise a building to the priority the user asked for."""
  b = state["buildings"][0]
  wanted = request["priority"]
  if b["priority"] == wanted:
    return _answer(f"The {b['type']} ({b['id']}) is already at priority {wanted}.")
  return {"name": "set_priority", "arguments": {"building_id": b["id"], "priority": wanted}}


def connection_link(state: dict, request: Request) -> Call:
  """`connection`: ensure a conveyor runs between two buildings."""
  link = state["link"]
  if link["connected"]:
    return _answer(f"{link['from_id']} already feeds {link['to_id']}.")
  return {"name": "connect", "arguments": {"from_id": link["from_id"], "to_id": link["to_id"]}}


def market_selldown(state: dict, request: Request) -> Call:
  """`market`: sell the surplus above a floor, or say there is none.

  Argument-only, like `inventory_topup` and for the same reason: both branches
  could plausibly call `sell`, and the quantity is `have - floor`, a number that
  exists only in `<state>`. A model that guesses the tool from the user turn
  still cannot fill the argument without reading state. `floor` is randomised by
  the scenario, so the sold quantity is not one of a handful of constants.
  """
  line = state["inventory"][0]
  floor = state["order"]["floor"]
  have = line["quantity"]
  if have <= floor:
    return _answer(f"{line['resource_id']} is at {have}, at or below the floor of {floor}.")
  return {"name": "sell", "arguments": {"resource_id": line["resource_id"], "quantity": have - floor}}


# ==========================================================================
# Held-out families. Present in no training spec; see `conflict.EVAL_SPECS`.
# ==========================================================================
def research_start(state: dict, request: Request) -> Call:
  """`research`: begin a technology unless it is already the active one."""
  active = state["research"]["active"]
  wanted = request["tech_id"]
  if active == wanted:
    return _answer(f"{wanted} is already being researched.")
  return {"name": "research", "arguments": {"tech_id": wanted}}


def weather_cache(state: dict, request: Request) -> Call:
  """`weather`: answer from a fresh cache, refetch a stale one.

  The stub world-knowledge oracle the design doc asks for. It exists so that
  the evaluation set contains a tool family, and a domain, that appear in no
  training spec.
  """
  cache = state["cache"]
  if cache["age_minutes"] <= state["policy"]["max_age_minutes"]:
    return _answer(f"It is {cache['temp_c']} C in {cache['city']}.")
  return {"name": "get_weather", "arguments": {"city": cache["city"]}}


def stock_cache(state: dict, request: Request) -> Call:
  """`stocks`: answer from a fresh quote, refetch a stale one.

  A second held-out world oracle, so the evaluation is not a single scenario.
  Same staleness shape as `weather_cache`, different family and state schema.
  """
  quote = state["quote"]
  if quote["age_minutes"] <= state["policy"]["max_age_minutes"]:
    return _answer(f"{quote['symbol']} is at {quote['price']}.")
  return {"name": "get_stock_price", "arguments": {"symbol": quote["symbol"]}}


ORACLES: dict[str, Oracle] = {
  "building_run": building_run,
  "building_stop": building_stop,
  "inventory_topup": inventory_topup,
  "production_recipe": production_recipe,
  "power_shed": power_shed,
  "power_priority": power_priority,
  "connection_link": connection_link,
  "market_selldown": market_selldown,
  "research_start": research_start,
  "weather_cache": weather_cache,
  "stock_cache": stock_cache,
}
