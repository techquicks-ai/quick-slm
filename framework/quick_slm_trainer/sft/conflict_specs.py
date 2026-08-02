"""The category-3 spec table, partitioned into train and eval before generation.

Each spec is a scenario generator plus a decisive field. `scenario(rng)` randomises
the incidentals: building ids, building types, resources, quantities, and the
phrasing of the user turn. It never touches the decisive field's meaning, because
`build_pair` overwrites that field in both branches from `variants`.

`<memory>` is written to assert one branch's world. Under the other branch it is
stale, which is what puts state and memory in conflict. `stale_under` records
which. Nothing in the pipeline reads it; it exists so the claim can be checked.

Adding capacity to category 3 means adding specs here, not raising a count. Two
instances of one spec share everything but a handful of randomised ids, and the
dedup pass will collapse them.
"""

from __future__ import annotations

import random

from .conflict import Scenario, SwapSpec

_BUILDING_TYPES = ("smelter", "furnace", "assembler", "miner", "refinery", "press")
_RESOURCES = ("iron_ore", "copper_ore", "coal", "steel_plate", "circuit", "gear")
_RECIPES = ("smelt_iron", "smelt_copper", "cast_plate", "press_gear")


def _bid(rng: random.Random) -> str:
  return f"b_{rng.randint(1, 999):03d}"


# ==========================================================================
# Train: factory families
# ==========================================================================
def _building_run(rng: random.Random) -> Scenario:
  bid, btype = _bid(rng), rng.choice(_BUILDING_TYPES)
  state = {
    "buildings": [
      {
        "id": bid,
        "type": btype,
        "paused": False, # overwritten in both branches
        "recipe_id": rng.choice(_RECIPES),
        "progress": round(rng.uniform(0.05, 0.95), 2),
      }
    ]
  }
  memory = {
    "recent": [f"user: 'pause the {btype}' -> pause_building"],
    "last_results": [f"pause_building {bid} OK"],
  }
  text = rng.choice(
    (
      f"Make sure the {btype} is running.",
      f"I need the {btype} producing again.",
      f"Get the {btype} back online if it has stopped.",
      f"The {btype} should be running right now, sort it out.",
      f"Bring the {btype} back up unless it is already up.",
      f"Can you get output flowing from the {btype} again?",
      f"Kick the {btype} back into production if it stalled.",
      f"The {btype} needs to be live. Check it and fix it if not.",
      f"Confirm the {btype} is producing, and start it if it is idle.",
      f"See that the {btype} is turning over, not sitting stopped.",
    )
  )
  return Scenario(state, memory, {"text": text, "building_id": bid})


def _building_stop(rng: random.Random) -> Scenario:
  bid, btype = _bid(rng), rng.choice(_BUILDING_TYPES)
  state = {
    "buildings": [
      {
        "id": bid,
        "type": btype,
        "paused": False,
        "recipe_id": rng.choice(_RECIPES),
        "progress": round(rng.uniform(0.05, 0.95), 2),
      }
    ]
  }
  memory = {
    "recent": [f"user: 'restart the {btype}' -> resume_building"],
    "last_results": [f"resume_building {bid} OK"],
  }
  text = rng.choice(
    (
      f"Stop the {btype}.",
      f"Halt the {btype} for now.",
      f"Take the {btype} offline please.",
      f"I want the {btype} paused.",
      f"Shut the {btype} down unless it is already down.",
      f"Pause the {btype} if it is still running.",
      f"Bring the {btype} to a stop.",
      f"Make sure the {btype} is not producing right now.",
      f"Idle the {btype}, we do not need it at the moment.",
      f"Cut production at the {btype}.",
    )
  )
  return Scenario(state, memory, {"text": text, "building_id": bid})


def _inventory_scenario(rng: random.Random, target: int) -> Scenario:
  resource = rng.choice(_RESOURCES)
  state = {
    "inventory": [{"resource_id": resource, "quantity": 0}], # overwritten
    "order": {"resource_id": resource, "target": target},
  }
  memory = {
    "recent": [f"user: 'buy 300 {resource}' -> buy"],
    "last_results": [f"buy {resource} 300 OK"],
  }
  text = rng.choice(
    (
      f"Top the {resource} up to {target}.",
      f"I want {target} {resource} in storage.",
      f"Bring {resource} up to {target} units.",
      f"Make sure we hold {target} {resource}.",
      f"Get {resource} to {target}, buy whatever is short.",
      f"We should be sitting on {target} {resource}. Sort the shortfall.",
      f"Stock {resource} to {target}.",
      f"Buy {resource} up to {target} if we are under.",
    )
  )
  return Scenario(state, memory, {"text": text, "resource_id": resource, "target": target})


# The target is randomised, not fixed. It appears both in the user turn and, via
# `have - target`, in the buy quantity the oracle emits, so it is the argument the
# fingerprint keys on. This is the lever that lifts the two inventory specs from
# 30 distinct pairs each to a few hundred: the sold/bought amount is a number that
# lives only in `<state>`, which is the whole point of an argument-only spec.
def _inventory_amount(rng: random.Random) -> Scenario:
  # Both branches buy (variants 120, 480), so the target must clear the larger.
  return _inventory_scenario(rng, target=rng.choice((520, 560, 600, 640, 680, 720, 760, 800)))


def _inventory_or_skip(rng: random.Random) -> Scenario:
  # Branch a (150) buys, branch b (600) skips, so the target must sit in (150, 600].
  return _inventory_scenario(rng, target=rng.choice((300, 360, 420, 480, 540, 600)))


def _production_recipe(rng: random.Random) -> Scenario:
  bid, btype = _bid(rng), rng.choice(("smelter", "furnace"))
  state = {"buildings": [{"id": bid, "type": btype, "recipe_id": "smelt_iron", "paused": False}]}
  memory = {
    "recent": [f"user: 'put the {btype} on copper' -> set_recipe"],
    "last_results": [f"set_recipe {bid} smelt_copper OK"],
  }
  text = rng.choice(
    (
      f"Set the {btype} to smelt copper.",
      f"The {btype} should be running smelt_copper.",
      f"Switch the {btype} over to copper smelting.",
      f"Put the {btype} on the copper line.",
      f"Make sure the {btype} is smelting copper, not iron.",
      f"The {btype} needs to be on copper. Change it if it is not.",
      f"Move the {btype} onto the copper recipe.",
      f"Have the {btype} produce copper from now on.",
      f"Confirm the {btype} is set to copper and fix it if it is on iron.",
    )
  )
  return Scenario(state, memory, {"text": text, "building_id": bid, "recipe_id": "smelt_copper"})


def _power_shed(rng: random.Random) -> Scenario:
  bid, btype = _bid(rng), rng.choice(_BUILDING_TYPES)
  # Generation sits strictly between the two consumption variants (60 and 140),
  # so the flip always changes which side of capacity the grid is on.
  generation = rng.randint(80, 120)
  state = {
    "grid": {"generation": generation, "consumption": 0}, # overwritten
    "buildings": [{"id": bid, "type": btype, "powered": True, "priority": rng.randint(1, 3)}],
  }
  memory = {
    "recent": ["user: 'are we browning out?' -> get_power_status"],
    "last_results": ["get_power_status OK: grid over capacity"],
  }
  text = rng.choice(
    (
      f"If we are over capacity, take the {btype} off the grid.",
      f"Shed the {btype} if the grid cannot carry it.",
      f"Drop the {btype} from power only if we are drawing too much.",
      f"We may be browning out. Cut the {btype} if so.",
      f"Pull the {btype} off power if consumption is over generation.",
      f"Should we be shedding load, take the {btype} down.",
      f"Cut the {btype} from the grid if we are past capacity.",
      f"Only if the grid is overdrawn, disconnect the {btype}.",
    )
  )
  return Scenario(state, memory, {"text": text, "building_id": bid})


def _power_priority(rng: random.Random) -> Scenario:
  bid, btype = _bid(rng), rng.choice(_BUILDING_TYPES)
  state = {
    "grid": {"generation": rng.randint(80, 120), "consumption": rng.randint(40, 70)},
    "buildings": [{"id": bid, "type": btype, "powered": True, "priority": 3}], # overwritten
  }
  memory = {
    "recent": [f"user: 'the {btype} matters most' -> set_priority"],
    "last_results": [f"set_priority {bid} 5 OK"],
  }
  text = rng.choice(
    (
      f"Give the {btype} top priority on the grid.",
      f"The {btype} should be at priority 5.",
      f"Make the {btype} the last thing we shed, priority 5.",
      f"Put the {btype} at the highest power priority.",
      f"The {btype} is critical. Set it to priority 5.",
      f"Protect the {btype} from load-shedding, priority 5.",
      f"Bump the {btype} to the top power priority.",
      f"Make sure the {btype} is at priority 5, raise it if not.",
    )
  )
  return Scenario(state, memory, {"text": text, "building_id": bid, "priority": 5})


def _connection_link(rng: random.Random) -> Scenario:
  src, dst = _bid(rng), _bid(rng)
  while dst == src:
    dst = _bid(rng)
  state = {"link": {"from_id": src, "to_id": dst, "connected": False}} # overwritten
  memory = {
    "recent": [f"user: 'run a belt from {src} to {dst}' -> connect"],
    "last_results": [f"connect {src} -> {dst} OK"],
  }
  text = rng.choice(
    (
      f"Make sure {src} feeds {dst}.",
      f"There should be a conveyor from {src} into {dst}.",
      f"Connect {src} to {dst} unless it already is.",
      f"Is {src} feeding {dst}? Wire it up if not.",
      f"Run a belt from {src} to {dst} if there is not one.",
      f"Check the link from {src} to {dst}, build it if missing.",
      f"Output from {src} should reach {dst}. See to it.",
      f"Hook {src} up to {dst}.",
    )
  )
  return Scenario(state, memory, {"text": text, "from_id": src, "to_id": dst})


def _market_selldown(rng: random.Random) -> Scenario:
  resource = rng.choice(_RESOURCES)
  # The floor is randomised, so the sold amount (have - floor) is not one of a
  # few constants. This is what makes `market` an argument-only family rather
  # than a tool-name-guessing one.
  floor = rng.choice((150, 200, 250, 300, 350, 400, 450, 500))
  state = {
    "inventory": [{"resource_id": resource, "quantity": 0}], # overwritten
    "order": {"resource_id": resource, "floor": floor},
  }
  memory = {
    "recent": [f"user: 'sell surplus {resource}' -> sell"],
    "last_results": [f"sell {resource} 200 OK"],
  }
  text = rng.choice(
    (
      f"Sell {resource} down to {floor}.",
      f"Offload {resource} above {floor} units.",
      f"Keep {floor} {resource} and sell the rest.",
      f"Trim {resource} back to {floor}.",
      f"Anything over {floor} {resource} can be sold.",
      f"Bring {resource} down to {floor}, sell the surplus.",
      f"We only need {floor} {resource}. Sell what is over.",
      f"Cut {resource} stock to {floor}.",
    )
  )
  return Scenario(state, memory, {"text": text, "resource_id": resource, "floor": floor})


TRAIN_SPECS: tuple[SwapSpec, ...] = (
  SwapSpec(
    id="building_run",
    family="building_control",
    domain="factory",
    subtype="stale_memory",
    tool_names=("inspect", "resume_building", "pause_building"),
    decisive_path="/buildings/0/paused",
    variants=(True, False),
    oracle_key="building_run",
    scenario=_building_run,
    stale_under="b", # memory says paused; branch b is running
  ),
  SwapSpec(
    id="building_stop",
    family="building_control",
    domain="factory",
    subtype="stale_memory",
    tool_names=("inspect", "resume_building", "pause_building"),
    decisive_path="/buildings/0/paused",
    variants=(False, True),
    oracle_key="building_stop",
    scenario=_building_stop,
    stale_under="b", # memory says running; branch b is paused
  ),
  SwapSpec(
    id="inventory_amount",
    family="inventory",
    domain="factory",
    subtype="memory_missing_context",
    tool_names=("get_resource", "buy", "sell", "get_market_price"),
    decisive_path="/inventory/0/quantity",
    # Both branches buy. Only the quantity differs, so a model that guesses
    # the tool from the user turn still has to read state to fill the argument.
    variants=(120, 480),
    oracle_key="inventory_topup",
    scenario=_inventory_amount,
    stale_under="a",
  ),
  SwapSpec(
    id="inventory_or_skip",
    family="inventory",
    domain="factory",
    subtype="stale_memory",
    tool_names=("get_resource", "buy", "sell", "get_market_price"),
    decisive_path="/inventory/0/quantity",
    variants=(150, 600),
    oracle_key="inventory_topup",
    scenario=_inventory_or_skip,
    stale_under="a",
  ),
  SwapSpec(
    id="production_recipe",
    family="production",
    domain="factory",
    subtype="stale_memory",
    tool_names=("set_recipe", "list_recipes", "get_production_rate"),
    decisive_path="/buildings/0/recipe_id",
    variants=("smelt_iron", "smelt_copper"),
    oracle_key="production_recipe",
    scenario=_production_recipe,
    stale_under="a", # memory says copper is set; branch a is still on iron
  ),
  SwapSpec(
    id="power_shed",
    family="power",
    domain="factory",
    subtype="memory_missing_context",
    tool_names=("get_power_status", "toggle_power", "set_priority"),
    decisive_path="/grid/consumption",
    variants=(60, 140),
    oracle_key="power_shed",
    scenario=_power_shed,
    stale_under="a", # memory says over capacity; branch a is under it
  ),
  SwapSpec(
    id="power_priority",
    family="power",
    domain="factory",
    subtype="memory_missing_context",
    tool_names=("get_power_status", "toggle_power", "set_priority"),
    decisive_path="/buildings/0/priority",
    variants=(1, 5),
    oracle_key="power_priority",
    scenario=_power_priority,
    stale_under="a",
  ),
  SwapSpec(
    id="connection_link",
    family="connection",
    domain="factory",
    subtype="stale_state",
    tool_names=("connect", "disconnect", "inspect"),
    decisive_path="/link/connected",
    variants=(False, True),
    oracle_key="connection_link",
    scenario=_connection_link,
    stale_under="a", # memory says the belt was built; branch a has no belt
  ),
  SwapSpec(
    id="market_selldown",
    family="market",
    domain="factory",
    subtype="memory_missing_context",
    tool_names=("get_resource", "sell", "get_market_price", "buy"),
    decisive_path="/inventory/0/quantity",
    # Branch a (100) is at or below any floor, so it skips; branch b (900) is
    # above any floor, so it sells the surplus. Both could plausibly call
    # `sell`; only state says how much, or whether, to sell.
    variants=(100, 900),
    oracle_key="market_selldown",
    scenario=_market_selldown,
    stale_under="a",
  ),
)


# ==========================================================================
# Eval: families and a domain that appear in no training spec
# ==========================================================================
def _research(rng: random.Random) -> Scenario:
  state = {
    "research": {"active": None, "available": ["smelting_2", "logistics_1", "power_2"]},
    "science": {"stock": rng.randint(50, 400)},
  }
  memory = {
    "recent": ["user: 'start smelting 2' -> research"],
    "last_results": ["research smelting_2 OK"],
  }
  text = rng.choice(
    (
      "Start researching smelting 2.",
      "Get smelting_2 going in the lab.",
      "I want smelting 2 under research.",
      "Put smelting_2 into the research queue.",
      "Kick off smelting 2 in research.",
      "Begin work on smelting_2.",
      "Queue up smelting 2 for the lab.",
      "Let us get smelting_2 researched.",
      "Set the lab onto smelting 2.",
      "Research smelting_2 next.",
    )
  )
  return Scenario(state, memory, {"text": text, "tech_id": "smelting_2"})


def _weather(rng: random.Random) -> Scenario:
  city = rng.choice(
    ("Tokyo", "Paris", "Lagos", "Lima", "Oslo", "Cairo", "Delhi", "Quito", "Perth", "Riga", "Accra", "Hanoi")
  )
  state = {
    "cache": {"city": city, "age_minutes": 0, "temp_c": rng.randint(-5, 34)}, # overwritten
    "policy": {"max_age_minutes": 60},
  }
  memory = {
    "recent": [f"user: 'weather in {city}' -> get_weather"],
    "last_results": [f"get_weather {city} OK, cached just now"],
  }
  text = rng.choice(
    (
      f"What's the temperature in {city}?",
      f"How warm is it in {city} right now?",
      f"Give me {city}'s current temperature.",
      f"Tell me the temperature in {city} at the moment.",
      f"How's the weather in {city} just now?",
      f"What are we looking at temperature-wise in {city}?",
      f"Current temperature for {city}, please.",
      f"Is it warm in {city} right now?",
    )
  )
  return Scenario(state, memory, {"text": text, "city": city})


def _stock_cache(rng: random.Random) -> Scenario:
  """A held-out world family beyond weather: a cached quote that may be stale.

  Its own tool family (`stocks`) and its own state schema, so the evaluation
  measures state-grounding on more than one unseen scenario. The staleness
  mechanic mirrors `_weather`; the schema and the family do not.
  """
  ticker = rng.choice(("AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "GOOG", "NFLX", "AMD", "INTC"))
  state = {
    "quote": {"symbol": ticker, "age_minutes": 0, "price": rng.randint(20, 900)}, # overwritten
    "policy": {"max_age_minutes": 15},
  }
  memory = {
    "recent": [f"user: 'price of {ticker}' -> get_stock_price"],
    "last_results": [f"get_stock_price {ticker} OK, quoted just now"],
  }
  text = rng.choice(
    (
      f"What's {ticker} trading at?",
      f"Give me the current {ticker} price.",
      f"Where is {ticker} right now?",
      f"How much is {ticker} a share at the moment?",
      f"Latest {ticker} quote, please.",
      f"What's the price on {ticker}?",
      f"Tell me {ticker}'s share price now.",
      f"Quote me {ticker}.",
    )
  )
  return Scenario(state, memory, {"text": text, "symbol": ticker})


EVAL_SPECS: tuple[SwapSpec, ...] = (
  SwapSpec(
    id="research_start",
    family="research",
    domain="factory",
    subtype="stale_memory",
    tool_names=("research", "list_research"),
    decisive_path="/research/active",
    variants=(None, "smelting_2"),
    oracle_key="research_start",
    scenario=_research,
    stale_under="a", # memory says research started; branch a has nothing active
  ),
  SwapSpec(
    id="research_switch",
    family="research",
    domain="factory",
    subtype="memory_missing_context",
    tool_names=("research", "list_research"),
    decisive_path="/research/active",
    variants=("logistics_1", "smelting_2"),
    oracle_key="research_start",
    scenario=_research,
    stale_under="a",
  ),
  SwapSpec(
    id="weather_cache",
    family="weather",
    # A held-out *domain*. No training conflict spec is a world-knowledge
    # scenario, so this measures whether state-grounding transferred at all.
    domain="world",
    subtype="stale_memory",
    tool_names=("get_weather", "web_search"),
    decisive_path="/cache/age_minutes",
    variants=(5, 240),
    oracle_key="weather_cache",
    scenario=_weather,
    stale_under="b", # memory says the cache is fresh; branch b is four hours old
  ),
  SwapSpec(
    id="stock_cache",
    family="stocks",
    # A second held-out world family. One unseen scenario cannot separate
    # "state-grounding transferred" from "this one scenario happened to work";
    # two independent families in the eval set can.
    domain="world",
    subtype="stale_memory",
    tool_names=("get_stock_price", "web_search"),
    decisive_path="/quote/age_minutes",
    variants=(4, 200),
    oracle_key="stock_cache",
    scenario=_stock_cache,
    stale_under="b",
  ),
)


ALL_SPECS: dict[str, SwapSpec] = {s.id: s for s in (*TRAIN_SPECS, *EVAL_SPECS)}


def eval_pairs(n: int, rng: random.Random) -> list:
  """Pairs from the held-out specs, for the paper's evaluation.

  No teacher is involved. The oracle is the label, so the harness prompts the
  student with `conflict.prompt_block` and compares its call to `branch.call`.
  A model that reads the user turn instead of `<state>` scores 50% by
  construction, whichever way it guesses.
  """
  from .conflict import plan_pairs

  return plan_pairs(EVAL_SPECS, n, rng)
