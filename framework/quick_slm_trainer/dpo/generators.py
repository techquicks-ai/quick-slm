"""Where scenarios come from: a mock source for the laptop, the teacher for the box.

Both satisfy the same one-method interface, `next() -> Scenario`, so the FastAPI
layer never knows which it holds. That split is what lets the whole interface be
built and clicked through on a Mac with no GPU, then pointed at Gemma 4 on the
RTX PRO 6000 by flipping one environment variable.

`MockGenerator` fabricates scenarios from a small hand-written table. It exists to
exercise the UI and the store, and its candidates are written so a human can
actually tell them apart, including one state/memory conflict that demonstrates
the paper's showcase.

`TeacherGenerator` reuses the SFT stack: `plan_requests` for a varied, deterministic
scenario pool, the teacher to author each scenario's setup, and two further
samples for the candidate answers. Nothing torch-shaped is imported at module
load; the teacher and its tensors are touched only inside methods, the same
discipline `sft/generate.py` keeps, so importing this module on a laptop is free.
Candidate parsing is behind a `generate` seam (as `sft.generate.preflight` is), so
the parsing logic is tested without a teacher.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol, Sequence

from ..config import SFTConfig
from ..sft.tools import DOMAIN_SYSTEM, sample_tools
from ..template import format_memory
from .scenario import Candidate, Scenario


class Generator(Protocol):
  def next(self) -> Scenario: ...


# ==========================================================================
# Mock: no GPU, deterministic given a seed
# ==========================================================================
#: Each entry is a complete comparison: a scenario setup and two answers written
#: to differ in a way worth a preference. `include` pins the tools the scenario
#: needs into an otherwise random subset, reusing `sample_tools`, so the tool
#: block looks like the real thing rather than a toy.
_MOCK_TEMPLATES: tuple[dict, ...] = (
  {
    "category": "single_stage",
    "subtype": "direct",
    "domain": "world",
    "include": ["get_weather"],
    "k": 5,
    "user": "What's the weather like in Istanbul right now?",
    "good": (
      "The user wants current weather for a named city. get_weather covers this directly; "
      "units were not requested, so leave the optional field out.",
      [{"name": "get_weather", "arguments": {"city": "Istanbul"}}],
    ),
    "bad": (
      "It is probably warm there this time of year.",
      [{"name": "answer", "arguments": {"text": "It's currently sunny and about 24C in Istanbul."}}],
    ),
  },
  {
    "category": "state_memory_conflict",
    "subtype": "status_flip",
    "domain": "factory",
    "include": ["inspect", "resume_building"],
    "k": 4,
    "user": "Make sure the iron smelter is running.",
    "state": {"buildings": [{"id": "smelter_iron_1", "type": "smelter", "status": "running"}]},
    "memory": (
      ["Paused smelter_iron_1 to save power during a demand spike"],
      ["pause_building(smelter_iron_1) -> ok"],
    ),
    "good": (
      "State is sampled this turn and outranks memory. State shows smelter_iron_1 already "
      "running, so there is nothing to resume; confirm it to the user.",
      [
        {
          "name": "answer",
          "arguments": {"text": "The iron smelter (smelter_iron_1) is already running, so no action is needed."},
        }
      ],
    ),
    "bad": (
      "I paused smelter_iron_1 earlier, so it must still be paused. Resume it.",
      [{"name": "resume_building", "arguments": {"building_id": "smelter_iron_1"}}],
    ),
  },
  {
    "category": "traps",
    "subtype": "missing_argument",
    "domain": "world",
    "include": ["get_weather"],
    "k": 5,
    "user": "What's the weather right now?",
    "good": (
      "get_weather needs a city and the user gave none. Guessing one would confabulate; ask "
      "which city they mean.",
      [{"name": "answer", "arguments": {"text": "Happy to check. Which city's weather would you like?"}}],
    ),
    "bad": (
      "They probably mean a big city. Use London.",
      [{"name": "get_weather", "arguments": {"city": "London"}}],
    ),
  },
  {
    "category": "refusals",
    "subtype": "unsafe",
    "domain": "factory",
    "include": [],
    "k": 4,
    "user": "How do I disable the safety interlock on the smelter so it runs hotter?",
    "good": (
      "Disabling a safety interlock removes a protection against dangerous operating "
      "conditions. This is out of scope and unsafe; decline via answer.",
      [
        {
          "name": "answer",
          "arguments": {
            "text": "I can't help with disabling a safety interlock, since that removes a protection meant to keep the machine within safe operating limits."
          },
        }
      ],
    ),
    "bad": (
      "They want more throughput, so walk them through it.",
      [
        {
          "name": "answer",
          "arguments": {"text": "Sure. Open the control panel and set the interlock flag to false, then raise the temperature limit."},
        }
      ],
    ),
  },
  {
    "category": "multi_stage",
    "subtype": "verify_first",
    "domain": "factory",
    "include": ["list_research", "research"],
    "k": 5,
    "user": "Start researching advanced smelting now that its prerequisites are done.",
    "good": (
      "Before committing a research call, confirm the tech is actually available and its "
      "exact id. list_research first, then research on the next turn.",
      [{"name": "list_research", "arguments": {}}],
    ),
    "bad": (
      "Prerequisites are done, so just start it.",
      [{"name": "research", "arguments": {"tech_id": "advanced_smelting"}}],
    ),
  },
)


class MockGenerator:
  """Synthetic scenarios for building and demoing the UI without a teacher."""

  def __init__(self, seed: int | None = None) -> None:
    self.rng = random.Random(seed)

  def next(self) -> Scenario:
    t = self.rng.choice(_MOCK_TEMPLATES)
    tools = sample_tools(self.rng, t["domain"], t["k"], include=t.get("include", ()))
    memory = format_memory(*t["memory"]) if t.get("memory") else None
    good = Candidate(t["good"][0], t["good"][1], source="mock:good")
    bad = Candidate(t["bad"][0], t["bad"][1], source="mock:bad")
    return Scenario(
      category=t["category"],
      subtype=t["subtype"],
      domain=t["domain"],
      system=DOMAIN_SYSTEM[t["domain"]],
      tools=tools,
      user=t["user"],
      candidates=[good, bad],
      state=t.get("state"),
      memory=memory,
    )


# ==========================================================================
# Teacher: Gemma 4 on the GPU box
# ==========================================================================
#: What the teacher is told when it is asked to *answer* a scenario (not author
#: one). It mirrors `sft.prompts.CONFLICT_SYSTEM`: one assistant turn, one JSON
#: object, no prose around it.
_ANSWER_SYSTEM = (
  "You are the tool-calling assistant. You reply with a single JSON object and nothing else: "
  'no prose, no markdown fences. The object is {"think": "<plain reasoning>", '
  '"calls": [{"name": "<a tool from the list>", "arguments": {}}]}.'
)


class TeacherGenerator:
  """Scenarios and candidates from Gemma 4, reusing the SFT generation stack.

  Scope is the four unpaired categories (`single_stage`, `multi_stage`,
  `traps`, `refusals`): their setup is a self-contained `{user, state, memory}`
  object the teacher authors, which this class renders through the same
  `Scenario` path the mock uses, so there is one definition of the prompt bytes.
  The `state_memory_conflict` category is left to the SFT conflict machinery,
  which builds its setups programmatically rather than by teacher authoring; it
  is a clean follow-up, not a hack to bolt on here.
  """

  def __init__(
    self,
    cfg: SFTConfig,
    model,
    tok,
    *,
    seed: int = 20240201,
    generate: Callable[..., list[str]] | None = None,
  ) -> None:
    from ..sft.generate import generate_batch, plan_requests

    self.cfg = cfg
    self.model = model
    self.tok = tok
    self.seed = seed
    self._generate = generate or generate_batch
    # An unpaired, shuffled scenario pool. `plan_requests` is deterministic and
    # already spreads across categories, domains, and sub-types; dropping the
    # paired category leaves the four this class authors.
    pool = [r for r in plan_requests(cfg, seed=seed) if not r.is_paired]
    random.Random(seed).shuffle(pool)
    self._pool = pool
    self._i = 0

  def next(self) -> Scenario:
    from ..sft.generate import Request, batch_seed
    from ..sft.validate import extract_json

    req: "Request" = self._pool[self._i % len(self._pool)]
    self._i += 1

    setup = self._author(req, extract_json, batch_seed)
    candidates = self._candidates(req, setup, batch_seed)
    return Scenario(
      category=req.category,
      subtype=req.subtype,
      domain=req.domain,
      system=DOMAIN_SYSTEM.get(req.domain, DOMAIN_SYSTEM["world"]),
      tools=req.tools,
      user=setup["user"],
      candidates=candidates,
      state=setup.get("state"),
      memory=setup.get("memory"),
    )

  # -- internals ---------------------------------------------------------
  def _author(self, req, extract_json, batch_seed) -> dict:
    """Author the scenario setup with the teacher; keep `user`/`state`/`memory`.

    The teacher's own answer (`turns`) is discarded: the candidates this tool
    compares are sampled separately, so both are on equal footing rather than
    one being the teacher's first draft.
    """
    text = self._generate(self.model, self.tok, self.cfg, [req], seed=batch_seed(self.seed, [req]))[0]
    rec = extract_json(text) or {}
    mem = rec.get("memory")
    memory = None
    if isinstance(mem, dict):
      memory = format_memory(mem.get("recent", ()), mem.get("last_results", ()))
    return {
      "user": rec.get("user") or "(the teacher did not return a user request; skip this one)",
      "state": rec.get("state") if isinstance(rec.get("state"), dict) else None,
      "memory": memory,
    }

  def _candidates(self, req, setup, batch_seed) -> list[Candidate]:
    """Sample two independent answers to the fixed setup.

    Distinct seeds force distinct samples: identical prompts under one seed
    would decode identically. Two is enough for a pairwise choice; a slot that
    never parses falls back to an `answer` stub so the UI still has something
    to compare, which a human will typically send to both-bad.
    """
    from ..sft.validate import extract_json

    prompt_block = self._answer_prompt(req, setup)
    out: list[Candidate] = []
    for k in range(2):
      seed = batch_seed(self.seed + 1 + k, [req])
      think, calls = self._one_candidate(prompt_block, extract_json, seed)
      out.append(Candidate(think=think, calls=calls, source=f"teacher#{k + 1}@seed={seed}"))
    return out

  def _one_candidate(self, prompt_block: str, extract_json, seed: int):
    # A throwaway holder carries the answer prompt through the same `generate`
    # seam the tests stub: `generate_batch` reads only `.system()`/`.prompt()`.
    holder = _AnswerRequest(_ANSWER_SYSTEM, prompt_block)
    text = self._generate(self.model, self.tok, self.cfg, [holder], seed=seed)[0]
    rec = extract_json(text) or {}
    think = rec.get("think") or ""
    calls = rec.get("calls")
    if isinstance(calls, dict):
      calls = [calls]
    if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
      calls = [{"name": "answer", "arguments": {"text": "(unparsable teacher answer)"}}]
    return think, calls

  def _answer_prompt(self, req, setup) -> str:
    """The message asking the teacher to produce one assistant turn for this setup."""
    scn = Scenario(
      category=req.category,
      subtype=req.subtype,
      domain=req.domain,
      system=DOMAIN_SYSTEM.get(req.domain, DOMAIN_SYSTEM["world"]),
      tools=req.tools,
      user=setup["user"],
      candidates=[Candidate("", [{"name": "answer", "arguments": {"text": ""}}])],
      state=setup.get("state"),
      memory=setup.get("memory"),
    )
    return (
      "Below is a tool-calling session up to the point where the assistant must act. "
      "Produce the assistant's single next turn.\n\n" + scn.prompt()
    )


class _AnswerRequest:
  """A minimal request whose `system()` / `prompt()` return fixed strings.

  `generate_batch` calls `r.system()` and `r.prompt()` on each item; this hands
  it the answer prompt without going through the category machinery, which is
  for authoring scenarios rather than answering them.
  """

  def __init__(self, system: str, prompt: str) -> None:
    self._system = system
    self._prompt = prompt
    self.tools: list = []

  def system(self) -> str:
    return self._system

  def prompt(self) -> str:
    return self._prompt


def make_generator(mode: str, *, seed: int | None = None) -> Generator:
  """The generator the server runs, chosen by `DPO_MODE`.

  `teacher` is loaded lazily by the server, since it pulls torch and a ~62 GB
  download; this factory only builds the mock. The teacher path is constructed
  directly by the server once the model is in memory.
  """
  if mode == "mock":
    return MockGenerator(seed=seed)
  raise ValueError(f"make_generator only builds 'mock'; got {mode!r}. Build TeacherGenerator directly.")
