"""FastAPI server for the DPO labelling UI.

Thin on purpose. Everything that decides what a scenario is, or how a preference
is stored, lives in `quick_slm_trainer.dpo`; this file only wires those to HTTP
and holds a small, bounded map of scenarios it has served so a POST can name one
without shipping the training bytes back up from the browser.

Two modes, chosen by `DPO_MODE`:

 mock   (default) synthetic scenarios, no GPU, runs anywhere. What the UI is
      built and demoed against.
 teacher Gemma 4 on the GPU box. Loaded lazily on the first scenario request,
      because it pulls torch and a large download that a laptop has no
      business triggering just by importing this module.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quick_slm_trainer.dpo import (
  Candidate,
  MockGenerator,
  PreferenceStore,
  Scenario,
  build_record,
)

# --------------------------------------------------------------------------
# Configuration from the environment
# --------------------------------------------------------------------------
MODE = os.getenv("DPO_MODE", "mock")
DATA_PATH = Path(os.getenv("DPO_DATA_PATH", "data/preferences.jsonl"))
SEED = int(os.getenv("DPO_SEED", "0")) or None
POOL_SIZE = int(os.getenv("DPO_POOL", "2000"))
#: How many recently served scenarios to keep addressable. A labeller works one
#: at a time; this only has to outlive the round trip between `next` and the
#: `preference` that answers it, with slack for a few open tabs.
SERVED_CAPACITY = 256

def _default_ui_dir() -> Path:
  """Where the front end lives when `DPO_UI_DIR` does not say.

  Backend and front end are separate trees: this package is the backend, under
  `framework/`, and the UI it serves is under `app/`. Nothing about the
  filesystem guarantees the second is reachable from the first, so the
  container sets `DPO_UI_DIR` outright. This fallback is for a developer
  running uvicorn from a checkout, where walking up to the repository root and
  back down is the only thing that can work.
  """
  for parent in Path(__file__).resolve().parents:
    candidate = parent / "app" / "quick_slm_dpo_studio"
    if candidate.is_dir():
      return candidate
  return Path(__file__).resolve().parent / "static"


UI_DIR = Path(os.environ["DPO_UI_DIR"]) if os.getenv("DPO_UI_DIR") else _default_ui_dir()

app = FastAPI(title="Quick SLM · DPO Studio")

_store = PreferenceStore(DATA_PATH)
_served: "OrderedDict[str, tuple[Scenario, Candidate, Candidate]]" = OrderedDict()
_lock = threading.Lock()
_generator = None
_rng = random.Random(SEED)


def _get_generator():
  """The scenario source, built once. Teacher construction is deferred to here."""
  global _generator
  if _generator is not None:
    return _generator
  with _lock:
    if _generator is not None:
      return _generator
    if MODE == "mock":
      _generator = MockGenerator(seed=SEED)
    elif MODE == "teacher":
      _generator = _build_teacher()
    else:
      raise RuntimeError(f"DPO_MODE must be 'mock' or 'teacher', got {MODE!r}")
    return _generator


def _build_teacher():
  """Load Gemma 4 in 4-bit and wrap it. Only reached in teacher mode.

  Imports are local so that a mock deployment never pulls torch, and a failure
  to load the teacher surfaces as an error on the first scenario request rather
  than as an import-time crash of the whole server.
  """
  from quick_slm_trainer.config import SFTConfig
  from quick_slm_trainer.dpo import TeacherGenerator
  from quick_slm_trainer.sft.generate import load_teacher

  cfg = SFTConfig(target_examples=POOL_SIZE)
  model, tok = load_teacher(cfg)
  return TeacherGenerator(cfg, model, tok)


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------
class Verdict(BaseModel):
  scenario_id: str
  decision: str
  notes: str = ""


def _now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _remember(scenario: Scenario, a: Candidate, b: Candidate) -> None:
  _served[scenario.id] = (scenario, a, b)
  while len(_served) > SERVED_CAPACITY:
    _served.popitem(last=False)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
  page = UI_DIR / "index.html"
  if not page.is_file():
    # A missing front end is a deployment fault, not a missing resource, and
    # a bare 404 here reads as a routing bug and sends people into the route
    # table. Say which directory was looked in and which variable moves it.
    raise HTTPException(
      status_code=500,
      detail=(
        f"DPO Studio UI not found at {page}. The front end lives in "
        "app/quick_slm_dpo_studio, separately from this backend; set "
        "DPO_UI_DIR to point at it."
      ),
    )
  return FileResponse(page)


@app.get("/api/health")
def health() -> dict:
  # `ui_dir` and `ui_present` are here because the two halves are deployed
  # separately now, so "the page is blank" has to be answerable without shell
  # access to the container.
  return {
    "ok": True,
    "mode": MODE,
    "data_path": str(DATA_PATH),
    "ui_dir": str(UI_DIR),
    "ui_present": (UI_DIR / "index.html").is_file(),
  }


@app.get("/api/stats")
def stats() -> dict:
  return {"mode": MODE, **_store.stats()}


@app.get("/api/next")
def next_scenario() -> dict:
  """Generate one scenario, assign display slots A and B, and remember it.

  The two candidates are shuffled into slots so the labeller cannot learn a
  position habit (the first mock answer is always the intended-better one, for
  instance). The mapping is held server-side; the browser only echoes back the
  scenario id and a verdict.
  """
  scenario = _get_generator().next()
  a, b = scenario.candidates[0], scenario.candidates[1]
  if _rng.random() < 0.5:
    a, b = b, a
  _remember(scenario, a, b)
  return {
    "scenario_id": scenario.id,
    "category": scenario.category,
    "subtype": scenario.subtype,
    "domain": scenario.domain,
    "system": scenario.system,
    "tools": scenario.tools,
    "state": scenario.state,
    "memory": scenario.memory,
    "user": scenario.user,
    "prompt": scenario.prompt(),
    "candidate_a": {"think": a.think, "calls": a.calls},
    "candidate_b": {"think": b.think, "calls": b.calls},
  }


@app.post("/api/preference")
def record_preference(v: Verdict) -> dict:
  """Persist one decision. `decision` is validated by `PreferenceRecord`."""
  entry = _served.get(v.scenario_id)
  if entry is None:
    # The server was restarted, or the scenario aged out of the map. The
    # browser handles this by fetching a fresh scenario.
    raise HTTPException(status_code=410, detail="scenario expired; fetch a new one")
  scenario, a, b = entry
  try:
    record = build_record(scenario, a, b, v.decision, notes=v.notes, created_utc=_now())
  except ValueError as exc:
    raise HTTPException(status_code=422, detail=str(exc))
  _store.append(record)
  _served.pop(v.scenario_id, None)
  return {"ok": True, "id": record.id, "total": len(_store)}
