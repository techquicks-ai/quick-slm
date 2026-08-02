"""A DPO scenario: one fixed prompt and two candidate assistant turns.

The unit of preference here is a single decisive assistant action. That is the
natural thing for a human to judge in a glance, and it is exactly the decision
the paper's central claim rests on: given a `<state>` that outranks `<memory>`,
does the model act on what is true now. Multi-turn trajectories are deliberately
out of scope; the first action carries most of the signal and all of the
ambiguity.

Rendering goes through `template.py` and nothing else, so the strings this module
hands to the store are byte-identical to what the student trains on and to what
the inference server later emits. `completion` is derived, not hand-assembled:
`render(ex)` for a one-assistant-turn example is `render_prompt(ex)` followed by
exactly the scored segment, so slicing the prompt off the front of the full
render yields the completion the loss mask would cover, with no second definition
of the surface form to drift from `template._segments`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..template import (
  AssistantTurn,
  Example,
  UserTurn,
  render,
  render_prompt,
)
from .pairs import PreferenceRecord


@dataclass
class Candidate:
  """One assistant turn offered for comparison: private reasoning and the calls.

  `source` records where the turn came from (a mock template, or the teacher at
  a given sampling seed) and is kept in the stored record for later analysis. It
  is never surfaced in a way that would tell the labeller which answer is meant
  to be better; the whole point is that the human decides.
  """

  think: str
  calls: list[dict]
  source: str = ""


@dataclass
class Scenario:
  """A fixed prompt plus the candidates that answer it.

  The candidates share one prompt because the prompt depends only on the system
  message, the tools, `<state>`, `<memory>`, and the user turn, none of which an
  assistant turn can change. So the store keeps one `prompt` and one rendered
  `completion` per candidate.
  """

  category: str
  subtype: str
  domain: str
  system: str
  tools: list[dict]
  user: str
  candidates: list[Candidate]
  state: dict | None = None
  memory: str | None = None
  id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

  def _example(self, cand: Candidate) -> Example:
    return Example(
      tools=self.tools,
      turns=[UserTurn(self.user), AssistantTurn(cand.think, list(cand.calls))],
      state=self.state,
      memory=self.memory,
      system=self.system,
      category=self.category,
      meta={"subtype": self.subtype, "domain": self.domain},
    )

  def prompt(self) -> str:
    """Everything the model conditions on, up to `<|im_start|>assistant`.

    Identical for every candidate, so the first one is enough to render it.
    """
    return render_prompt(self._example(self.candidates[0]))

  def completion(self, cand: Candidate) -> str:
    """The scored bytes for one candidate: `<think>...</think><response>...`.

    Taken as the tail of the full render past the prompt, so it is whatever
    `template._segments` marks `scored=True`, never a re-spelling of it.
    """
    ex = self._example(cand)
    full, head = render(ex), render_prompt(ex)
    return full[len(head) :]

  def payload(self) -> dict:
    """The JSON the API hands the browser. Candidates keep their canonical order;
    the server assigns display slots A and B."""
    return {
      "scenario_id": self.id,
      "category": self.category,
      "subtype": self.subtype,
      "domain": self.domain,
      "system": self.system,
      "tools": self.tools,
      "state": self.state,
      "memory": self.memory,
      "user": self.user,
      "prompt": self.prompt(),
      "candidates": [
        {
          "think": c.think,
          "calls": c.calls,
          "source": c.source,
          "completion": self.completion(c),
        }
        for c in self.candidates
      ],
    }


def build_record(
  scenario: Scenario,
  a: Candidate,
  b: Candidate,
  decision: str,
  *,
  notes: str = "",
  annotator: str = "human",
  created_utc: str | None = None,
) -> PreferenceRecord:
  """Assemble the record the store persists.

  `a` and `b` are the candidates in the slots the labeller saw, so `decision`
  of `a` means the left card won and the right lost. The rendered prompt and
  completions are computed here, once, from `scenario`, so the browser never
  supplies training bytes and cannot tamper with the surface form.
  """
  prompt = scenario.prompt()
  comp_a, comp_b = scenario.completion(a), scenario.completion(b)

  chosen = rejected = None
  if decision == "a":
    chosen, rejected = comp_a, comp_b
  elif decision == "b":
    chosen, rejected = comp_b, comp_a

  def _face(cand: Candidate, completion: str) -> dict:
    return {"think": cand.think, "calls": cand.calls, "source": cand.source, "completion": completion}

  return PreferenceRecord(
    id="dpo-" + uuid.uuid4().hex[:12],
    scenario_id=scenario.id,
    category=scenario.category,
    subtype=scenario.subtype,
    domain=scenario.domain,
    system=scenario.system,
    tools=scenario.tools,
    state=scenario.state,
    memory=scenario.memory,
    user=scenario.user,
    prompt=prompt,
    decision=decision,
    candidate_a=_face(a, comp_a),
    candidate_b=_face(b, comp_b),
    chosen=chosen,
    rejected=rejected,
    annotator=annotator,
    notes=notes,
    created_utc=created_utc,
  )
