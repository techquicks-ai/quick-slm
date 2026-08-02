"""Preference records for human-in-the-loop DPO, and their JSONL store.

A preference record is one labelling decision: a fixed prompt, two candidate
completions rendered in the student's exact ChatML surface form, and the human's
verdict. The prompt and both completions are stored as the literal strings the
student would train on, produced by `template.render` / `render_prompt`, so a
later DPO trainer consumes them without re-rendering and cannot drift from the
SFT surface form.

Storage is append-only JSONL, the same discipline the raw SFT shards use in
`sft/generate.py`: one self-contained line per decision, safe to tail while
labelling and safe to resume after a crash. Nothing here imports torch, FastAPI,
or anything a GPU needs, so the record format is testable on a laptop.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

#: The five verdicts the UI can send. Only `a` and `b` carry a chosen/rejected
#: ordering; the rest are recorded so the labelling session is fully accounted
#: for, and a trainer filters to `{"a", "b"}` when it wants clean pairs.
DECISIONS = ("a", "b", "tie", "both_bad", "skip")


@dataclass
class PreferenceRecord:
  """One labelling decision, self-contained on a single JSONL line.

  `prompt`, `chosen`, and `rejected` are the only fields a DPO trainer strictly
  needs; everything else is provenance. `chosen` / `rejected` are populated only
  when `decision` is `a` or `b`. The two candidates are always stored in full
  (`candidate_a` / `candidate_b`, each carrying `think`, `calls`, `source`, and
  the rendered `completion`) so a tie or a both-bad verdict still preserves what
  was compared.
  """

  id: str
  scenario_id: str
  category: str
  subtype: str
  domain: str
  system: str
  tools: list[dict]
  state: dict | None
  memory: str | None
  user: str
  prompt: str
  decision: str
  candidate_a: dict
  candidate_b: dict
  chosen: str | None = None
  rejected: str | None = None
  annotator: str = "human"
  notes: str = ""
  created_utc: str | None = None

  def __post_init__(self) -> None:
    if self.decision not in DECISIONS:
      raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")

  def to_json(self) -> str:
    return json.dumps(asdict(self), ensure_ascii=False)

  @classmethod
  def from_dict(cls, d: Mapping) -> "PreferenceRecord":
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PreferenceStore:
  """An append-only JSONL file of `PreferenceRecord`s.

  Append is a single line write followed by a flush, which is atomic enough for
  one labeller on one machine: a torn final line loses at most the decision in
  flight, and `read` skips it on the next load exactly as `generate.read_shard`
  does. There is no update and no delete, on purpose: a preference dataset is an
  audit trail, and a mislabel is corrected by labelling again, not by editing
  history.
  """

  path: Path
  _count: int = field(default=0, init=False)

  def __post_init__(self) -> None:
    self.path = Path(self.path)
    self._count = sum(1 for _ in self.read())

  def append(self, record: PreferenceRecord) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with open(self.path, "a", encoding="utf-8") as fh:
      fh.write(record.to_json() + "\n")
      fh.flush()
    self._count += 1

  def read(self) -> Iterator[PreferenceRecord]:
    if not self.path.exists():
      return
    with open(self.path, encoding="utf-8") as fh:
      for line in fh:
        line = line.strip()
        if not line:
          continue
        try:
          yield PreferenceRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError):
          # A line torn by a dying process, or written by an older
          # schema. Skipped rather than raised: one bad line must not
          # make the whole session unreadable.
          continue

  def __len__(self) -> int:
    return self._count

  def stats(self) -> dict:
    """Cheap tallies for the UI header. Re-reads the file; it is small."""
    by_decision: Counter = Counter()
    by_category: Counter = Counter()
    total = 0
    for rec in self.read():
      total += 1
      by_decision[rec.decision] += 1
      by_category[rec.category] += 1
    self._count = total
    return {
      "total": total,
      "pairs": by_decision["a"] + by_decision["b"],
      "by_decision": dict(by_decision),
      "by_category": dict(by_category),
    }
