"""Human-in-the-loop DPO: build preference pairs by picking the better answer.

The pieces here are pure and testable on a laptop. `framework/quick_slm_dpo_studio/` (a FastAPI app
and a single-page UI, outside `src/`) is the thin server that drives them. The
flow: a `Generator` yields a `Scenario` (one prompt, two candidate assistant
turns), a human picks, and a `PreferenceRecord` is appended to a `PreferenceStore`
in the student's exact ChatML surface form, ready for a later DPO run.
"""

from __future__ import annotations

from .generators import Generator, MockGenerator, TeacherGenerator, make_generator
from .pairs import DECISIONS, PreferenceRecord, PreferenceStore
from .scenario import Candidate, Scenario, build_record

__all__ = [
  "Generator",
  "MockGenerator",
  "TeacherGenerator",
  "make_generator",
  "DECISIONS",
  "PreferenceRecord",
  "PreferenceStore",
  "Candidate",
  "Scenario",
  "build_record",
]
