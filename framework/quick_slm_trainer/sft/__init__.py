"""Supervised fine-tuning: corpus design, teacher generation, filtering, packing.

The stage in one line:

  plan_requests -> run_generation -> load_and_validate -> dedup -> pack_split

Category 3 takes a different road through the same pipeline. Its samples are
generated and accepted as counterfactual pairs, judged against an oracle rather
than against the wording of their reasoning. See `conflict.py`.

Only `generate` and `pack` need torch or transformers, and each imports them
inside the functions that use them, so `specs`, `tools`, `pointer`, `oracles`,
`conflict`, `validate`, and `dedup` are testable on a laptop.

`dataset` is the exception: it subclasses `torch.utils.data.IterableDataset` and
cannot be defined without torch, so it is resolved on first attribute access.
Corpus construction (notebook 04) therefore never imports torch; only training
(notebook 05) does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import (
  conflict,
  conflict_specs,
  corpus,
  dedup,
  generate,
  oracles,
  pack,
  pointer,
  prompts,
  specs,
  tools,
  validate,
)
from .conflict import (
  Pair,
  SpecRejected,
  SwapSpec,
  assert_no_leakage,
  build_pair,
  check_spec,
  describe_capacity,
  distinct_capacity,
  plan_pairs,
  recommended_conflict_pairs,
)
from .conflict_specs import ALL_SPECS, EVAL_SPECS, TRAIN_SPECS
from .corpus import load_and_validate, paired_integrity
from .generate import (
  Request,
  SeedYield,
  describe_plan,
  describe_preflight,
  describe_seed_yield,
  distinct_turns_per_seed,
  plan_requests,
  preflight,
  run_generation,
)
from .oracles import ORACLES, canonical_call, calls_agree
from .specs import CATEGORIES, CategorySpec, plan_counts
from .tools import ALL_TOOLS, ANSWER, DOMAIN_SYSTEM, DOMAIN_TOOLS, sample_tools
from .validate import ValidationStats, extract_json, validate_and_convert, validate_record

if TYPE_CHECKING: # pragma: no cover - for type checkers only
  from .dataset import MaskedWindowDataset

_LAZY = {"MaskedWindowDataset"}


def __getattr__(name: str):
  if name in _LAZY:
    from . import dataset

    return getattr(dataset, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return sorted(__all__)


__all__ = [
  "ALL_SPECS",
  "ALL_TOOLS",
  "ANSWER",
  "CATEGORIES",
  "CategorySpec",
  "DOMAIN_SYSTEM",
  "DOMAIN_TOOLS",
  "EVAL_SPECS",
  "MaskedWindowDataset",
  "ORACLES",
  "Pair",
  "Request",
  "SeedYield",
  "SpecRejected",
  "SwapSpec",
  "TRAIN_SPECS",
  "ValidationStats",
  "assert_no_leakage",
  "build_pair",
  "calls_agree",
  "canonical_call",
  "check_spec",
  "conflict",
  "conflict_specs",
  "corpus",
  "dedup",
  "describe_capacity",
  "describe_plan",
  "describe_preflight",
  "describe_seed_yield",
  "distinct_capacity",
  "distinct_turns_per_seed",
  "extract_json",
  "generate",
  "load_and_validate",
  "oracles",
  "pack",
  "paired_integrity",
  "plan_counts",
  "plan_pairs",
  "plan_requests",
  "pointer",
  "preflight",
  "prompts",
  "recommended_conflict_pairs",
  "run_generation",
  "sample_tools",
  "specs",
  "tools",
  "validate",
  "validate_and_convert",
  "validate_record",
]
