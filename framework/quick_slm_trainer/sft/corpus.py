"""Read the raw teacher shards, filter them, and hand back `Example` objects.

The bridge between `generate` (which only ever appends raw text) and `pack`
(which only ever sees validated examples). Nothing here touches a GPU, so the
whole filter can be re-run, tightened, and re-run again on a CPU runtime while
the expensive artifact, the raw JSONL, stays untouched on Drive.

Each raw record carries the exact tool subset the teacher was shown. Validation
uses that subset and never a freshly sampled one: a call is legal or not with
respect to the tools that were on offer at the time, and re-sampling would make
the filter's verdict depend on an RNG.

The paired category takes the other road. Its records carry the `<state>`, the
`<memory>`, and the user turn that were planned for them, and the oracle is
recomputed here from `conflict_specs`. Both branches of a pair are accepted or
dropped together.
"""

from __future__ import annotations

from typing import Sequence

from ..config import SFTConfig
from ..paths import Layout
from ..template import Example
from . import conflict
from .conflict_specs import ALL_SPECS
from .generate import read_shard
from .specs import CATEGORIES
from .tools import DOMAIN_SYSTEM
from .validate import SWAP_MISSING_BRANCH, ValidationStats, extract_json, validate_and_convert


def load_and_validate(
  layout: Layout,
  cfg: SFTConfig,
  *,
  categories: Sequence[str] | None = None,
  progress: bool = True,
) -> tuple[list[Example], ValidationStats, dict[str, ValidationStats]]:
  """Parse every raw shard into validated examples.

  Returns the accepted examples, the corpus-wide stats, and the per-category
  stats. The per-category breakdown is what tells a targeted regeneration run
  which category to regenerate.
  """
  keys = list(categories or CATEGORIES)
  overall = ValidationStats()
  per_category: dict[str, ValidationStats] = {}
  examples: list[Example] = []

  for key in keys:
    spec = CATEGORIES[key]
    rows = list(read_shard(layout.sft_raw(key)))

    if spec.paired:
      accepted, stats = _validate_paired(rows, progress=progress)
    else:
      accepted, stats = _validate_unpaired(rows, key, progress=progress)

    per_category[key] = stats
    examples.extend(accepted)
    overall.accepted += stats.accepted
    overall.rejected.update(stats.rejected)
    print(f"[{key}] {stats.report()}")

  return examples, overall, per_category


def _validate_unpaired(rows: list[dict], key: str, *, progress: bool) -> tuple[list[Example], ValidationStats]:
  spec = CATEGORIES[key]
  stats = ValidationStats()
  out: list[Example] = []

  iterator = rows
  if progress and rows:
    from tqdm.auto import tqdm

    iterator = tqdm(rows, desc=f"validate {key}", unit="ex")

  for row in iterator:
    rec = extract_json(row.get("raw", ""))
    ex = validate_and_convert(
      rec,
      spec,
      row.get("tools") or [],
      system=DOMAIN_SYSTEM[row.get("domain", "world")],
      subtype=row.get("subtype", ""),
      stats=stats,
    )
    if ex is not None:
      out.append(ex)
  return out, stats


def _validate_paired(rows: list[dict], *, progress: bool) -> tuple[list[Example], ValidationStats]:
  """Group the shard by `pair_id` and accept or reject each pair as a unit.

  A pair failure is tallied against *both* branches, because both examples are
  lost. Reading the histogram as a per-sample reject rate is therefore correct,
  which it would not be if the pair were counted once.
  """
  stats = ValidationStats()
  out: list[Example] = []

  pairs: dict[str, dict[str, dict]] = {}
  for row in rows:
    pair_id, branch = row.get("pair_id"), row.get("branch")
    if pair_id is None or branch not in ("a", "b"):
      # A record from before the pair rewrite, or a torn line.
      stats.record(SWAP_MISSING_BRANCH)
      continue
    pairs.setdefault(pair_id, {})[branch] = row

  iterator = pairs.items()
  if progress and pairs:
    from tqdm.auto import tqdm

    iterator = tqdm(iterator, total=len(pairs), desc="validate state_memory_conflict", unit="pair")

  for _pair_id, branches in iterator:
    spec_id = next(iter(branches.values())).get("spec_id")
    spec = ALL_SPECS.get(spec_id)
    if spec is None:
      for _ in branches:
        stats.record(SWAP_MISSING_BRANCH)
      continue

    reasons, accepted = conflict.validate_pair(branches, spec)
    if accepted:
      for example in accepted.values():
        stats.record(None)
        out.append(example)
    else:
      for reason in reasons.values():
        stats.record(reason)

  return out, stats


def category_histogram(examples: Sequence[Example]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for ex in examples:
    counts[ex.category] = counts.get(ex.category, 0) + 1
  return counts


def subtype_histogram(examples: Sequence[Example]) -> dict[str, dict[str, int]]:
  out: dict[str, dict[str, int]] = {}
  for ex in examples:
    sub = out.setdefault(ex.category, {})
    key = ex.meta.get("subtype", "")
    sub[key] = sub.get(key, 0) + 1
  return out


def paired_integrity(examples: Sequence[Example]) -> tuple[int, list[str]]:
  """Every accepted pair must contribute exactly two examples.

  A lone branch surviving into the corpus is the failure the whole construction
  exists to prevent, and it is cheap to assert rather than assume. Returns the
  pair count and the ids of any pair that did not arrive intact.
  """
  seen: dict[str, set[str]] = {}
  for ex in examples:
    if ex.category != "state_memory_conflict":
      continue
    seen.setdefault(ex.meta["pair_id"], set()).add(ex.meta["branch"])
  broken = sorted(pid for pid, branches in seen.items() if branches != {"a", "b"})
  return len(seen), broken
