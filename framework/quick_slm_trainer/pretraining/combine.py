"""Interleave the per-source `.bin` files into one `combined.bin`.

Windows are drawn from the four sources in a shuffled pattern, so any contiguous
slice of the result matches the target ratio and the trainer never needs to know
about source boundaries. It just reads fixed windows off a flat memmap.

The allocation step is pure and lives in `plan_allocation` so it can be tested
without a corpus. The earlier notebook wrapped a source back to offset zero when
it ran out of data, which is how a missing FineWeb-Edu shard once passed
unnoticed. Running past the end of a source is a bug in the allocation, and this
module raises instead of hiding it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..config import DataConfig
from ..manifest import Manifest
from ..paths import Layout


class CombineRefused(RuntimeError):
  """Raised under `strict` when a source has not reached its budget."""


def plan_allocation(
  budgets: dict[str, int],
  actuals: dict[str, int],
  *,
  ctx: int,
  mode: str,
) -> dict[str, int]:
  """Decide how many tokens to take from each source. Returns CTX multiples.

  `strict`
    Refuse if any source is short. The default: a corpus whose ratios do not
    match the ratios the paper claims is worse than no corpus.
  `cap_proportional`
    Scale every source to the most-incomplete one's fraction, preserving the
    77.5 / 10 / 2.5 / 10 ratio at a smaller total.
  `use_all`
    Take whatever exists. Ratios skew. A last resort.
  """
  short = [k for k in budgets if actuals.get(k, 0) < budgets[k]]

  if mode == "strict":
    if short:
      detail = ", ".join(f"{k} short by {budgets[k] - actuals.get(k, 0):,}" for k in short)
      raise CombineRefused(
        f"combine_mode='strict' and these sources are short of budget: {detail}. "
        "Let them finish, or set combine_mode to 'cap_proportional' "
        "(preserves ratios) or 'use_all' (skews ratios)."
      )
    take = dict(budgets)
  elif mode == "cap_proportional":
    scale = min(min(actuals.get(k, 0) / budgets[k] for k in budgets), 1.0)
    take = {k: int(budgets[k] * scale) for k in budgets}
  elif mode == "use_all":
    take = {k: min(actuals.get(k, 0), budgets[k]) for k in budgets}
  else:
    raise ValueError(f"unknown combine_mode: {mode!r}")

  # Round each allocation down to a whole number of windows. A partial window
  # at the tail of a source would be silently zero-padded by the memmap read.
  return {k: (take[k] // ctx) * ctx for k in budgets}


def combine_sources(
  *,
  layout: Layout,
  manifest: Manifest,
  cfg: DataConfig,
  progress: bool = True,
) -> int:
  """Build `combined.bin` locally, verify it, then copy to Drive.

  Returns the token count written. A no-op when a verified Drive copy exists.
  """
  if manifest.combined_ok():
    size = layout.drive_combined.stat().st_size
    print(f"combined.bin already verified on Drive ({size / 1e9:.2f} GB); skipping")
    return int(manifest.data["combined"]["tokens_written"])

  ctx = cfg.ctx
  budgets = cfg.budgets

  print("Source status:")
  actuals: dict[str, int] = {}
  for k in budgets:
    local, drive = layout.local_source(k), layout.drive_source(k)
    if not local.exists() and drive.exists():
      print(f" [{k}] restoring local from Drive")
      shutil.copy(drive, local)
    actuals[k] = local.stat().st_size // 2 if local.exists() else 0
    flag = "ok" if actuals[k] >= budgets[k] else f"short by {budgets[k] - actuals[k]:,}"
    print(f" {k:<14s} {actuals[k]:>15,} / {budgets[k]:>15,} tokens  {flag}")

  take = plan_allocation(budgets, actuals, ctx=ctx, mode=cfg.combine_mode)
  windows = {k: take[k] // ctx for k in budgets}
  total_windows = sum(windows.values())
  total_tokens = total_windows * ctx
  print(f"\ncombined.bin: {total_tokens:,} tokens ({total_windows:,} windows, "
     f"{total_tokens * 2 / 1e9:.2f} GB)")

  arrs = {
    k: np.memmap(layout.local_source(k), dtype=np.uint16, mode="r",
           shape=(layout.local_source(k).stat().st_size // 2,))
    for k in budgets
  }

  dst = layout.local_combined
  dst.parent.mkdir(parents=True, exist_ok=True)
  if dst.exists():
    dst.unlink()
  out = np.memmap(dst, dtype=np.uint16, mode="w+", shape=(total_tokens,))

  pattern: list[str] = []
  for k in budgets:
    pattern.extend([k] * windows[k])
  np.random.default_rng(cfg.seed).shuffle(pattern)

  bar = None
  if progress:
    from tqdm.auto import tqdm

    bar = tqdm(total=len(pattern), desc="combine", unit="win")

  cursor = {k: 0 for k in budgets}
  for i, k in enumerate(pattern):
    src = arrs[k]
    start = cursor[k] * ctx
    end = start + ctx
    if end > src.shape[0]:
      raise RuntimeError(
        f'source "{k}" ran out at window {cursor[k]}: the file holds '
        f"{src.shape[0]:,} tokens but the allocation wanted to read past its end. "
        "This is an allocation bug; do not wrap."
      )
    out[i * ctx : (i + 1) * ctx] = src[start:end]
    cursor[k] += 1
    if i % 1000 == 0:
      out.flush()
    if bar is not None:
      bar.update(1)
  if bar is not None:
    bar.close()
  out.flush()
  del out

  print(f"\nsyncing combined.bin to Drive ({total_tokens * 2 / 1e9:.2f} GB)")
  layout.drive_combined.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(dst, layout.drive_combined)
  if dst.stat().st_size != layout.drive_combined.stat().st_size:
    print(" copy incomplete; retrying once")
    shutil.copy(dst, layout.drive_combined)
    if dst.stat().st_size != layout.drive_combined.stat().st_size:
      raise RuntimeError(
        "Drive sync verification failed twice; the local copy remains the source of truth"
      )
  print(f"sync verified ({total_tokens * 2 / 1e9:.2f} GB)")

  manifest.set_combined(total_tokens, ctx, cfg.combine_mode, take)
  manifest.save()
  return total_tokens


def mirror_to_local(drive_path: Path, local_path: Path) -> Path:
  """Copy the corpus to local SSD so random window reads are not throttled
  by Drive's FUSE mount. Skips when the sizes already agree."""
  import time

  drive_size = drive_path.stat().st_size if drive_path.exists() else 0
  if drive_size == 0:
    raise RuntimeError(f"combined.bin missing on Drive: {drive_path}")

  local_path.parent.mkdir(parents=True, exist_ok=True)
  local_size = local_path.stat().st_size if local_path.exists() else 0
  if local_size == drive_size:
    print(f"local mirror ready: {local_size / 1e9:.2f} GB (skipping copy)")
    return local_path

  if local_size:
    print(f"local mirror size mismatch ({local_size / 1e9:.2f} GB vs "
       f"{drive_size / 1e9:.2f} GB); recopying")
    local_path.unlink()

  print(f"copying combined.bin to local SSD ({drive_size / 1e9:.2f} GB)")
  t0 = time.time()
  shutil.copy(drive_path, local_path)
  dt = max(time.time() - t0, 1e-9)
  print(f" copied in {dt / 60:.1f} min ({drive_size / 1e6 / dt:.0f} MB/s)")
  return local_path
