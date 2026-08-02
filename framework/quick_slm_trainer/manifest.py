"""Resumable progress tracking for corpus construction.

The manifest is the one small file on Drive, and it is *not* the source of
truth. The on-disk `.bin` files are. `reconcile` exists because a Colab runtime
can die between an `fsync` and a manifest write, leaving the manifest claiming
fewer tokens than the file holds; appending from the manifest's count would then
duplicate a batch. Trust order is: local file, then the Drive copy, then the
manifest's claim.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .paths import Layout


class Manifest:
  def __init__(self, layout: Layout, budgets: dict[str, int]):
    self.layout = layout
    self.budgets = budgets
    self.data = self._load()

  # --- persistence ------------------------------------------------------
  def _load(self) -> dict:
    path = self.layout.manifest_path
    if path.exists():
      data = json.loads(path.read_text())
    else:
      data = {"sources": {}, "combined": {"tokens_written": 0, "done": False},
          "created_at": time.time()}
    for k in self.budgets:
      data.setdefault("sources", {}).setdefault(k, {"tokens_written": 0, "done": False})
    return data

  def save(self) -> None:
    self.layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    self.layout.manifest_path.write_text(json.dumps(self.data, indent=2))

  # --- accessors --------------------------------------------------------
  def source(self, key: str) -> dict:
    return self.data["sources"][key]

  def tokens_written(self, key: str) -> int:
    return int(self.source(key)["tokens_written"])

  def is_done(self, key: str) -> bool:
    return bool(self.source(key)["done"])

  def set_progress(self, key: str, tokens: int, rows_read: int = 0) -> None:
    entry = self.source(key)
    entry["tokens_written"] = int(tokens)
    entry["rows_read"] = int(rows_read)
    entry["done"] = tokens >= self.budgets[key]

  # --- repair -----------------------------------------------------------
  def reconcile(self, key: str, *, verbose: bool = True) -> int:
    """Repair one source's entry from on-disk truth. Returns tokens on disk."""
    local = self.layout.local_source(key)
    drive = self.layout.drive_source(key)

    if not local.exists() and drive.exists():
      if verbose:
        print(f" [{key}] restoring local from Drive ({drive.stat().st_size / 1e9:.2f} GB)")
      local.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy(drive, local)

    on_disk = local.stat().st_size // 2 if local.exists() else 0
    entry = self.source(key)
    if on_disk != entry["tokens_written"]:
      if verbose:
        print(
          f" [{key}] manifest says {entry['tokens_written']:,}, "
          f"file has {on_disk:,}; repairing manifest"
        )
    self.set_progress(key, on_disk)
    return on_disk

  def reconcile_all(self, *, verbose: bool = True) -> None:
    for k in self.budgets:
      self.reconcile(k, verbose=verbose)
    self.save()

  # --- combined ---------------------------------------------------------
  def combined_ok(self) -> bool:
    c = self.data["combined"]
    if not c.get("done"):
      return False
    dst = self.layout.drive_combined
    if not dst.exists():
      return False
    expected = int(c.get("tokens_written", 0)) * 2
    return bool(expected) and dst.stat().st_size == expected

  def set_combined(self, tokens: int, ctx: int, mode: str, per_source: dict[str, int]) -> None:
    self.data["combined"] = {
      "tokens_written": int(tokens),
      "ctx": int(ctx),
      "done": True,
      "mode": mode,
      "per_source": per_source,
    }

  def finalise(self, *, tokenizer_dir: Path, ctx: int, vocab_size: int) -> None:
    """Write the fields the pretraining notebook reads back."""
    self.data["tokenizer_dir"] = str(tokenizer_dir)
    self.data["combined_path"] = str(self.layout.drive_combined)
    self.data["local_combined_path"] = str(self.layout.local_combined)
    self.data["ctx"] = int(ctx)
    self.data["vocab_size"] = int(vocab_size)
    self.data["budgets"] = self.budgets
    self.save()

  # --- diagnostics ------------------------------------------------------
  def integrity_report(self) -> tuple[str, bool]:
    """Read-only comparison of manifest claims against local and Drive."""
    lines = [f'{"source":<14s} {"manifest":>15s} {"local":>15s} {"drive":>15s} {"match":>8s}',
         "-" * 75]
    ok = True
    for k in self.budgets:
      claimed = self.tokens_written(k)
      local = self.layout.local_source(k)
      drive = self.layout.drive_source(k)
      l = local.stat().st_size // 2 if local.exists() else 0
      d = drive.stat().st_size // 2 if drive.exists() else 0
      match = (l == claimed and (d == 0 or d == l)) or (l == 0 and d == claimed)
      ok &= match
      lines.append(f'{k:<14s} {claimed:>15,} {l:>15,} {d:>15,} {"OK" if match else "MISMATCH":>8s}')

    lines.append("-" * 75)
    cclaim = int(self.data.get("combined", {}).get("tokens_written", 0))
    cl = self.layout.local_combined.stat().st_size // 2 if self.layout.local_combined.exists() else 0
    cd = self.layout.drive_combined.stat().st_size // 2 if self.layout.drive_combined.exists() else 0
    cmatch = (cclaim == 0) or (cl == cclaim and (cd == 0 or cd == cclaim)) or (cl == 0 and cd == cclaim)
    ok &= cmatch
    lines.append(f'{"combined":<14s} {cclaim:>15,} {cl:>15,} {cd:>15,} {"OK" if cmatch else "MISMATCH":>8s}')
    return "\n".join(lines), bool(ok)
