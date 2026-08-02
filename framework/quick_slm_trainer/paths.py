"""Filesystem layout.

Every path the project touches is derived from a `Layout`. Nothing else in the
package names a directory.

The Drive root is still `quick-slm`. The model was renamed to Quick
SLM, but the notebooks that produced the live corpus and checkpoints hardcoded
the old name, and renaming the directory while a run was in flight would have
broken it. Changing `DEFAULT_DRIVE_ROOT` below, once the corpus and checkpoints
have been moved, is the whole of the rename.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/quick-slm")
DEFAULT_LOCAL_ROOT = Path("/content/quick-slm-local")

#: Order matters. It fixes the column order of every status table the package
#: prints, and the iteration order of the combine step's allocation.
SOURCE_KEYS = ("fineweb_edu", "finemath", "tool_calling", "starcoder")


@dataclass(frozen=True)
class Layout:
  """Where everything lives.

  Bulk binary writes go to `local_root`; `drive_root` only ever receives the
  small manifest, the tokenizer, and size-verified copies of finished `.bin`
  files. Drive's FUSE mount silently truncates large writes under load, and
  writing the 15 GB FineWeb-Edu shard straight to it cost a run. Keeping the
  two roots distinct makes that failure structurally impossible rather than
  merely unlikely.
  """

  drive_root: Path = DEFAULT_DRIVE_ROOT
  local_root: Path = DEFAULT_LOCAL_ROOT
  chunk_idx: int | None = None

  # --- Drive: small files and verified copies ---------------------------
  @property
  def data_dir(self) -> Path:
    return self.drive_root / "data"

  @property
  def ckpt_dir(self) -> Path:
    return self.drive_root / "checkpoints"

  @property
  def final_dir(self) -> Path:
    return self.drive_root / "final"

  @property
  def logs_dir(self) -> Path:
    return self.drive_root / "logs"

  @property
  def tokenizer_dir(self) -> Path:
    return self.drive_root / "tokenizer"

  @property
  def manifest_path(self) -> Path:
    if self.chunk_idx is not None:
      return self.data_dir / f"manifest_{self.chunk_idx:02d}.json"
    return self.data_dir / "manifest.json"

  @property
  def drive_combined(self) -> Path:
    if self.chunk_idx is not None:
      return self.data_dir / f"combined_{self.chunk_idx:02d}.bin"
    return self.data_dir / "combined.bin"

  # --- SFT artifacts ----------------------------------------------------
  @property
  def sft_dir(self) -> Path:
    return self.drive_root / "sft"

  @property
  def sft_raw_dir(self) -> Path:
    """Teacher output, one JSONL shard per category, before any filtering."""
    return self.sft_dir / "raw"

  @property
  def sft_ckpt_dir(self) -> Path:
    return self.drive_root / "sft_checkpoints"

  @property
  def sft_final_dir(self) -> Path:
    return self.drive_root / "sft_final"

  @property
  def sft_stats_path(self) -> Path:
    """Reject tallies and pack utilisation. The paper's data section reads this."""
    return self.sft_dir / "corpus_stats.json"

  def sft_raw(self, category: str) -> Path:
    """The raw teacher text for one category, on Drive. Append-only, resumable."""
    return self.sft_raw_dir / f"{category}.jsonl"

  @property
  def local_sft_raw_dir(self) -> Path:
    """Local staging for the teacher shards.

    The raw shard is the most expensive artifact in the SFT stage, a full day
    of teacher time, and until now it was the one written straight through the
    Drive FUSE mount, which the class docstring says silently truncates large
    writes under load. Generation appends here, on real local disk, and copies
    each finished category up to Drive with a line-count check. Same discipline
    the pretraining `.bin` files already use, for the same reason.
    """
    return self.local_root / "sft" / "raw"

  def local_sft_raw(self, category: str) -> Path:
    return self.local_sft_raw_dir / f"{category}.jsonl"

  def sft_bin(self, split: str, kind: str, *, stage: int | None = None) -> Path:
    """`kind` is 'ids' (uint16 token stream) or 'mask' (uint8 loss mask).

    `stage` names one leg of a multi-stage fine-tune, which uses to keep
    its two packed corpora apart. `None` is the single-corpus layout 
    wrote and must keep resolving to exactly the paths it produced.
    """
    if kind not in ("ids", "mask"):
      raise ValueError(f"kind must be 'ids' or 'mask', got {kind!r}")
    prefix = "sft" if stage is None else f"sft_stage{stage}"
    return self.sft_dir / f"{prefix}_{split}_{kind}.bin"

  # --- Local: primary write location ------------------------------------
  @property
  def local_data(self) -> Path:
    return self.local_root / "data"

  @property
  def local_combined(self) -> Path:
    if self.chunk_idx is not None:
      return self.local_data / f"combined_{self.chunk_idx:02d}.bin"
    return self.local_data / "combined.bin"

  def local_source(self, key: str) -> Path:
    if self.chunk_idx is not None:
      return self.local_data / f"{key}_{self.chunk_idx:02d}.bin"
    return self.local_data / f"{key}.bin"

  def drive_source(self, key: str) -> Path:
    if self.chunk_idx is not None:
      return self.data_dir / f"{key}_{self.chunk_idx:02d}.bin"
    return self.data_dir / f"{key}.bin"

  # --- Setup ------------------------------------------------------------
  def mkdirs(self) -> "Layout":
    for p in (
      self.data_dir,
      self.ckpt_dir,
      self.final_dir,
      self.logs_dir,
      self.tokenizer_dir,
      self.local_data,
    ):
      p.mkdir(parents=True, exist_ok=True)
    return self

  def mkdirs_sft(self) -> "Layout":
    for p in (
      self.sft_dir,
      self.sft_raw_dir,
      self.sft_ckpt_dir,
      self.sft_final_dir,
      self.local_sft_raw_dir,
    ):
      p.mkdir(parents=True, exist_ok=True)
    return self
