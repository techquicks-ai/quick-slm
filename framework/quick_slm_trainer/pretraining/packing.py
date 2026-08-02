"""Stream documents from HuggingFace, tokenize, append to a uint16 `.bin`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..config import DataConfig
from ..manifest import Manifest
from ..paths import Layout


def encode_batch(tok, texts: list[str], eos_id: int) -> np.ndarray:
  """Tokenize a batch and terminate each document with EOS.

  `add_special_tokens=False` because the EOS is appended by hand and a BOS at
  the head of every document would be noise once documents are concatenated
  into fixed windows.
  """
  enc = tok(
    texts,
    add_special_tokens=False,
    return_attention_mask=False,
    return_token_type_ids=False,
  )
  out: list[int] = []
  for ids in enc["input_ids"]:
    out.extend(ids)
    out.append(eos_id)
  return np.asarray(out, dtype=np.uint16)


def sync_local_to_drive(layout: Layout, key: str, *, verify: bool = True) -> bool:
  """Copy a finished source to Drive, checking the byte count survived.

  Drive's FUSE mount can return from a copy having written fewer bytes than
  the source holds, without raising. One retry, then the local file stays the
  source of truth rather than a short file being blessed.
  """
  src, dst = layout.local_source(key), layout.drive_source(key)
  if not src.exists():
    print(f" [{key}] no local file to sync")
    return False

  src_size = src.stat().st_size
  print(f" [{key}] syncing local to Drive ({src_size / 1e9:.2f} GB)")
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(src, dst)
  if not verify:
    return True

  for attempt in (1, 2):
    if dst.stat().st_size == src_size:
      print(f" [{key}] sync verified ({src_size / 1e9:.2f} GB)")
      return True
    print(f" [{key}] copy incomplete: local {src_size:,} vs Drive {dst.stat().st_size:,}")
    if attempt == 1:
      print(f" [{key}] retrying once")
      shutil.copy(src, dst)
  print(f" [{key}] copy still incomplete; local file remains the source of truth")
  return False


def stream_source(
  *,
  layout: Layout,
  manifest: Manifest,
  cfg: DataConfig,
  tok,
  key: str,
  documents: Iterable,
  text_fn: Callable[[object], str | None],
  budget: int,
  progress: bool = True,
) -> int:
  """Tokenize `documents` into `local_source(key)` until `budget` tokens land.

  Resumable at batch granularity: the manifest is rewritten and the file
  fsynced every `flush_every` batches, so a disconnect costs at most one
  batch. Returns the total token count for the source.
  """
  manifest.reconcile(key)
  written = manifest.tokens_written(key)

  if manifest.is_done(key):
    print(f"[{key}] already done: {written:,} tokens")
    local, drive = layout.local_source(key), layout.drive_source(key)
    if not drive.exists() or drive.stat().st_size != local.stat().st_size:
      sync_local_to_drive(layout, key)
    return written

  print(f"[{key}] resuming at {written:,} / target {budget:,}")
  path = layout.local_source(key)
  path.parent.mkdir(parents=True, exist_ok=True)
  eos_id = tok.eos_token_id

  bar = None
  if progress:
    from tqdm.auto import tqdm

    bar = tqdm(total=budget, initial=written, unit="tok", unit_scale=True,
          smoothing=0.05, desc=key)

  fh = open(path, "ab" if path.exists() else "wb")
  buf: list[str] = []
  since_flush = 0

  def _flush_buf() -> None:
    nonlocal written, buf
    arr = encode_batch(tok, buf, eos_id)
    buf = []
    need = budget - written
    if arr.size > need:
      arr = arr[:need]
    arr.tofile(fh)
    written += int(arr.size)
    if bar is not None:
      bar.update(int(arr.size))

  rows_read = manifest.source(key).get("rows_read", 0)

  try:
    for row in documents:
      if written >= budget:
        break
      rows_read += 1
      text = text_fn(row)
      if not text:
        continue
      buf.append(text)
      if len(buf) < cfg.batch_docs:
        continue
      _flush_buf()
      since_flush += 1
      if since_flush >= cfg.flush_every:
        fh.flush()
        os.fsync(fh.fileno())
        manifest.set_progress(key, written, rows_read=rows_read)
        manifest.save()
        since_flush = 0
    if buf and written < budget:
      _flush_buf()
  finally:
    fh.flush()
    os.fsync(fh.fileno())
    fh.close()
    if bar is not None:
      bar.close()

  manifest.set_progress(key, written, rows_read=rows_read)
  manifest.save()
  print(f"[{key}] wrote {written:,} tokens, done={manifest.is_done(key)}")

  if manifest.is_done(key):
    sync_local_to_drive(layout, key)
  return written
