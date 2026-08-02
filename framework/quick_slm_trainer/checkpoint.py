"""Atomic checkpoint save, resume, and pruning.

A checkpoint directory holds three things:

  training_state.pt  model, optimizer, RNG, step counters
  hf/         a `from_pretrained`-loadable snapshot plus the tokenizer
  config.json     the `Config` that produced it

The `hf/` copy is redundant with `training_state.pt` and worth its bytes anyway:
it is what the evaluation notebooks, the inference server, and the HuggingFace
upload all read, none of which should have to know this package exists.

Writes go to a dotted temporary directory and are renamed into place. A Colab
runtime dying mid-`torch.save` therefore leaves `.tmp_step_0004500` behind, not
a truncated `step_0004500` that the next run would happily resume from.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .model import unwrap


@dataclass
class TrainState:
  """Everything needed to restart a run exactly where it stopped."""

  step: int = 0
  tokens_seen: int = 0
  next_window: int = 0
  epoch: int = 0
  val_loss: float | None = None


def step_of(path: Path) -> int:
  return int(path.name.split("_")[1])


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
  ckpt_dir = Path(ckpt_dir)
  if not ckpt_dir.exists():
    return []
  return sorted(ckpt_dir.glob("step_*"), key=step_of)


def find_latest(ckpt_dir: str | Path) -> Path | None:
  cks = list_checkpoints(ckpt_dir)
  return cks[-1] if cks else None


def save(
  ckpt_dir: str | Path,
  *,
  step: int,
  state: TrainState,
  model,
  optimizer,
  config: Config,
  tokenizer=None,
  save_hf: bool = True,
) -> Path:
  import torch

  ckpt_dir = Path(ckpt_dir)
  dst = ckpt_dir / f"step_{step:07d}"
  tmp = ckpt_dir / f".tmp_step_{step:07d}"
  if tmp.exists():
    shutil.rmtree(tmp)
  tmp.mkdir(parents=True)

  core = unwrap(model)
  torch.save(
    {
      "step": step,
      "tokens_seen": state.tokens_seen,
      "next_window": state.next_window,
      "epoch": state.epoch,
      "val_loss": state.val_loss,
      "model": core.state_dict(),
      "optimizer": optimizer.state_dict(),
      "cpu_rng": torch.get_rng_state(),
      "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
      "created_at": time.time(),
    },
    tmp / "training_state.pt",
  )

  config.save(tmp / "config.json")

  if save_hf:
    core.save_pretrained(str(tmp / "hf"), safe_serialization=True)
    if tokenizer is not None:
      tokenizer.save_pretrained(str(tmp / "hf"))

  if dst.exists():
    shutil.rmtree(dst)
  tmp.rename(dst)
  return dst


def load_for_resume(path: str | Path, model, optimizer=None, *, map_location: str = "cpu") -> TrainState:
  """Restore weights, optimizer, and RNG from a checkpoint directory."""
  import torch

  path = Path(path)
  # The file was written by `save` above, so it holds tensors and plain
  # scalars only. `weights_only=False` is stated rather than inherited because
  # the torch default flipped and the optimizer state would stop loading.
  blob = torch.load(path / "training_state.pt", map_location=map_location, weights_only=False)

  unwrap(model).load_state_dict(blob["model"])
  if optimizer is not None:
    optimizer.load_state_dict(blob["optimizer"])

  torch.set_rng_state(blob["cpu_rng"].cpu() if hasattr(blob["cpu_rng"], "cpu") else blob["cpu_rng"])
  if torch.cuda.is_available() and blob.get("cuda_rng") is not None:
    torch.cuda.set_rng_state(blob["cuda_rng"].cpu())

  return TrainState(
    step=int(blob["step"]),
    tokens_seen=int(blob["tokens_seen"]),
    next_window=int(blob.get("next_window", 0)),
    epoch=int(blob.get("epoch", 0)),
    val_loss=blob.get("val_loss"),
  )


def prune(
  ckpt_dir: str | Path,
  *,
  keep_last_n: int,
  total_steps: int,
  save_every: int,
  deciles: bool = True,
) -> list[Path]:
  """Delete all but the last N checkpoints and the decile milestones.

  A milestone is kept when its step falls within one save interval of a tenth
  of the run, which is the closest any saved step can land to an exact decile.
  Returns the deleted paths.
  """
  cks = list_checkpoints(ckpt_dir)
  if len(cks) <= keep_last_n:
    return []

  keep = set(cks[-keep_last_n:])
  if deciles and total_steps > 0:
    targets = {round(total_steps * d / 10) for d in range(1, 11)}
    for c in cks:
      s = step_of(c)
      if any(abs(s - t) < save_every for t in targets):
        keep.add(c)

  removed = []
  for c in cks:
    if c not in keep:
      shutil.rmtree(c, ignore_errors=True)
      removed.append(c)
  return removed


def save_final(final_dir: str | Path, model, tokenizer=None, config: Config | None = None) -> Path:
  """Write the HuggingFace-format artifact the inference server loads."""
  final_dir = Path(final_dir)
  final_dir.mkdir(parents=True, exist_ok=True)
  unwrap(model).save_pretrained(str(final_dir), safe_serialization=True)
  if tokenizer is not None:
    tokenizer.save_pretrained(str(final_dir))
  if config is not None:
    config.save(final_dir / "quick_slm_config.json")
  return final_dir
