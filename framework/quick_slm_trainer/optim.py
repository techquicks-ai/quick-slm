"""AdamW with the conventional two-group weight-decay split."""

from __future__ import annotations

from .config import OptimConfig


def split_decay_groups(model) -> tuple[list, list]:
  """Partition parameters into (decayed, not-decayed).

  Anything with fewer than two dimensions is a bias, a gain, or an RMSNorm
  weight. Decaying those pulls a normalisation scale toward zero, which the
  layer then has to fight, so they are excluded. Embeddings are two
  dimensional and are decayed, which is standard for LLaMA-family recipes.

  `named_parameters` deduplicates shared tensors, so with `tie_word_embeddings`
  the single tied matrix is visited once and lands in exactly one group.
  """
  decay, no_decay = [], []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
      no_decay.append(p)
    else:
      decay.append(p)
  return decay, no_decay


def build_optimizer(model, cfg: OptimConfig, *, device: str = "cuda"):
  """AdamW. `fused` is silently dropped off CUDA, where torch rejects it."""
  import torch

  decay, no_decay = split_decay_groups(model)
  fused = bool(cfg.fused) and device.startswith("cuda") and torch.cuda.is_available()
  return torch.optim.AdamW(
    [
      {"params": decay, "weight_decay": cfg.weight_decay},
      {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=cfg.lr_peak,
    betas=tuple(cfg.betas),
    eps=cfg.eps,
    fused=fused,
  )
