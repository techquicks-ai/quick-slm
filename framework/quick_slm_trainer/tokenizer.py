"""Tokenizer construction and the reserved special-token list.

Fourteen domain tokens are added to the LLaMA-2 vocabulary before pretraining
so that every structural block the model will ever see has a dedicated
single-token slot, at any stage, without an embedding-resize migration later.
Cost is ~10 x 576 additional parameters for the ten tokens that pretraining
never emits, which is invisible against the 103M budget.

A caution worth recording, because the paper's Section 5 raises it and this
module is where a reader will come looking: the claim that a reserved row
"stays near random initialisation" through pretraining is probably false. Weights
are tied, so the row is also an output-embedding row, and the softmax
denominator hands it gradient on every single step as a negative example. Its
logit gets driven down. Whether its *norm* stays near initialisation is an
empirical question that has not been measured.
"""

from __future__ import annotations

from pathlib import Path

from .template import (
  CALL_CLOSE,
  CALL_OPEN,
  IM_END,
  IM_START,
  MEMORY_CLOSE,
  MEMORY_OPEN,
  RESPONSE_CLOSE,
  RESPONSE_OPEN,
  STATE_CLOSE,
  STATE_OPEN,
  THINK_CLOSE,
  THINK_OPEN,
  TOOL_CLOSE,
  TOOL_OPEN,
)

#: Order is fixed. It determines the id assigned to each token, and the ids are
#: baked into the base checkpoint's embedding matrix. Appending is safe;
#: reordering or inserting is not.
SPECIAL_TOKENS: tuple[str, ...] = (
  # Seen during pretraining, in the xLAM slice, and at every inference call.
  TOOL_OPEN,
  TOOL_CLOSE,
  RESPONSE_OPEN,
  RESPONSE_CLOSE,
  CALL_OPEN,
  CALL_CLOSE,
  # Injected by the inference server. Reserved from day zero.
  STATE_OPEN,
  STATE_CLOSE,
  MEMORY_OPEN,
  MEMORY_CLOSE,
  # Introduced at SFT.
  THINK_OPEN,
  THINK_CLOSE,
  IM_START,
  IM_END,
)

EXPECTED_VOCAB_SIZE = 32_000 + len(SPECIAL_TOKENS) # 32_014


def missing_special_tokens(tok) -> list[str]:
  vocab = tok.get_vocab()
  return [t for t in SPECIAL_TOKENS if t not in vocab]


def load_tokenizer(tokenizer_dir: str | Path, *, patch: bool = True):
  """Load the saved tokenizer, optionally adding any special token it lacks.

  Patching exists so the token list can be extended without deleting the
  directory. It cannot repair a *reordering*, and it should not try to.
  """
  from transformers import AutoTokenizer

  tokenizer_dir = Path(tokenizer_dir)
  tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
  missing = missing_special_tokens(tok)
  if missing and patch:
    tok.add_special_tokens({"additional_special_tokens": missing})
    tok.save_pretrained(str(tokenizer_dir))
  elif missing:
    raise ValueError(f"tokenizer at {tokenizer_dir} is missing {missing}")
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token
  return tok


def build_tokenizer(repo: str, save_to: str | Path):
  """Fetch the base tokenizer, add the special tokens, save."""
  from transformers import AutoTokenizer

  save_to = Path(save_to)
  tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
  tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token
  save_to.mkdir(parents=True, exist_ok=True)
  tok.save_pretrained(str(save_to))
  return tok


def get_or_build_tokenizer(repo: str, tokenizer_dir: str | Path):
  """Load from disk if present, otherwise build and save. Idempotent."""
  tokenizer_dir = Path(tokenizer_dir)
  if (tokenizer_dir / "tokenizer.json").exists():
    return load_tokenizer(tokenizer_dir, patch=True)
  return build_tokenizer(repo, tokenizer_dir)


def special_token_ids(tok) -> dict[str, int]:
  return {t: tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
