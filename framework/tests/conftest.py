"""Shared fixtures.

`FakeTokenizer` is a character-level stand-in with one property that matters:
it splits its input on the added special tokens, exactly as a HuggingFace fast
tokenizer does. `template.encode` relies on that behaviour to guarantee that
encoding segments separately and concatenating yields the same ids as encoding
the whole string, and a fake that merged across boundaries would let a broken
`encode` pass.
"""

from __future__ import annotations

import re

import pytest

from quick_slm_trainer.tokenizer import SPECIAL_TOKENS


class FakeTokenizer:
  """One id per character, plus one id per special token."""

  def __init__(self) -> None:
    self._specials = {tok: 1000 + i for i, tok in enumerate(SPECIAL_TOKENS)}
    self._split = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
    self.eos_token_id = 2
    self.pad_token_id = 2
    self.bos_token_id = 1

  def __len__(self) -> int:
    return 1000 + len(SPECIAL_TOKENS)

  def _encode_one(self, text: str) -> list[int]:
    ids: list[int] = []
    for piece in self._split.split(text):
      if not piece:
        continue
      if piece in self._specials:
        ids.append(self._specials[piece])
      else:
        ids.extend(ord(c) % 900 + 10 for c in piece)
    return ids

  def __call__(self, text, add_special_tokens=False, **kwargs):
    if isinstance(text, str):
      return {"input_ids": self._encode_one(text)}
    return {"input_ids": [self._encode_one(t) for t in text]}


@pytest.fixture
def tok() -> FakeTokenizer:
  return FakeTokenizer()
