"""Near-duplicate removal by MinHash over the user request and the calls made.

`datasketch` would do this in ten lines. It is not a dependency because the
whole of it needed here is sixty lines of numpy, and because the permutation
seed has to be pinned for the corpus to be reproducible, which a library's
internal RNG does not promise across versions.

What gets hashed matters more than how. Hashing the whole rendered example would
compare `<think>` blocks, and two examples with identical user requests and
identical calls but differently-worded reasoning are duplicates for every purpose
the student cares about. Hashing the user text plus the call signatures catches
them.

Two things that look like content are not, and both once inflated the fingerprint:

 - The re-randomised entity ids. `conflict_specs._bid` mints `b_%03d`, a thousand
  values wide, and stamps it into the user turn and the call arguments. Two
  examples identical but for that id fell just under the 0.85 threshold and both
  survived. Stripping it, the 16,000 planned category-3 examples reduce to 981
  distinct documents, of which the old fingerprint kept 10,402.
 - `answer`'s `text`. `oracles.FREE_TEXT_ARGS` already declares it free prose that
  no comparison may depend on, and `canonical_call` excludes it when a branch is
  judged against its oracle. The fingerprint did not, so the same decision on the
  same request could be re-sold as a fresh example as often as the teacher could
  rephrase itself. Forty-four percent of category-3 branches resolve to `answer`.

The ids are stripped from the fingerprint rather than drawn from a small pool,
because the corpus wants them wide: a student that cannot memorise an id has to
copy it out of `<state>`, which is the behaviour category 3 exists to teach.
"""

from __future__ import annotations

import re
import zlib
from typing import Any, Callable, Sequence

import numpy as np

from .oracles import canonical_call

#: Mersenne prime. With 32-bit shingle hashes and coefficients below 2^31, the
#: product `a * h` stays under 2^63 and never wraps a uint64, so the modulus is
#: the modulus and not an accident of overflow.
_PRIME = np.uint64((1 << 61) - 1)
_MAX_COEFF = 1 << 31

_WORD = re.compile(r"[a-z0-9_]+")


def _permutations(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  a = rng.integers(1, _MAX_COEFF, size=num_perm, dtype=np.uint64)
  b = rng.integers(0, _MAX_COEFF, size=num_perm, dtype=np.uint64)
  return a, b


def shingles(text: str, k: int = 5) -> np.ndarray:
  """Unique 32-bit hashes of the word k-grams of `text`."""
  words = _WORD.findall(text.lower())
  if not words:
    return np.empty(0, dtype=np.uint64)
  if len(words) < k:
    grams = [" ".join(words)]
  else:
    grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
  return np.fromiter(
    {zlib.crc32(g.encode()) for g in grams}, dtype=np.uint64, count=-1
  )


def signature(text: str, a: np.ndarray, b: np.ndarray, *, k: int = 5) -> np.ndarray:
  h = shingles(text, k)
  if h.size == 0:
    # An empty document collides with nothing, including other empty ones.
    return np.full(a.size, _PRIME, dtype=np.uint64)
  return ((a[:, None] * h[None, :] + b[:, None]) % _PRIME).min(axis=1)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
  return float(np.mean(sig_a == sig_b))


def dedup_indices(
  texts: Sequence[str],
  *,
  threshold: float = 0.85,
  num_perm: int = 128,
  bands: int = 16,
  seed: int = 0xC0FFEE,
  progress: bool = False,
) -> list[int]:
  """Indices to keep, first occurrence wins.

  Banded LSH proposes candidates, and the estimated Jaccard confirms them, so
  the `threshold` in the signature is the threshold that is applied. LSH only
  ever costs recall, never precision.

  With 128 permutations in 16 bands of 8, the probability a pair is proposed is
  `1 - (1 - s**8)**16`. That curve's midpoint sits at s = 0.674, and at the
  0.85 actually enforced it reads 0.9938: about one true duplicate in 160 at
  exactly the threshold is missed, and essentially none above it. Halving
  `bands` to 8 drops recall at 0.85 to 0.92, which is the wrong trade for a
  pass that takes seconds.
  """
  if num_perm % bands:
    raise ValueError(f"bands={bands} does not divide num_perm={num_perm}")
  rows = num_perm // bands
  a, b = _permutations(num_perm, seed)

  iterator = enumerate(texts)
  if progress:
    from tqdm.auto import tqdm

    iterator = tqdm(iterator, total=len(texts), desc="dedup", unit="ex")

  buckets: dict[tuple[int, bytes], list[int]] = {}
  kept_sigs: list[np.ndarray] = []
  keep: list[int] = []

  for i, text in iterator:
    sig = signature(text, a, b)
    keys = [(band, sig[band * rows : (band + 1) * rows].tobytes()) for band in range(bands)]

    candidates: set[int] = set()
    for key in keys:
      candidates.update(buckets.get(key, ()))

    if any(estimated_jaccard(sig, kept_sigs[j]) >= threshold for j in candidates):
      continue

    slot = len(kept_sigs)
    kept_sigs.append(sig)
    keep.append(i)
    for key in keys:
      buckets.setdefault(key, []).append(slot)

  return keep


#: Entity ids that are re-randomised per example and mean nothing across examples.
#: Kept narrow on purpose: it must not touch a *quantity*, because `inventory_amount`
#: is the one spec whose two branches call the same tool and differ only in how much
#: they buy, and a normaliser that ate digits would collapse the sharpest pair in the
#: corpus into a self-duplicate. `tests/test_sft_pack.py` pins this against `_bid`.
_VOLATILE_ID = re.compile(r"\bb_\d+\b")


def normalise_ids(text: str) -> str:
  return _VOLATILE_ID.sub("<id>", text)


def _flatten(node: Any) -> str:
  if isinstance(node, tuple):
    return " ".join(_flatten(x) for x in node)
  return str(node)


def call_signature(call: dict) -> str:
  """The part of one call that is content, as flat text.

  Built from `oracles.canonical_call`, so the dedup pass and the pair-accept pass
  agree about what a call *is*. `answer`'s free prose drops out, key order stops
  mattering, and 500 stops differing from 500.0.
  """
  return _flatten(canonical_call(call))


def example_fingerprint(ex) -> str:
  """The text a duplicate check should compare: the request and the calls.

  Two examples that ask the same thing and answer with the same calls are
  duplicates however differently their reasoning is phrased, however the teacher
  worded the sentence it handed back, and whichever random building id the
  scenario happened to draw.
  """
  from ..template import AssistantTurn, UserTurn

  parts: list[str] = []
  for turn in ex.turns:
    if isinstance(turn, UserTurn):
      parts.append(turn.text)
    elif isinstance(turn, AssistantTurn):
      parts.append(" ".join(call_signature(c) for c in turn.calls))
  return normalise_ids("\n".join(parts))


def dedup_examples(
  examples: Sequence,
  *,
  threshold: float = 0.85,
  num_perm: int = 128,
  fingerprint: Callable[[object], str] = example_fingerprint,
  seed: int = 0xC0FFEE,
  progress: bool = False,
) -> list:
  """Drop near-duplicates, moving grouped examples together.

  Grouping matters for the counterfactual pairs. Their two branches share a
  user turn and differ only in the call, so an ungrouped pass would compare
  them against the branches of *other* pairs and could keep one while dropping
  its twin. A lone surviving branch is exactly the sample the pair construction
  exists to exclude, so a group is deduplicated as one document and kept or
  dropped whole.
  """
  from ..template import group_of

  groups: dict[str, list[int]] = {}
  for i, ex in enumerate(examples):
    key = group_of(ex) or f"\x00solo\x00{i}"
    groups.setdefault(key, []).append(i)

  keys = list(groups)
  # A group's fingerprint is its members' fingerprints in a fixed order, so two
  # pairs collide only when both of their branches collide.
  texts = ["\n".join(fingerprint(examples[i]) for i in groups[k]) for k in keys]

  kept = dedup_indices(
    texts, threshold=threshold, num_perm=num_perm, seed=seed, progress=progress
  )
  survivors = {keys[i] for i in kept}
  return [ex for i, ex in enumerate(examples) if (group_of(ex) or f"\x00solo\x00{i}") in survivors]
