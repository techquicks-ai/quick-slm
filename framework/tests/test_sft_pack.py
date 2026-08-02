"""Dedup and packing.

The packing invariant that matters: an example is never split across a window
boundary. Splitting one cuts an assistant turn mid-JSON, scores the truncated
half, and teaches the model to emit malformed structured output, which is the
one thing this stage exists to prevent.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from quick_slm_trainer.config import SFTConfig
from quick_slm_trainer.sft.dedup import (
  _VOLATILE_ID,
  call_signature,
  dedup_examples,
  dedup_indices,
  estimated_jaccard,
  example_fingerprint,
  normalise_ids,
  shingles,
  signature,
  _permutations,
)
from quick_slm_trainer.sft.pack import bin_pack, encode_examples, pack, split_examples
from quick_slm_trainer.template import AssistantTurn, Example, UserTurn

TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}, "required": []}}]


def example(user: str, city: str = "Tokyo", think: str = "reasoning about it") -> Example:
  return Example(
    tools=TOOLS,
    turns=[UserTurn(user), AssistantTurn(think, [{"name": "get_weather", "arguments": {"city": city}}])],
    category="single_stage",
  )


# ==========================================================================
# Dedup
# ==========================================================================
def test_identical_texts_collapse_to_one():
  keep = dedup_indices(["the quick brown fox jumps over the lazy dog"] * 5)
  assert keep == [0]


def test_unrelated_texts_all_survive():
  texts = [
    "the weather in tokyo is mild today and pleasant",
    "iron smelter b_004 is paused and needs resuming now",
    "convert forty dollars into euros for the trip",
  ]
  assert dedup_indices(texts) == [0, 1, 2]


def test_first_occurrence_wins():
  keep = dedup_indices(["alpha beta gamma delta epsilon zeta", "x y z q r s", "alpha beta gamma delta epsilon zeta"])
  assert keep == [0, 1]


# A pair differing only in the final word. Long enough that its true Jaccard is
# 0.92, where the band curve's recall is 0.99999. The obvious 13-word version has
# a Jaccard of 0.82, where recall is 0.972, and the test would flake once in 36
# runs on a change of permutation seed.
_NEAR_A = (
  "please buy twelve units of iron ore from the market right now because the "
  "smelter is running low and production will stall before the next shift ends today"
)
_NEAR_B = _NEAR_A.replace("today", "tonight")


def test_a_near_duplicate_is_caught():
  assert dedup_indices([_NEAR_A, _NEAR_B], threshold=0.6) == [0]


def test_lsh_proposes_but_the_estimated_jaccard_decides():
  # The pair above collides in a band and is proposed. At a threshold above its
  # similarity it must still survive, or the enforced threshold is a fiction
  # and LSH's band curve is silently doing the filtering.
  #
  # 0.999 rather than 0.95: the estimate is a mean of 128 Bernoulli draws, so
  # its standard error at a true 0.92 is 0.024, and a 0.95 threshold would sit
  # only 1.25 standard errors away and flake one run in ten.
  assert dedup_indices([_NEAR_A, _NEAR_B], threshold=0.999) == [0, 1]


def test_signature_of_identical_text_is_identical():
  a, b = _permutations(128, 1)
  text = "some words that will be shingled into five grams"
  assert np.array_equal(signature(text, a, b), signature(text, a, b))


def test_signature_is_never_a_negative_or_wrapped_value():
  # `a * h` must not overflow uint64, or the modulus stops being a modulus.
  a, b = _permutations(128, 2)
  sig = signature("a b c d e f g h i j k l m n o p", a, b)
  assert sig.dtype == np.uint64
  assert (sig < np.uint64((1 << 61) - 1)).all()


def test_jaccard_of_a_text_with_itself_is_one():
  a, b = _permutations(128, 3)
  sig = signature("the quick brown fox jumps over the lazy dog", a, b)
  assert estimated_jaccard(sig, sig) == 1.0


def test_shingles_of_a_short_text_do_not_vanish():
  assert shingles("two words", k=5).size == 1


def test_empty_text_produces_a_signature_that_matches_nothing():
  a, b = _permutations(128, 4)
  s1, s2 = signature("", a, b), signature("", a, b)
  # Two empty documents have identical signatures, so dedup collapses them.
  # That is correct: they carry no content to distinguish.
  assert np.array_equal(s1, s2)


def test_bands_must_divide_num_perm():
  with pytest.raises(ValueError, match="does not divide"):
    dedup_indices(["a"], num_perm=128, bands=7)


def test_fingerprint_ignores_the_think_block():
  # Two examples asking the same thing with the same call are duplicates
  # however differently the reasoning is worded.
  a = example("weather in Tokyo?", think="one way of putting it entirely")
  b = example("weather in Tokyo?", think="a completely different phrasing here")
  assert example_fingerprint(a) == example_fingerprint(b)
  assert len(dedup_examples([a, b])) == 1


def test_fingerprint_separates_different_calls():
  a = example("weather in Tokyo?", city="Tokyo")
  b = example("weather in Osaka?", city="Osaka")
  assert len(dedup_examples([a, b])) == 2


# --------------------------------------------------------------------------
# What the fingerprint must refuse to count as content
# --------------------------------------------------------------------------
def test_the_volatile_id_pattern_matches_what_the_specs_actually_mint():
  # `dedup` normalises an id format that `conflict_specs` owns. Pin them together,
  # so changing `_bid` fails here rather than silently restoring a thousand-way
  # multiplier on apparent diversity.
  import random as _random

  from quick_slm_trainer.sft.conflict_specs import _bid

  for seed in range(50):
    assert _VOLATILE_ID.fullmatch(_bid(_random.Random(seed))), seed


def test_the_fingerprint_ignores_the_random_building_id():
  # Two examples identical but for the id sat just under the 0.85 threshold and
  # both survived. That is how 16,000 category-3 examples kept 10,402 of 981
  # distinct documents.
  assert normalise_ids("Make sure b_427 feeds b_133.") == "Make sure <id> feeds <id>."

  a = Example(tools=TOOLS, turns=[UserTurn("restart it"),
    AssistantTurn("t", [{"name": "resume_building", "arguments": {"building_id": "b_427"}}])],
    category="state_memory_conflict")
  b = Example(tools=TOOLS, turns=[UserTurn("restart it"),
    AssistantTurn("t", [{"name": "resume_building", "arguments": {"building_id": "b_003"}}])],
    category="state_memory_conflict")
  assert example_fingerprint(a) == example_fingerprint(b)
  assert len(dedup_examples([a, b])) == 1


def test_the_fingerprint_ignores_the_prose_inside_answer():
  # `oracles.FREE_TEXT_ARGS` already says no comparison may depend on it, and
  # `canonical_call` honours that when a branch is judged against its oracle.
  # 44% of category-3 branches resolve to `answer`; counting the teacher's
  # sentence as content lets one decision be re-sold as many examples.
  a = Example(tools=TOOLS, turns=[UserTurn("is it running?"),
    AssistantTurn("t", [{"name": "answer", "arguments": {"text": "The smelter is already running."}}])],
    category="state_memory_conflict")
  b = Example(tools=TOOLS, turns=[UserTurn("is it running?"),
    AssistantTurn("t", [{"name": "answer", "arguments": {"text": "Yes, it has been up the whole time."}}])],
    category="state_memory_conflict")
  assert example_fingerprint(a) == example_fingerprint(b)
  assert len(dedup_examples([a, b])) == 1


def test_the_fingerprint_still_separates_the_decisive_quantity():
  # The guard on the normaliser. `inventory_amount` is the one spec whose two
  # branches call the same tool and differ only in how much they buy, and
  # `SFT_README.md` calls it the sharpest of the eight. A normaliser that ate
  # digits would turn that pair into a self-duplicate and delete the only spec
  # a tool-guessing student cannot beat by coin flip.
  def buy(qty):
    return Example(tools=TOOLS, turns=[UserTurn("Top the iron_ore up to 500."),
      AssistantTurn("t", [{"name": "buy", "arguments": {"resource_id": "iron_ore", "quantity": qty}}])],
      category="state_memory_conflict")

  assert example_fingerprint(buy(380)) != example_fingerprint(buy(20))
  assert len(dedup_examples([buy(380), buy(20)])) == 2


def test_call_signature_agrees_with_the_oracles_notion_of_a_call():
  # Same policy object, so dedup and pair-accept cannot drift apart.
  assert call_signature({"name": "answer", "arguments": {"text": "anything at all"}}) == call_signature(
    {"name": "answer", "arguments": {"text": "something else entirely"}}
  )
  # 500 and 500.0 are one quantity; key order is not content.
  assert call_signature({"name": "buy", "arguments": {"resource_id": "coal", "quantity": 500}}) == call_signature(
    {"name": "buy", "arguments": {"quantity": 500.0, "resource_id": "coal"}}
  )
  # `toggle_power(on=False)` must not equal `toggle_power(on=0)`.
  assert call_signature({"name": "toggle_power", "arguments": {"building_id": "x", "on": False}}) != call_signature(
    {"name": "toggle_power", "arguments": {"building_id": "x", "on": 0}}
  )


# ==========================================================================
# bin_pack
# ==========================================================================
def test_every_example_lands_in_exactly_one_bin():
  rng = random.Random(0)
  lengths = [rng.randint(100, 900) for _ in range(500)]
  bins = bin_pack(lengths, 1024, rng=random.Random(1))
  placed = [i for b in bins for i in b]
  assert sorted(placed) == list(range(len(lengths)))


def test_no_bin_overflows_the_context():
  rng = random.Random(2)
  lengths = [rng.randint(100, 900) for _ in range(500)]
  ctx = 1024
  for b in bin_pack(lengths, ctx, rng=random.Random(3)):
    assert sum(lengths[i] for i in b) <= ctx


def test_an_example_exactly_the_context_length_gets_its_own_bin():
  bins = bin_pack([1024, 5, 5], 1024, rng=random.Random(4))
  full = [b for b in bins if 0 in b]
  assert full == [[0]]


def test_utilisation_is_high_on_the_realistic_length_distribution():
  # The length filter admits 100 to 3000 tokens, with most examples near 750.
  rng = random.Random(5)
  lengths = [max(100, min(3000, int(rng.lognormvariate(6.6, 0.55)))) for _ in range(4000)]
  bins = bin_pack(lengths, 4096, rng=random.Random(6))
  assert sum(lengths) / (len(bins) * 4096) > 0.98


def test_utilisation_survives_a_uniform_length_distribution():
  # The default `max_open_bins` was raised from 32 because this case fell to
  # 84%: within a descending chunk the small items arrive last, and the bins
  # that could take them had already been retired.
  rng = random.Random(8)
  lengths = [rng.randint(100, 3000) for _ in range(4000)]
  bins = bin_pack(lengths, 4096, rng=random.Random(9))
  assert sum(lengths) / (len(bins) * 4096) > 0.95


def test_chunking_keeps_bins_heterogeneous_in_length():
  # A global descending sort packs marginally better and puts every long
  # example in an early bin. Length correlates with category, so windows would
  # come out category-homogeneous. Chunked sorting is what prevents that.
  #
  # The gap widens with corpus size, because a global sort's spread is roughly
  # `max_open_bins / n_bins`. Measured on the real 42k corpus it is 0.26 against
  # 0.006, a factor of 44. At the 20k used here it is a factor of 16.
  rng = random.Random(10)
  lengths = [max(100, min(3000, int(rng.lognormvariate(6.6, 0.55)))) for _ in range(20_000)]
  rank = {i: r / len(lengths) for r, i in enumerate(sorted(range(len(lengths)), key=lambda i: lengths[i]))}

  def spread(chunk):
    bins = bin_pack(lengths, 4096, rng=random.Random(11), chunk=chunk)
    multi = [b for b in bins if len(b) > 1]
    return sum(max(rank[i] for i in b) - min(rank[i] for i in b) for b in multi) / len(multi)

  chunked, global_sort = spread(chunk=256), spread(chunk=len(lengths))
  assert chunked > 0.20
  assert chunked > 5 * global_sort


def test_bin_pack_of_nothing_is_no_bins():
  assert bin_pack([], 1024, rng=random.Random(7)) == []


def test_an_item_larger_than_ctx_would_be_unpackable_so_the_filter_drops_it_first():
  # `encode_examples` rejects anything above `min(max_example_tokens, ctx)`,
  # which is what makes the no-bin-overflows invariant reachable at all.
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=10_000, ctx=8)
  assert cfg.max_example_tokens > cfg.ctx


# ==========================================================================
# pack / encode
# ==========================================================================
# `FakeTokenizer` is character-level, so `example()` encodes to ~256 tokens: the
# tools block dominates, and the user string barely moves it. Any cap below that
# drops every example, which leaves `encoded` empty and quietly turns the packing
# assertions below into loops over nothing. These bounds keep examples packable
# and let ~3 of them share a window, which is what makes the whole-example and
# padding invariants worth asserting.
PACKABLE_CTX = 1024
PACKABLE_MAX = 400


def assert_packing_is_exercised(stats, n_examples: int) -> None:
  """Guard against the vacuous pass: every example survived to be packed."""
  assert stats.dropped_too_long == 0, "cap is below the encoded example length"
  assert stats.dropped_too_short == 0
  assert stats.examples_packed == n_examples


def test_encode_drops_examples_outside_the_length_bounds(tok):
  cfg = SFTConfig(min_example_tokens=10_000, max_example_tokens=20_000, ctx=32_000)
  encoded, stats = encode_examples([example("hi")], tok, cfg, progress=False)
  assert encoded == [] and stats.dropped_too_short == 1

  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=5, ctx=4096)
  encoded, stats = encode_examples([example("hi")], tok, cfg, progress=False)
  assert encoded == [] and stats.dropped_too_long == 1


def test_encode_never_emits_an_example_longer_than_ctx(tok):
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=10_000, ctx=64)
  encoded, stats = encode_examples([example("hi")], tok, cfg, progress=False)
  assert encoded == [] and stats.dropped_too_long == 1


def test_encode_produces_a_mask_that_marks_only_the_assistant_span(tok):
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=10_000, ctx=4096)
  encoded, _ = encode_examples([example("hi")], tok, cfg, progress=False)
  ids, mask = encoded[0]
  assert len(ids) == len(mask)
  assert mask.any() and not mask.all()
  # The scored region is a single contiguous suffix-adjacent span.
  on = np.flatnonzero(mask)
  assert np.array_equal(on, np.arange(on[0], on[-1] + 1))


def test_pack_writes_whole_examples_and_pads_the_rest(tok):
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=PACKABLE_MAX, ctx=PACKABLE_CTX)
  examples = [example(f"request number {i}") for i in range(20)]
  encoded, stats = encode_examples(examples, tok, cfg, progress=False)
  assert_packing_is_exercised(stats, len(examples))
  ids, mask = pack(encoded, ctx=cfg.ctx, eos_id=tok.eos_token_id, stats=stats)

  assert ids.dtype == np.uint16 and mask.dtype == np.uint8
  assert len(ids) == len(mask) == stats.windows * cfg.ctx
  assert stats.real_tokens == sum(len(i) for i, _ in encoded)
  assert stats.scored_tokens == sum(int(m.sum()) for _, m in encoded)
  assert stats.examples_packed == len(encoded)

  # The real tokens form a contiguous prefix of every window and the padding
  # sits at the tail. An example split across a boundary would leave pad bytes
  # stranded in the middle of a window.
  for window in ids.reshape(stats.windows, cfg.ctx):
    real = np.flatnonzero(window != tok.eos_token_id)
    assert real.size, "an empty window was emitted"
    assert real[0] == 0
    assert np.array_equal(real, np.arange(real[0], real[-1] + 1))


def test_packed_padding_is_never_scored(tok):
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=PACKABLE_MAX, ctx=PACKABLE_CTX)
  encoded, stats = encode_examples([example(f"r{i}") for i in range(9)], tok, cfg, progress=False)
  assert_packing_is_exercised(stats, 9)
  ids, mask = pack(encoded, ctx=cfg.ctx, eos_id=tok.eos_token_id, stats=stats)
  pad_positions = np.flatnonzero((ids == tok.eos_token_id) & (mask == 1))
  assert pad_positions.size == 0


def test_pack_stats_utilisation_is_a_fraction(tok):
  cfg = SFTConfig(min_example_tokens=1, max_example_tokens=PACKABLE_MAX, ctx=PACKABLE_CTX)
  encoded, stats = encode_examples([example(f"r{i}") for i in range(30)], tok, cfg, progress=False)
  assert_packing_is_exercised(stats, 30)
  pack(encoded, ctx=cfg.ctx, eos_id=tok.eos_token_id, stats=stats)
  assert 0.0 < stats.utilisation <= 1.0
  assert 0.0 < stats.scored_fraction < 1.0


def test_encode_rejects_a_tokenizer_too_large_for_uint16(tok):
  class Big:
    def __len__(self):
      return 70_000

  cfg = SFTConfig()
  with pytest.raises(ValueError, match="uint16"):
    encode_examples([], Big(), cfg, progress=False)


# ==========================================================================
# split
# ==========================================================================
def test_split_is_at_the_example_level_and_partitions_exactly():
  examples = [example(f"r{i}") for i in range(100)]
  train, val = split_examples(examples, val_fraction=0.05)
  assert len(val) == 5 and len(train) == 95
  assert len(train) + len(val) == len(examples)


def test_split_is_deterministic_given_the_seed():
  examples = [example(f"r{i}") for i in range(50)]
  a, _ = split_examples(examples, val_fraction=0.2, seed=3)
  b, _ = split_examples(examples, val_fraction=0.2, seed=3)
  assert [e.turns[0].text for e in a] == [e.turns[0].text for e in b]


def test_split_shares_no_example_between_train_and_val():
  examples = [example(f"r{i}") for i in range(60)]
  train, val = split_examples(examples, val_fraction=0.1)
  train_texts = {e.turns[0].text for e in train}
  val_texts = {e.turns[0].text for e in val}
  assert not (train_texts & val_texts)


# ==========================================================================
# Grouping: the two branches of a counterfactual pair move together
# ==========================================================================
def conflict_examples(n_pairs: int = 30, prefix: str = "p"):
  import random as _random

  from quick_slm_trainer.sft import conflict
  from quick_slm_trainer.sft.conflict_specs import ALL_SPECS

  out = []
  for seed in range(n_pairs):
    pair = conflict.build_pair(
      ALL_SPECS["building_run"], _random.Random(seed), pair_id=f"{prefix}{seed}"
    )
    for label in ("a", "b"):
      out.append(
        conflict.example_from_branch(pair, label, "reads the state before acting", [pair.branch(label).call])
      )
  return out


def _pairs_in(examples):
  seen: dict[str, set[str]] = {}
  for ex in examples:
    if ex.category == "state_memory_conflict":
      seen.setdefault(ex.meta["pair_id"], set()).add(ex.meta["branch"])
  return seen


def test_a_pair_never_straddles_the_train_val_boundary():
  # One branch in val and its twin in train is a validation example whose
  # near-identical partner the model trained on.
  mixed = conflict_examples() + [example(f"r{i}") for i in range(90)]
  random.Random(0).shuffle(mixed)

  train, val = split_examples(mixed, val_fraction=0.1)
  assert not set(_pairs_in(train)) & set(_pairs_in(val))
  assert all(branches == {"a", "b"} for branches in _pairs_in(train).values())
  assert all(branches == {"a", "b"} for branches in _pairs_in(val).values())
  assert len(train) + len(val) == len(mixed)


def test_the_split_still_hits_its_target_fraction_with_groups_present():
  mixed = conflict_examples() + [example(f"r{i}") for i in range(90)]
  _, val = split_examples(mixed, val_fraction=0.1)
  assert abs(len(val) - round(len(mixed) * 0.1)) <= 2


def test_dedup_keeps_or_drops_a_pair_whole():
  # A lone surviving branch is exactly the sample the pair was built to exclude:
  # in isolation it cannot be told apart from a lucky pattern match. The two
  # batches share seeds, so their content is identical and only the pair ids
  # differ; with the ids stripped from the fingerprint the second batch is a pure
  # duplicate of the first and must add nothing.
  first = conflict_examples(n_pairs=6, prefix="p")
  second = conflict_examples(n_pairs=6, prefix="q") # same content, other ids
  kept = dedup_examples(first + second, threshold=0.85)

  # The load-bearing invariant: a pair is kept or dropped as a unit, never split.
  assert all(branches == {"a", "b"} for branches in _pairs_in(kept).values())

  # The second batch collapses entirely into the first: dedup keeps exactly the
  # distinct pairs of one batch, no more. (The distinct count is below 6 because
  # the stripped fingerprint now recognises two same-type, same-phrasing pairs as
  # the duplicates they always were.)
  kept_one_batch = dedup_examples(first, threshold=0.85)
  assert len(kept) == len(kept_one_batch)
  assert len(_pairs_in(kept)) == len(_pairs_in(kept_one_batch))
  assert len(kept) < len(first) + len(second) # dedup did real work


def test_the_two_branches_of_a_pair_have_different_fingerprints():
  # Same user turn, different calls. The group is what keeps them together, not
  # a fingerprint collision.
  a, b = conflict_examples(n_pairs=1)
  assert example_fingerprint(a) != example_fingerprint(b)
  assert a.meta["group"] == b.meta["group"]


def test_an_ungrouped_example_is_its_own_group():
  examples = [example("same request"), example("same request")]
  assert all(ex.meta.get("group") is None for ex in examples)
  assert len(dedup_examples(examples)) == 1
