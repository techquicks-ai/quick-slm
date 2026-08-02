"""Generation plumbing: off-FUSE staging, resume, and the truncation guard.

No GPU. `generate_batch` is replaced with a deterministic stub, so these tests
exercise the shard bookkeeping and the Drive copy discipline, which is where a
day of teacher output is won or lost, not the teacher itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quick_slm_trainer.config import SFTConfig
from quick_slm_trainer.paths import Layout
from quick_slm_trainer.sft import generate as G
from quick_slm_trainer.sft.generate import (
  copy_up_verified,
  plan_requests,
  read_shard,
  run_generation,
)


def _stub_teacher(monkeypatch):
  """A perfect, deterministic teacher: one well-formed line per request."""

  def fake_generate(model, tok, cfg, batch, *, seed=None):
    assert seed is not None, "generation must be seeded"
    return [json.dumps({"ok": r.id}) for r in batch]

  monkeypatch.setattr(G, "generate_batch", fake_generate)


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
  return Layout(drive_root=tmp_path / "drive", local_root=tmp_path / "local")


CFG = SFTConfig(target_examples=200, gen_batch_size=8, sft_sync_every_batches=2)


def test_generation_writes_a_verified_drive_shard(monkeypatch, layout):
  _stub_teacher(monkeypatch)
  reqs = plan_requests(CFG)
  written = run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)

  assert sum(written.values()) == len(reqs)
  for category in written:
    drive, local = layout.sft_raw(category), layout.local_sft_raw(category)
    assert drive.exists() and local.exists()
    # The Drive copy is byte-identical to the local source.
    assert drive.stat().st_size == local.stat().st_size
  total = sum(1 for cat in written for _ in read_shard(layout.sft_raw(cat)))
  assert total == len(reqs)


def test_a_fresh_runtime_resumes_from_the_drive_shard(monkeypatch, layout):
  # Local disk is ephemeral on Colab; Drive survives. Wiping local and re-running
  # must regenerate nothing, because the Drive shard seeds local first.
  _stub_teacher(monkeypatch)
  reqs = plan_requests(CFG)
  run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)

  shutil.rmtree(layout.local_root) # the runtime died; only Drive remains
  again = run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)
  assert all(v == 0 for v in again.values()), "resume regenerated already-done work"


def test_a_partial_local_shard_only_generates_the_remainder(monkeypatch, layout):
  _stub_teacher(monkeypatch)
  reqs = plan_requests(CFG)
  single = [r for r in reqs if r.category == "single_stage"]

  half = single[: len(single) // 2]
  run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=half, progress=False)
  written = run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=single, progress=False)
  assert written["single_stage"] == len(single) - len(half)


def test_copy_up_refuses_a_truncated_write_and_keeps_the_old_shard(monkeypatch, tmp_path):
  local = tmp_path / "local.jsonl"
  local.write_text("a" * 1000 + "\n")
  drive = tmp_path / "drive.jsonl"
  drive.write_text("PREVIOUS GOOD SHARD\n")
  before = drive.read_bytes()

  def truncating_copyfile(src, dst):
    Path(dst).write_bytes(Path(src).read_bytes()[:500]) # FUSE truncates the copy

  monkeypatch.setattr(G.shutil, "copyfile", truncating_copyfile)
  with pytest.raises(IOError, match="truncated"):
    copy_up_verified(local, drive)

  assert drive.read_bytes() == before, "a truncated copy clobbered the good Drive shard"
  assert not (drive.parent / (drive.name + ".tmp")).exists(), "temp file left behind"


def test_copy_up_replaces_the_shard_on_a_faithful_write(tmp_path):
  local = tmp_path / "local.jsonl"
  local.write_text("x" * 4096 + "\n")
  drive = tmp_path / "sub" / "drive.jsonl" # parent does not exist yet
  copy_up_verified(local, drive)
  assert drive.read_bytes() == local.read_bytes()


# --------------------------------------------------------------------------
# Surviving a CUDA out-of-memory
# --------------------------------------------------------------------------
class _OOM(RuntimeError):
  """What older torch raises: a plain RuntimeError naming the condition."""

  def __init__(self) -> None:
    super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


def _ooms_above(width: int):
  """A teacher that OOMs on any batch wider than `width`, and records the widths."""
  seen: list[int] = []

  def fake_generate(model, tok, cfg, batch, *, seed=None):
    seen.append(len(batch))
    if len(batch) > width:
      raise _OOM()
    return [json.dumps({"ok": r.id}) for r in batch]

  return fake_generate, seen


def test_is_oom_reads_the_message_when_the_class_is_absent():
  # Newer torch raises `torch.cuda.OutOfMemoryError`; older torch raised a plain
  # RuntimeError. The message check is the fallback, and it is what lets this be
  # tested with no torch installed at all.
  assert G._is_oom(_OOM())
  assert G._is_oom(RuntimeError("CUDA out of memory"))
  assert not G._is_oom(ValueError("malformed prompt"))


def test_an_oom_batch_is_halved_until_the_pieces_fit(monkeypatch):
  gen, widths = _ooms_above(2)
  monkeypatch.setattr(G, "generate_batch", gen)
  batch = plan_requests(CFG)[:8]

  generated, dropped = G._generate_piece(None, None, CFG, batch, seed=1, category="single_stage")

  assert not dropped
  assert [r.id for r, _ in generated] == [r.id for r in batch], "shard order must track input order"
  assert max(widths) == 8 and min(widths) <= 2, widths


def test_only_the_example_that_ooms_alone_is_dropped(monkeypatch):
  # Dropping the whole batch on its first OOM, which this replaced, threw away
  # seven good examples to be rid of one.
  def gen(model, tok, cfg, batch, *, seed=None):
    if any(r.id.endswith("000003") for r in batch):
      raise _OOM()
    return [json.dumps({"ok": r.id}) for r in batch]

  monkeypatch.setattr(G, "generate_batch", gen)
  batch = plan_requests(CFG)[:8]

  generated, dropped = G._generate_piece(None, None, CFG, batch, seed=1, category="single_stage")

  assert [r.id for r in dropped] == [batch[3].id]
  assert len(generated) == 7
  assert batch[3].id not in {r.id for r, _ in generated}


def test_a_non_oom_failure_is_dropped_without_being_halved(monkeypatch):
  # Halving a batch that failed for a reason unrelated to memory just pays the
  # same failure log(n) more times.
  calls = []

  def gen(model, tok, cfg, batch, *, seed=None):
    calls.append(len(batch))
    raise ValueError("the tokenizer is misconfigured")

  monkeypatch.setattr(G, "generate_batch", gen)
  batch = plan_requests(CFG)[:8]

  generated, dropped = G._generate_piece(None, None, CFG, batch, seed=1, category="single_stage")

  assert not generated and len(dropped) == 8
  assert calls == [8], "a non-OOM failure must not be retried"


def test_a_category_whose_every_wide_batch_ooms_is_still_generated(monkeypatch, layout):
  # The regression this guards: `multi_stage` prompts do not fit at the width that
  # suited `single_stage`, so every batch OOMed and the old code wrote an empty
  # shard for the whole category after hours of runtime.
  gen, _ = _ooms_above(1)
  monkeypatch.setattr(G, "generate_batch", gen)
  reqs = [r for r in plan_requests(CFG) if r.category == "multi_stage"]

  written = run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)

  assert written["multi_stage"] == len(reqs), "a category that only fits at width 1 wrote nothing"
  assert sum(1 for _ in read_shard(layout.sft_raw("multi_stage"))) == len(reqs)


def test_a_dropped_id_is_absent_from_the_shard_so_a_resume_replans_it(monkeypatch, layout):
  def gen(model, tok, cfg, batch, *, seed=None):
    if any(r.id.endswith("000001") for r in batch):
      raise _OOM()
    return [json.dumps({"ok": r.id}) for r in batch]

  monkeypatch.setattr(G, "generate_batch", gen)
  reqs = [r for r in plan_requests(CFG) if r.category == "refusals"]
  run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)

  on_disk = {row["id"] for row in read_shard(layout.sft_raw("refusals"))}
  dropped = {r.id for r in reqs} - on_disk
  assert len(dropped) == 1

  # A healthy teacher on the next run picks the dropped id back up.
  _stub_teacher(monkeypatch)
  again = run_generation(layout=layout, cfg=CFG, model=None, tok=None, requests=reqs, progress=False)
  assert again["refusals"] == 1


def test_emptying_the_cache_is_a_no_op_without_torch():
  # It runs at every category boundary and inside the OOM path, so on a CPU box
  # with no torch it must not raise.
  G._empty_cache()


# --------------------------------------------------------------------------
# Distinct turns per seed
# --------------------------------------------------------------------------
CELL = {"category": "single_stage", "domain": "world", "subtype": "direct"}


def _pool_size() -> int:
  from quick_slm_trainer.sft.prompts import usable_seeds

  return len(usable_seeds(CELL["category"], CELL["domain"], CELL["subtype"]))


def _teacher_keyed_on(field):
  """A teacher whose reply varies only with `field(request)`."""

  def gen(model, tok, cfg, batch, *, seed=None):
    assert seed is not None, "the measurement must be seeded"
    return [
      json.dumps(
        {
          "user": f"please handle {field(r)}",
          "turns": [{"role": "assistant", "think": "t", "calls": [{"name": "answer"}]}],
        }
      )
      for r in batch
    ]

  return gen


def test_the_yield_is_the_distinct_replies_divided_by_the_pool(monkeypatch):
  # A teacher that says something different for every seed and nothing different
  # within one: exactly one distinct turn per seed, so the yield is 1.0 and a cap
  # above 1 would be buying duplicates.
  reqs = plan_requests(CFG)
  y = G.distinct_turns_per_seed(
    None, None, CFG, reqs, per_seed=3, generate=_teacher_keyed_on(lambda r: r.seed_topic), **CELL
  )

  assert y.seeds == _pool_size()
  assert y.generated == y.seeds * 3
  assert y.parsed == y.generated
  assert y.distinct == y.seeds
  assert y.per_seed_yield == 1.0


def test_a_teacher_that_repeats_itself_yields_far_less_than_one_per_seed(monkeypatch):
  reqs = plan_requests(CFG)
  y = G.distinct_turns_per_seed(
    None, None, CFG, reqs, per_seed=3, generate=_teacher_keyed_on(lambda r: "the same thing"), **CELL
  )

  assert y.distinct == 1
  assert y.per_seed_yield < 0.1
  assert "near-duplicate" in G.describe_seed_yield(y)


def test_unparseable_replies_are_counted_out_not_counted_distinct():
  def gen(model, tok, cfg, batch, *, seed=None):
    return ["the teacher wandered off into prose" for _ in batch]

  reqs = plan_requests(CFG)
  y = G.distinct_turns_per_seed(None, None, CFG, reqs, per_seed=2, generate=gen, **CELL)

  assert y.generated == y.seeds * 2
  assert y.parsed == 0 and y.distinct == 0
  assert y.per_seed_yield == 0.0
  assert G.describe_seed_yield(y) # must not divide by zero


def test_measuring_a_cell_the_plan_does_not_contain_is_an_error():
  reqs = [r for r in plan_requests(CFG) if r.category == "refusals"]
  with pytest.raises(ValueError, match="no planned request for cell"):
    G.distinct_turns_per_seed(None, None, CFG, reqs, per_seed=1, generate=_teacher_keyed_on(str), **CELL)


def test_the_measurement_varies_only_the_seed():
  # The point of the measurement is that the teacher's own sampling is its only
  # remaining freedom, so every trial must be the cell's template with one field
  # swapped. If the tools or the sub-type moved too, the count would mix the
  # teacher's variety with the planner's.
  from quick_slm_trainer.sft.prompts import usable_seeds

  seen = []

  def gen(model, tok, cfg, batch, *, seed=None):
    seen.extend(batch)
    return [json.dumps({"user": r.seed_topic, "turns": []}) for r in batch]

  reqs = plan_requests(CFG)
  cell = (CELL["category"], CELL["domain"], CELL["subtype"])
  template = next(r for r in reqs if (r.category, r.domain, r.subtype) == cell)
  G.distinct_turns_per_seed(None, None, CFG, reqs, per_seed=2, generate=gen, **CELL)

  assert {(r.category, r.domain, r.subtype) for r in seen} == {cell}
  assert all(r.tools == template.tools for r in seen)
  assert {r.seed_topic for r in seen} == {s.topic for s in usable_seeds(*cell)}
