"""Teacher generation: plan requests, batch them through the teacher, append JSONL.

The request plan is built first and in full, from a seeded RNG, before a single
token is generated. Two consequences, both wanted:

 - the corpus composition is knowable, and printable, before ten hours of H100
  time are spent finding out what it turned into
 - generation is resumable at request granularity, because a request has a
  stable id that does not depend on how far the run got

Restarting reads the ids already in the shard and skips them. It does not
truncate and it does not re-plan, so a run interrupted at 60 percent resumes at
60 percent with the same remaining 40 percent it would have generated anyway.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from ..config import SFTConfig
from ..paths import Layout
from . import conflict
from .conflict_specs import TRAIN_SPECS
from .prompts import (
  CONFLICT_SYSTEM,
  TEACHER_SYSTEM,
  build_conflict_prompt,
  build_teacher_prompt,
  choose_seed,
  usable_seeds,
)
from .specs import CATEGORIES, CategorySpec, cycle, plan_counts
from .tools import Tool, sample_tools
from .validate import NOT_JSON, extract_json, validate_record

#: Seeds the request plan and, through `batch_seed`, the teacher's sampling.
PLAN_SEED = 20240201

#: The `no_relevant_tool` seeds already name capabilities absent from the tool
#: universe, so the trap holds without withholding anything. `web_search` is the
#: exception: a general-purpose search can plausibly serve "translate this" or
#: "set a reminder", which turns the trap into a reasonable tool call and the
#: example into a rejection. It is withheld from the world domain for that
#: sub-type only. The factory domain has no such escape tool.
_NO_RELEVANT_TOOL_EXCLUSIONS: dict[str, tuple[str, ...]] = {
  "world": ("web_search",),
  "factory": (),
}


@dataclass
class Request:
  """One unit of generation. Deterministic given the plan seed.

  The `pair_*` fields are populated for the paired category only. They carry
  the *inputs* of a counterfactual branch: the state, the shared memory, and
  the shared request. They deliberately do not carry the oracle's call. The
  shard stores inputs; `conflict.py` owns the ground truth and recomputes it at
  validation, so a corrected oracle costs a revalidation pass rather than
  another eleven hours of teacher time.
  """

  id: str
  category: str
  subtype: str
  domain: str
  seed_topic: str = ""
  #: The tools `seed_topic` presupposes, pinned into `tools` at plan time. Kept
  #: on the request so the invariant "a seed is never offered a list that cannot
  #: serve it" is checkable after the fact rather than trusted.
  requires: tuple[str, ...] = ()
  tools: list[Tool] = field(default_factory=list)

  # Paired category only.
  pair_id: str | None = None
  branch: str | None = None
  spec_id: str | None = None
  state: dict | None = None
  memory: dict | None = None
  request: dict | None = None

  @property
  def is_paired(self) -> bool:
    return self.pair_id is not None

  def prompt(self) -> str:
    if self.is_paired:
      spec = conflict_spec(self.spec_id)
      pair = _pair_view(self)
      return build_conflict_prompt(
        conflict.prompt_block(pair, self.branch),
        guidance=CATEGORIES[self.category].guidance,
        subtype_guidance=CATEGORIES[self.category].subtype_guidance.get(self.subtype, ""),
      )
    return build_teacher_prompt(
      CATEGORIES[self.category],
      self.tools,
      subtype=self.subtype,
      seed_topic=self.seed_topic,
      domain=self.domain,
    )

  def system(self) -> str:
    """The system prompt the *teacher* is given. Not the one it writes about.

    This returned `DOMAIN_SYSTEM[self.domain]` and was never called; every
    request was generated under `TEACHER_SYSTEM` instead, so a paired branch
    never saw `CONFLICT_SYSTEM` and was asked for a full example when the
    conflict schema asks for one turn. Calling the method as it stood would
    have been worse: `DOMAIN_SYSTEM` is what the *student* is conditioned on
    at inference. Handing it to the teacher tells the teacher it is the
    tool-calling assistant, and it would answer each request rather than
    author an example around it. `corpus.record_to_example` is where
    `DOMAIN_SYSTEM` belongs, and where it stays.
    """
    return CONFLICT_SYSTEM if self.is_paired else TEACHER_SYSTEM


def conflict_spec(spec_id: str):
  from .conflict_specs import ALL_SPECS

  return ALL_SPECS[spec_id]


def _pair_view(req: Request) -> conflict.Pair:
  """A one-branch `Pair` good enough to render this branch's prompt.

  The other branch's state is never read by `prompt_block`, so it is filled
  with this one's. The oracle calls are absent for the same reason they are
  absent from the shard.
  """
  branch = conflict.Branch(req.branch, req.state, {})
  other = conflict.Branch("b" if req.branch == "a" else "a", req.state, {})
  a, b = (branch, other) if req.branch == "a" else (other, branch)
  spec = conflict_spec(req.spec_id)
  return conflict.Pair(
    pair_id=req.pair_id,
    spec_id=req.spec_id,
    family=spec.family,
    domain=req.domain,
    subtype=req.subtype,
    tools=req.tools,
    memory=req.memory,
    request=req.request,
    a=a,
    b=b,
  )


def plan_requests(cfg: SFTConfig, *, seed: int = PLAN_SEED) -> list[Request]:
  """The full generation plan. Pure, deterministic, cheap to inspect.

  Each category is planned by one of two helpers, and both are capacity-aware. The
  paired category goes through `_plan_pair_requests`, capped at `conflict_max_pairs`;
  every other category goes through `_plan_unpaired_requests`, capped at
  `max_examples_per_seed` per distinct seed. Both caps exist for the same reason:
  beyond what the scenario source can distinguish, the teacher only repeats itself
  and dedup deletes the surplus at full price.

  The domain is chosen first and the sub-type second, from the sub-types that
  domain supports. Choosing them independently produces combinations no teacher
  can satisfy, such as a `search_fallback` chain inside a simulation that has
  no `web_search`.

  The sub-type is indexed by `i // len(spec.domains)` rather than by `i`. Both
  cycles once read the same counter, and two round-robins on one index only
  reach the combinations their lengths' common orbit passes through. Two domains
  and four sub-types reached four of the eight pairs: the factory simulation,
  which is the demo environment, was never once shown a `direct` request, and
  `multi_stage`/`factory` was never shown a two-step or a four-step chain.
  `traps` and `refusals` escaped only because 2 and 3 happen to be coprime.
  """
  rng = random.Random(seed)
  counts = plan_counts(cfg)
  requests: list[Request] = []

  for key, n in counts.items():
    spec: CategorySpec = CATEGORIES[key]

    if spec.paired:
      requests.extend(_plan_pair_requests(key, n, rng, max_pairs=cfg.conflict_max_pairs))
      continue

    requests.extend(_plan_unpaired_requests(spec, n, rng, max_per_seed=cfg.max_examples_per_seed))
  return requests


def _plan_unpaired_requests(
  spec: CategorySpec, n: int, rng: random.Random, *, max_per_seed: int | None = None
) -> list[Request]:
  """`n` requests for one unpaired category, round-robin over (domain, subtype).

  `max_per_seed` is the unpaired twin of `_plan_pair_requests`'s `max_pairs`. It
  caps each cell at `max_per_seed * distinct-seeds`, where the distinct seeds are
  `prompts.usable_seeds` (which also narrows `optional_args`). The cell's scenarios
  come from a pool of that many hand-written topics, and past the cap the teacher
  can only rephrase a seed it already has, which dedup then deletes at teacher
  price. When the cap bites, a cell plans fewer than its round-robin share, so the
  category plans fewer than `n` examples, on purpose, exactly as `max_pairs` shrinks
  the paired plan; `describe_plan` counts what was actually planned. `None` plans
  the full share and leaves the duplicates for dedup, preserving the old contract.

  The domain is chosen first and the sub-type second, from the sub-types that domain
  supports, so no cell asks for a combination its domain cannot serve. Ids are keyed
  off `i` rather than off a running counter, so a capped cell leaves every other
  request's id where it was, and a resume still matches the shard by id.

  Each request is built and only then kept or dropped, never skipped before its
  draws. The seed and the tool sample come from an `rng` shared by the whole
  category, so returning early would advance that stream differently and hand every
  later request a different seed. Building first makes the capped plan an exact
  subsequence of the uncapped one: raising the cap later adds requests without
  redefining the ones already generated.
  """
  key = spec.key
  cell_cap: dict[tuple[str, str], int] = {}
  cell_count: dict[tuple[str, str], int] = {}
  requests: list[Request] = []

  for i in range(n):
    domain = cycle(spec.domains, i)
    subtype = cycle(spec.subtypes_for(domain), i // len(spec.domains))

    k = rng.randint(*spec.n_tools)
    chosen = choose_seed(rng, key, domain, subtype)

    exclude: Sequence[str] = ()
    if key == "traps" and subtype == "no_relevant_tool":
      exclude = _NO_RELEVANT_TOOL_EXCLUSIONS[domain]

    # The seed's own tools first, then whatever else the category pins.
    # De-duplicated because `sample_tools` pins each name it is given and
    # a repeated name would be offered to the teacher twice.
    include: tuple[str, ...] = chosen.requires
    if key == "multi_stage" and subtype == "search_fallback":
      include = (*include, "web_search")
    include = tuple(dict.fromkeys(include))

    request = Request(
      id=f"{key}-{i:06d}",
      category=key,
      subtype=subtype,
      domain=domain,
      seed_topic=chosen.topic,
      requires=chosen.requires,
      tools=sample_tools(rng, domain, k, include=include, exclude=exclude),
    )

    if max_per_seed is not None:
      cell = (domain, subtype)
      if cell not in cell_cap:
        cell_cap[cell] = max_per_seed * len(usable_seeds(key, domain, subtype))
      if cell_count.get(cell, 0) >= cell_cap[cell]:
        continue
      cell_count[cell] = cell_count.get(cell, 0) + 1

    requests.append(request)
  return requests


def _plan_pair_requests(
  key: str, n_examples: int, rng: random.Random, *, max_pairs: int | None = None
) -> list[Request]:
  """Two requests per counterfactual pair, one for each branch.

  Only `TRAIN_SPECS` is ever planned. `EVAL_SPECS` shares no tool family, no
  decisive path, and no state schema with it, and is never generated into the
  training corpus: the paper's evaluation is this same construction, and a
  sample-level split would leak because the two branches of a pair differ in
  one field.

  `max_pairs` caps the plan. The category is capacity-bound: `TRAIN_SPECS`
  distinguishes a fixed number of pairs, and planning past it generates
  duplicates that dedup deletes, at teacher price. When the cap bites, fewer
  than `n_examples` examples are planned, on purpose; `describe_plan` shows the
  real number and `sft_config` sets the cap from `recommended_conflict_pairs()`.
  """
  if n_examples % 2:
    raise ValueError(f"paired category {key!r} planned an odd count ({n_examples})")

  n_pairs = n_examples // 2
  if max_pairs is not None:
    n_pairs = min(n_pairs, max_pairs)

  requests: list[Request] = []
  for pair in conflict.plan_pairs(TRAIN_SPECS, n_pairs, rng):
    for label in ("a", "b"):
      requests.append(
        Request(
          id=f"{pair.pair_id}-{label}",
          category=key,
          subtype=pair.subtype,
          domain=pair.domain,
          tools=pair.tools,
          pair_id=pair.pair_id,
          branch=label,
          spec_id=pair.spec_id,
          state=pair.branch(label).state,
          memory=pair.memory,
          request=pair.request,
        )
      )
  return requests


def describe_plan(requests: Sequence[Request]) -> str:
  from collections import Counter

  by_cat = Counter(r.category for r in requests)
  lines = [f"{len(requests):,} requests planned", ""]
  for cat, n in by_cat.most_common():
    subs = Counter(r.subtype for r in requests if r.category == cat)
    lines.append(f" {cat:<24s} {n:>7,} ({100.0 * n / len(requests):.0f}%)")
    for sub, m in sorted(subs.items()):
      lines.append(f"   {sub:<26s} {m:>7,}")
  return "\n".join(lines)


# --------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------
def load_teacher(cfg: SFTConfig, *, device_map: str = "auto"):
  """Load the teacher and its tokenizer.

  Left padding, because a decoder-only model batched with right padding
  generates from the pad tokens and returns nonsense for every sequence in the
  batch shorter than the longest one. This is the single most common way a
  batched generation script silently produces garbage.
  """
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tok = AutoTokenizer.from_pretrained(cfg.teacher_model)
  tok.padding_side = "left"
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token

  from transformers import BitsAndBytesConfig

  # 4-bit nf4, and nothing else. The teacher is only ever loaded quantized to
  # 4-bit: bf16 (~62 GB) OOMs a 96 GB G4 once the KV cache and a batch of
  # activations are added. Google's QAT Q4 weights land near ~18 GB; compute
  # stays bf16.
  kwargs: dict = {
    "device_map": device_map,
    "quantization_config": BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
    ),
  }
  model = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, **kwargs)

  model.eval()
  return model, tok


def _render_chat(tok, system: str, user: str) -> str:
  messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
  if getattr(tok, "chat_template", None):
    try:
      return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception: # noqa: BLE001 - some templates reject a system role
      merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
      return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
  return f"{system}\n\n{user}\n"


def batch_seed(base_seed: int, requests: Sequence[Request]) -> int:
  """A sampling seed derived from a batch's contents, never from its position.

  Seeding once per process would leave a resumed run sampling differently from
  an uninterrupted one: the resume opens a fresh stream, and every batch after
  the resume point draws from a different offset than it would have. Keying the
  seed to the request ids means a batch generates the same text whenever it is
  generated, which is the reproducibility `Request` promises in its docstring.

  `hash()` is salted per process and cannot be used for this.
  """
  payload = "\x00".join((str(base_seed), *(r.id for r in requests)))
  digest = hashlib.blake2b(payload.encode(), digest_size=8).digest()
  return int.from_bytes(digest, "big") % (2**31 - 1)


def generate_batch(
  model, tok, cfg: SFTConfig, requests: Sequence[Request], *, seed: int | None = None
) -> list[str]:
  import torch

  from ..model import set_seed

  if seed is not None:
    set_seed(seed)

  prompts = [_render_chat(tok, r.system(), r.prompt()) for r in requests]
  enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

  with torch.no_grad():
    out = model.generate(
      **enc,
      max_new_tokens=cfg.max_new_tokens,
      do_sample=True,
      temperature=cfg.temperature,
      top_p=cfg.top_p,
      repetition_penalty=cfg.repetition_penalty,
      pad_token_id=tok.pad_token_id,
    )
  # Left padding makes every prompt end at the same column, so the completion
  # of every row starts at exactly `enc.input_ids.shape[1]`.
  start = enc["input_ids"].shape[1]
  return [tok.decode(row[start:], skip_special_tokens=True) for row in out]


# --------------------------------------------------------------------------
# Resumable JSONL shards
# --------------------------------------------------------------------------
def completed_ids(path: Path) -> set[str]:
  if not path.exists():
    return set()
  done = set()
  with open(path) as fh:
    for line in fh:
      line = line.strip()
      if not line:
        continue
      try:
        done.add(json.loads(line)["id"])
      except (json.JSONDecodeError, KeyError):
        # A line torn in half by a dying runtime. It will be regenerated.
        continue
  return done


def read_shard(path: Path) -> Iterator[dict]:
  if not path.exists():
    return
  with open(path) as fh:
    for line in fh:
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        continue


def _batched(items: Sequence[Request], n: int) -> Iterable[Sequence[Request]]:
  for i in range(0, len(items), n):
    yield items[i : i + n]


# --------------------------------------------------------------------------
# Off-FUSE staging for the raw shard
# --------------------------------------------------------------------------
# The raw shard is a day of teacher time, and it was the one artifact written
# straight through the Drive FUSE mount, which truncates large writes under load.
# Generation now appends to local disk and copies each category up to Drive with a
# byte-size check, the same discipline `pretraining` and `checkpoint` already use.
def copy_up_verified(local_path: Path, drive_path: Path) -> None:
  """Copy `local_path` to `drive_path` via a temp file, refusing a short write.

  FUSE can return success on a copy it silently truncated. Writing to a sibling
  temp and comparing byte sizes before the rename means a truncated copy never
  replaces a good shard: the temp is discarded and the previous Drive shard,
  which resume can still read, stays intact.
  """
  drive_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = drive_path.parent / (drive_path.name + ".tmp")
  shutil.copyfile(local_path, tmp)
  copied, expected = tmp.stat().st_size, local_path.stat().st_size
  if copied != expected:
    tmp.unlink() # after reading the sizes, so the message can report them
    raise IOError(
      f"Drive copy of {drive_path.name} truncated ({copied} != {expected} bytes); "
      "Drive shard left untouched"
    )
  tmp.replace(drive_path)


def _seed_local_from_drive(local_path: Path, drive_path: Path) -> None:
  """Prime local staging with prior progress on a fresh Colab runtime.

  Local disk is ephemeral; Drive survives a runtime death. When local is absent
  but Drive holds an earlier run's shard, that shard becomes the working copy so
  `completed_ids` counts the ids already generated and the run resumes them. Once
  local exists it is authoritative, because it only ever grows and is copied up.
  """
  if drive_path.exists() and not local_path.exists():
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(drive_path, local_path)


def _empty_cache() -> None:
  """Hand the caching allocator's free blocks back to CUDA. A no-op off-GPU.

  PyTorch keeps freed GPU blocks in a per-process cache and reuses them, which is
  what makes generation fast. It also means the memory a category reserved at its
  peak stays reserved after it finishes, in blocks shaped for that category's
  batches. The next category, whose prompts and completions are larger, then meets a
  device that reads as almost entirely allocated and OOMs beside tens of gigabytes of
  idle cache: `single_stage` fills the pool with short single-turn work, and every
  wide `multi_stage` batch after it fails with ~200 MiB free. Emptying the cache at
  the boundary returns that memory so the next category can reshape it.
  """
  try:
    import torch
  except ImportError:
    return
  if torch.cuda.is_available():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
  """Whether `exc` is a CUDA out-of-memory error, by class or by message.

  Newer torch raises `torch.cuda.OutOfMemoryError`; older torch raised a plain
  `RuntimeError` whose message carries "out of memory". The message check is the
  fallback, and it also lets this be reasoned about with torch absent.
  """
  try:
    import torch

    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
  except ImportError:
    oom_cls = ()
  if oom_cls and isinstance(exc, oom_cls):
    return True
  return "out of memory" in str(exc).lower()


def _generate_piece(
  model, tok, cfg: SFTConfig, batch: Sequence[Request], *, seed: int, category: str
) -> tuple[list[tuple[Request, str]], list[Request]]:
  """Generate one batch, halving it on CUDA OOM until each piece fits.

  Dropping a whole batch on its first OOM, which is what this replaced, wrote nothing
  at all for a category whose every batch OOMs: a `multi_stage` chain prompt and its
  2,048-token completion do not fit at the width that suited the `single_stage` turns
  generated before it. Here an OOM empties the cache and halves the batch, retrying
  down to a single example, so the category is generated at whatever width fits rather
  than skipped. Only an example that OOMs alone, or a non-OOM failure, is dropped, and
  its id is named. A dropped id is simply absent from the shard, so the next resume
  plans it again.

  Returns `(generated, dropped)`: `generated` is `(request, text)` in input order,
  `dropped` is the requests no width could generate.
  """
  generated: list[tuple[Request, str]] = []
  dropped: list[Request] = []
  # A LIFO stack of sub-batches still to do. On a split the back half is pushed first
  # so the front half is popped and generated first, and shard order tracks the input.
  stack: list[list[Request]] = [list(batch)]
  while stack:
    piece = stack.pop()
    try:
      texts = generate_batch(model, tok, cfg, piece, seed=batch_seed(seed, piece))
    except Exception as e: # noqa: BLE001
      # Reclaim whatever the failed attempt reserved before retrying or moving on;
      # without it the free memory only shrinks from one failed batch to the next.
      _empty_cache()
      if _is_oom(e) and len(piece) > 1:
        mid = len(piece) // 2
        stack.append(piece[mid:])
        stack.append(piece[:mid])
        continue
      reason = "out of memory, even alone" if _is_oom(e) else f"{type(e).__name__}: {e}"
      print(f" [{category}] dropped {len(piece)} example(s) ({reason}); first id {piece[0].id}")
      dropped.extend(piece)
      continue
    generated.extend(zip(piece, texts))
  return generated, dropped


def run_generation(
  *,
  layout: Layout,
  cfg: SFTConfig,
  model,
  tok,
  requests: Sequence[Request],
  progress: bool = True,
  seed: int = PLAN_SEED,
) -> dict[str, int]:
  """Generate every outstanding request, appending to one JSONL shard per category.

  Returns the number of new records written per category. Raw teacher text is
  stored, not parsed output: parsing is cheap and re-runnable, generation is
  not, and a filter tightened next week should not need eleven more H100 hours.

  Two memory disciplines keep a long multi-category run alive on one card. The
  allocator cache is emptied at each category boundary, because a category holds its
  peak reservation after it finishes and the next category's larger batches would
  otherwise face a full device. And a batch that OOMs is halved and retried rather
  than dropped, so a category whose full-width batch does not fit is still generated
  at whatever width does, instead of writing nothing. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before loading the teacher to
  reduce the fragmentation that makes both necessary.
  """
  layout.mkdirs_sft()
  written: dict[str, int] = {}

  by_category: dict[str, list[Request]] = {}
  for r in requests:
    by_category.setdefault(r.category, []).append(r)

  for category, reqs in by_category.items():
    drive_path = layout.sft_raw(category)
    local_path = layout.local_sft_raw(category)
    _seed_local_from_drive(local_path, drive_path)

    done = completed_ids(local_path)
    todo = [r for r in reqs if r.id not in done]
    # Batched left-padding costs the longest prompt in the batch, times the
    # batch width, for every row in it. Sorting groups prompts of a similar
    # length together, so a 3,000-token multi-stage prompt no longer sets the
    # price of the seven short ones it happened to be batched with. Shard
    # order changes; `completed_ids` keys on the request id, so resume does not.
    todo.sort(key=lambda r: len(r.prompt()))
    print(f"[{category}] {len(done):,} done, {len(todo):,} to generate")
    if not todo:
      written[category] = 0
      continue

    bar = None
    if progress:
      from tqdm.auto import tqdm

      bar = tqdm(total=len(todo), desc=category, unit="ex", smoothing=0.05)

    # Hand the previous category's reserved-but-idle allocator blocks back to CUDA
    # before this one generates. Without it, `single_stage` fills the pool with
    # short single-turn work and the wider `multi_stage` batches after it OOM with
    # a nearly full device beside tens of gigabytes of idle cache.
    _empty_cache()

    count = 0
    n_dropped = 0
    batches_since_sync = 0
    with open(local_path, "a") as fh:
      for batch in _batched(todo, cfg.gen_batch_size):
        # A batch that OOMs is halved and retried, not dropped whole: for a
        # category whose every wide batch OOMs, dropping wrote nothing at all.
        generated, dropped = _generate_piece(
          model, tok, cfg, batch, seed=seed, category=category
        )
        n_dropped += len(dropped)

        for req, text in generated:
          fh.write(json.dumps(shard_record(req, text)) + "\n")
          count += 1
        fh.flush()
        batches_since_sync += 1

        # Push local progress up to Drive every so often, so a runtime
        # death loses at most this many batches rather than the category.
        if batches_since_sync >= cfg.sft_sync_every_batches:
          fh.flush()
          copy_up_verified(local_path, drive_path)
          batches_since_sync = 0
        if bar is not None:
          bar.update(len(batch))

    if bar is not None:
      bar.close()
    # Final verified copy-up: this is the write that must not truncate.
    copy_up_verified(local_path, drive_path)
    written[category] = count
    # The per-batch drop lines scroll past during a long run; this is the one
    # line that gets read. Without the tally a category that dropped a tenth of
    # its plan looks identical to one that generated cleanly.
    note = f", dropped {n_dropped:,} of {len(todo):,}" if n_dropped else ""
    print(f"[{category}] wrote {count:,} new records{note} -> {drive_path} (verified)")

  return written


def shard_record(req: Request, raw: str) -> dict:
  """One JSONL line. Inputs plus the teacher's raw reply, never a verdict."""
  record = {
    "id": req.id,
    "category": req.category,
    "subtype": req.subtype,
    "domain": req.domain,
    "tools": req.tools,
    "raw": raw,
  }
  if req.is_paired:
    record.update(
      pair_id=req.pair_id,
      branch=req.branch,
      spec_id=req.spec_id,
      state=req.state,
      memory=req.memory,
      request=req.request,
    )
  return record


def request_to_dict(r: Request) -> dict:
  return asdict(r)


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
@dataclass
class Preflight:
  """What one sampled request says about the teacher, before eighty thousand of them."""

  id: str
  category: str
  subtype: str
  #: Everything the teacher emitted before its first `{`. `extract_json` skips
  #: it, which is how "Here is the example:" is recovered rather than discarded,
  #: and also how a reasoning trace containing a brace would be parsed in place
  #: of the example.
  preamble: str
  reject: str | None
  raw: str

  @property
  def parsed(self) -> bool:
    return self.reject != NOT_JSON

  @property
  def usable(self) -> bool:
    return self.reject is None


def _spread(requests: Sequence[Request], n: int) -> list[Request]:
  """`n` requests, round-robin across categories, so no category goes unsampled."""
  by_category: dict[str, list[Request]] = {}
  for r in requests:
    by_category.setdefault(r.category, []).append(r)

  out: list[Request] = []
  for i in range(max((len(v) for v in by_category.values()), default=0)):
    for reqs in by_category.values():
      if len(out) >= n:
        return out
      if i < len(reqs):
        out.append(reqs[i])
  return out


def check_one(req: Request, raw: str) -> Preflight:
  brace = raw.find("{")
  spec = CATEGORIES[req.category]
  if spec.paired:
    reason, _ = conflict.validate_branch(raw, req.tools)
  else:
    record = extract_json(raw)
    reason = NOT_JSON if record is None else validate_record(record, spec, req.tools)
  return Preflight(
    id=req.id,
    category=req.category,
    subtype=req.subtype,
    preamble=raw[:brace] if brace > 0 else "",
    reject=reason,
    raw=raw,
  )


def preflight(
  model,
  tok,
  cfg: SFTConfig,
  requests: Sequence[Request],
  *,
  n: int = 20,
  seed: int = PLAN_SEED,
  generate: Callable[..., list[str]] | None = None,
) -> list[Preflight]:
  """Generate a handful of examples and report what the teacher actually did.

  `extract_json` takes the first balanced `{...}` span of the decoded reply, and
  `generate_batch` decodes with `skip_special_tokens=True`, which strips a
  thinking delimiter without stripping what it delimited. Gemma 4 leaves
  thinking off unless `<|think|>` appears in the system prompt, and no prompt
  here contains it, but the chat template is the teacher's and not ours. If it
  ever opens a reasoning trace, a brace inside that trace is what gets parsed,
  and the failure is silent: a well-formed object that is not the example.

  Twenty samples settle it in a minute. Eighty thousand cost a day.

  `generate` is a seam for tests; production passes `generate_batch`.
  """
  run = generate or generate_batch
  results: list[Preflight] = []
  for batch in _batched(_spread(requests, n), cfg.gen_batch_size):
    texts = run(model, tok, cfg, batch, seed=batch_seed(seed, batch))
    results.extend(check_one(req, raw) for req, raw in zip(batch, texts))
  return results


def describe_preflight(results: Sequence[Preflight]) -> str:
  from collections import Counter

  n = len(results)
  lines = [f"{n} sampled {sum(r.parsed for r in results)} parsed {sum(r.usable for r in results)} usable"]

  preambles = [r for r in results if r.preamble.strip()]
  if preambles:
    lines += [
      "",
      f" {len(preambles)}/{n} replies emit text before their first '{{'. `extract_json` takes the",
      " first balanced brace span, so a brace anywhere in that text is parsed in place of the",
      " example. Read these before committing to a full run:",
    ]
    lines += [f"  [{r.id}] {r.preamble.strip()[:96]!r}" for r in preambles[:3]]

  rejects = Counter(r.reject for r in results if r.reject)
  if rejects:
    lines.append("")
    lines += [f" {reason:<28s} {m:>4}" for reason, m in rejects.most_common()]
  return "\n".join(lines)


# --------------------------------------------------------------------------
# Distinct turns per seed: the empirical form of the per-seed cap
# --------------------------------------------------------------------------
def _turn_signature(record: dict) -> str:
  """The dedup axes of one parsed teacher reply: its user turn and its call names.

  `dedup.example_fingerprint` compares the request and the call signatures and
  excludes `think`, because two replies that ask the same thing and answer with
  the same calls are duplicates however differently the reasoning is phrased. This
  is the same axis, kept light enough to run inside the pre-flight, and it is what
  `distinct_turns_per_seed` counts distinct values of. It matches the fingerprint
  notebook `04a_sft_seed_audit` section 6 forms by hand.
  """
  names = [
    str(call.get("name", ""))
    for turn in record.get("turns", [])
    if isinstance(turn, dict)
    for call in (turn.get("calls") or [])
  ]
  return f"{record.get('user', '')} || {' '.join(names)}"


@dataclass
class SeedYield:
  """How much distinct signal one (domain, subtype) cell yields per seed.

  The empirical quantity `SFTConfig.max_examples_per_seed` approximates, measured
  the way `conflict.distinct_capacity` measures the paired ceiling, except that the
  unpaired variety is the teacher's and not the spec table's, so it cannot be read
  off a fingerprint on CPU and has to be generated. `per_seed_yield` is the number
  a cap should sit near rather than below: set the cap under it and the plan drops
  distinct examples the teacher would have produced; set it far above and the plan
  buys duplicates dedup deletes at teacher price.
  """

  category: str
  domain: str
  subtype: str
  #: Distinct seeds in the cell's pool (`usable_seeds`, `optional_args` narrowed).
  seeds: int
  #: Completions generated per seed.
  per_seed: int
  generated: int
  parsed: int
  #: Distinct turn-signatures across every parsed completion of the cell.
  distinct: int

  @property
  def per_seed_yield(self) -> float:
    return self.distinct / self.seeds if self.seeds else 0.0


def distinct_turns_per_seed(
  model,
  tok,
  cfg: SFTConfig,
  requests: Sequence[Request],
  *,
  category: str,
  domain: str,
  subtype: str,
  per_seed: int = 20,
  seed: int = PLAN_SEED,
  generate: Callable[..., list[str]] | None = None,
) -> SeedYield:
  """Measure the distinct turn-structures the teacher yields per seed, for one cell.

  `max_examples_per_seed` caps a cell at `cap * distinct-seeds` on the theory that,
  past the cap, the teacher can only rephrase a seed it has already been given. This
  measures the theory rather than assuming it. Each distinct seed of the cell is
  replayed `per_seed` times through the teacher, the turns of every parsed reply are
  fingerprinted the way `dedup` does (the user request and the call signatures,
  `think` excluded), and the distinct fingerprints are counted. Dividing by the seed
  count gives the yield per seed, which is where `max_examples_per_seed` should sit.

  Notebook `04a_sft_seed_audit` section 6 runs this by hand as a preview, gated
  behind an explicit opt-in because it spends teacher time. Folded here it is a
  single call the pre-flight can make against a chosen cell, usually the one dedup
  hits hardest (`single_stage`/`world`/`direct`), for a few hundred generations
  rather than the eighty thousand of a full run.

  `generate` is the same test seam `preflight` uses; production passes
  `generate_batch`.
  """
  run = generate or generate_batch

  pool = usable_seeds(category, domain, subtype)
  template = next(
    (
      r
      for r in requests
      if not r.is_paired and (r.category, r.domain, r.subtype) == (category, domain, subtype)
    ),
    None,
  )
  if template is None:
    raise ValueError(
      f"no planned request for cell {category}/{domain}/{subtype}; "
      "distinct_turns_per_seed measures a cell the plan contains"
    )

  # One request per (seed, completion), a copy of the cell's template with only the
  # seed swapped. The teacher's sole remaining freedom is its own sampling, which is
  # exactly the variety this measures.
  trials: list[Request] = []
  for s in pool:
    for _ in range(per_seed):
      trials.append(
        replace(template, id=f"seed-yield-{len(trials):06d}", seed_topic=s.topic, requires=s.requires)
      )

  signatures: set[str] = set()
  parsed = 0
  for batch in _batched(trials, cfg.gen_batch_size):
    texts = run(model, tok, cfg, batch, seed=batch_seed(seed, batch))
    for raw in texts:
      record = extract_json(raw)
      if record is None:
        continue
      parsed += 1
      signatures.add(_turn_signature(record))

  return SeedYield(
    category=category,
    domain=domain,
    subtype=subtype,
    seeds=len(pool),
    per_seed=per_seed,
    generated=len(trials),
    parsed=parsed,
    distinct=len(signatures),
  )


def describe_seed_yield(y: SeedYield) -> str:
  dup = 100 * (y.parsed - y.distinct) / y.parsed if y.parsed else 0.0
  suggested = max(1, round(y.per_seed_yield))
  return "\n".join(
    [
      f"{y.category}/{y.domain}/{y.subtype}: {y.seeds} seeds x {y.per_seed} completions",
      f" {y.generated} generated {y.parsed} parsed {y.distinct} distinct "
      f"({dup:.0f}% near-duplicate)",
      f" {y.per_seed_yield:.1f} distinct per seed -> set max_examples_per_seed near "
      f"{suggested}, not below it",
    ]
  )
