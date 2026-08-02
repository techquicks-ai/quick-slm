"""Typed configuration, serialisable to JSON.

The paper promises "exact architecture settings", "exact filters per source",
"exact checkpoint cadence". Those promises are only worth something if the
values are readable outside a notebook. Every config here round-trips through
JSON and is written into each checkpoint by `checkpoint.save`.

The presets record what the run *actually did*, which in three places is not
what `training/README.md` says it did:

 - effective batch is 260 sequences (20 x 13), not 256
 - attention is `sdpa`, not `flash_attention_2`
 - gradient checkpointing was off, not on

Those are the notebook's values. The documentation drifted; the code did not.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


def _from_mapping(cls, data: Mapping[str, Any]):
  """Build a dataclass from a mapping, ignoring keys it does not declare.

  Tolerating unknown keys means an old checkpoint written by an older version
  of this package still loads after a field is added or removed.
  """
  known = {f.name for f in fields(cls)}
  return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelConfig:
  """Architecture. Deep-and-thin per MobileLLM (Liu et al., 2024)."""

  hidden_size: int = 576
  intermediate_size: int = 1536
  num_hidden_layers: int = 24
  num_attention_heads: int = 9
  num_key_value_heads: int = 3
  head_dim: int = 64
  max_position_embeddings: int = 4096
  rope_theta: float = 10_000.0
  tie_word_embeddings: bool = True
  rms_norm_eps: float = 1e-5
  hidden_act: str = "silu"
  attention_dropout: float = 0.0
  attn_implementation: str = "sdpa"

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "ModelConfig":
    return _from_mapping(cls, d)


@dataclass
class DataConfig:
  """Pretraining corpus recipe: budgets, filters, packing."""

  budgets: dict[str, int] = field(
    default_factory=lambda: {
      "fineweb_edu": 7_750_000_000, # int_score >= 4
      "finemath": 1_000_000_000, # int_score == 5 within finemath-4plus
      "tool_calling": 250_000_000, # xLAM (3 epochs) then Glaive
      "starcoder": 1_000_000_000, # Python subset
    }
  )
  ctx: int = 4096
  tokenizer_repo: str = "NousResearch/Llama-2-7b-hf"

  # Filter thresholds, named rather than buried in a lambda, because the
  # paper's data section quotes them.
  fineweb_min_int_score: int = 4
  finemath_min_int_score: int = 5
  xlam_epochs: int = 3
  starcoder_langs: tuple[str, ...] = ("python",)

  # Streaming/packing.
  batch_docs: int = 256
  flush_every: int = 50
  shuffle_buffer: int = 1_000
  seed: int = 42

  # 'strict' refuses to combine when a source is short of budget.
  # 'cap_proportional' scales every source down to the most-incomplete one,
  # preserving the 77.5/10/2.5/10 ratio. 'use_all' skews the ratios.
  combine_mode: str = "strict"

  @property
  def target_total_tokens(self) -> int:
    return sum(self.budgets.values())

  def shares(self) -> dict[str, float]:
    total = self.target_total_tokens
    return {k: v / total for k, v in self.budgets.items()}

  def to_dict(self) -> dict:
    d = asdict(self)
    d["starcoder_langs"] = list(self.starcoder_langs)
    return d

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "DataConfig":
    d = dict(d)
    if "starcoder_langs" in d:
      d["starcoder_langs"] = tuple(d["starcoder_langs"])
    return _from_mapping(cls, d)


@dataclass
class OptimConfig:
  lr_peak: float = 3e-4
  lr_min: float = 3e-5
  warmup_steps: int = 1_000
  weight_decay: float = 0.1
  grad_clip: float = 1.0
  betas: tuple[float, float] = (0.9, 0.95)
  eps: float = 1e-8
  fused: bool = True

  def to_dict(self) -> dict:
    d = asdict(self)
    d["betas"] = list(self.betas)
    return d

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "OptimConfig":
    d = dict(d)
    if "betas" in d:
      d["betas"] = tuple(d["betas"])
    return _from_mapping(cls, d)


@dataclass
class RunConfig:
  """Batch shape, run length, cadence, and hardware knobs."""

  micro_batch: int = 20
  grad_accum: int = 13
  target_tokens: int = 10_000_000_000

  save_every_steps: int = 500
  log_every_steps: int = 10
  eval_every_steps: int = 500
  eval_batches: int = 50
  keep_last_n_ckpts: int = 3
  decile_milestones: bool = True

  num_workers: int = 4
  prefetch_factor: int = 4
  use_grad_checkpt: bool = False
  use_torch_compile: bool = False
  use_fp8: bool = False

  seed: int = 1337
  dtype: str = "bfloat16"
  device: str = "cuda"

  @property
  def effective_batch(self) -> int:
    return self.micro_batch * self.grad_accum

  def tokens_per_step(self, ctx: int) -> int:
    return self.effective_batch * ctx

  def total_steps(self, ctx: int) -> int:
    return self.target_tokens // self.tokens_per_step(ctx)

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "RunConfig":
    return _from_mapping(cls, d)


@dataclass
class SFTConfig:
  """Corpus design and generation settings for the supervised stage."""

  # Category shares by example count. The multi-stage and conflict categories
  # run heavier by token count, which is where the hard learning is.
  category_shares: dict[str, float] = field(
    default_factory=lambda: {
      "single_stage": 0.40,
      "multi_stage": 0.25,
      "state_memory_conflict": 0.20,
      "traps": 0.10,
      "refusals": 0.05,
    }
  )
  target_examples: int = 80_000 # raw, before validation and dedup

  teacher_model: str = "google/gemma-4-31B-it-qat-q4_0-unquantized"
  # The teacher is loaded in 4-bit (nf4) only. bf16 and 8-bit are not options:
  # the full 31B in bf16 is ~62 GB before KV cache and activations, and a batch
  # of 32 at 2048 tokens on top of that OOMs a 96 GB G4. Google's QAT Q4 weights
  # in 4-bit land near ~18 GB and leave the rest of the card for KV headroom;
  # QAT means the Q4 weights were tuned for quantization, so the signal loss is
  # small. The `-unquantized` repo ships QAT-tuned weights that transformers
  # loads and bitsandbytes re-quantizes to 4-bit on the way in.
  temperature: float = 0.7
  top_p: float = 0.95
  # 1.0, not 1.05. The teacher's output is JSON, whose whole surface is repeated
  # punctuation, and a multi-turn example repeats "role", "assistant", "think",
  # and "calls" once per turn. A penalty on repetition is a penalty on the
  # format that every downstream filter requires.
  repetition_penalty: float = 1.0
  max_new_tokens: int = 2048
  # Raised from 8. `generate.run_generation` sorts by prompt length first, so a
  # wider batch pads less rather than more. Tune it against the pre-flight run:
  # this is a starting point on 96 GB, not a measurement.
  gen_batch_size: int = 32
  # How often generation copies the local raw shard up to Drive. Every write to
  # Drive risks a FUSE truncation, so it is verified and not done per batch; a
  # runtime death loses at most this many batches of teacher output.
  sft_sync_every_batches: int = 20

  # Cap on planned counterfactual pairs. The paired category is capacity-bound:
  # its scenarios come from a fixed spec table, and beyond the table's distinct
  # capacity every extra pair is a duplicate that dedup deletes, at full teacher
  # price. `None` plans the raw share and lets dedup sort it out; an int caps the
  # plan. `sft_config` sets this from `conflict.recommended_conflict_pairs()`.
  conflict_max_pairs: int | None = None

  # The unpaired twin of `conflict_max_pairs`. The unpaired categories draw their
  # scenarios from the hand-written seed pools in `sft/prompts.py`, 5 to 29 topics
  # per (domain, subtype) cell, while the raw share asks for thousands of examples
  # per cell. Past the pool the teacher can only rephrase a seed it already has,
  # and dedup deletes the surplus at full teacher price. `None` plans the raw share
  # and lets dedup sort it out; an int caps each cell at `cap * distinct-seeds`.
  # `sft_config` sets this from `prompts.recommended_examples_per_seed()`.
  max_examples_per_seed: int | None = None

  # Filters.
  min_example_tokens: int = 100
  max_example_tokens: int = 3_000
  dedup_jaccard: float = 0.85
  minhash_perms: int = 128

  val_fraction: float = 0.05
  ctx: int = 4096

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "SFTConfig":
    return _from_mapping(cls, d)


@dataclass
class Config:
  """The whole run, in one serialisable object."""

  model: ModelConfig = field(default_factory=ModelConfig)
  data: DataConfig = field(default_factory=DataConfig)
  optim: OptimConfig = field(default_factory=OptimConfig)
  run: RunConfig = field(default_factory=RunConfig)
  sft: SFTConfig = field(default_factory=SFTConfig)
  notes: str = ""

  def to_dict(self) -> dict:
    return {
      "model": self.model.to_dict(),
      "data": self.data.to_dict(),
      "optim": self.optim.to_dict(),
      "run": self.run.to_dict(),
      "sft": self.sft.to_dict(),
      "notes": self.notes,
    }

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "Config":
    return cls(
      model=ModelConfig.from_dict(d.get("model", {})),
      data=DataConfig.from_dict(d.get("data", {})),
      optim=OptimConfig.from_dict(d.get("optim", {})),
      run=RunConfig.from_dict(d.get("run", {})),
      sft=SFTConfig.from_dict(d.get("sft", {})),
      notes=d.get("notes", ""),
    )

  def save(self, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(self.to_dict(), indent=2))
    return path

  @classmethod
  def load(cls, path: str | Path) -> "Config":
    return cls.from_dict(json.loads(Path(path).read_text()))


# ==========================================================================
# Version presets
#
# A `*_v1` preset describes a run that has already happened. Its checkpoints
# exist and the paper reports numbers measured from them, so what these
# functions resolve to is part of the record rather than a current opinion.
#
# **They are frozen.** So is everything they read: the field defaults above, the
# seed pools, the category specs, and the `recommended_*` functions. A later
# version that wants different behaviour adds a preset beside these rather than
# editing them, which costs nothing because a frozen version cannot use what it
# never referenced.
#
# `framework/frozen_config.json` records what these produce and
# `framework/tests/test_frozen.py` fails on any drift, naming the knob that
# moved. That test is the reason a retune for cannot quietly rewrite its
# corpus. See `training/README.md` for the full rule.
# ==========================================================================
def pretrain_config() -> Config:
  """The run as executed, on one RTX PRO 6000 Blackwell.

  `LR_PEAK = 3e-4` is preserved here because it is what ran, not because it
  was right. `docs/important_notes.md` argues 6e-4 to 1e-3 is the
  consensus band at 103M and estimates the choice cost ~5% final capability.
  Changing this constant would falsify the record; `pretrain_v2()` is where
  the corrected recipe lives.
  """
  return Config(notes="as executed: 103M, 10B tokens, RTX PRO 6000 Blackwell")



def sft_config(base: Config | None = None) -> Config:
  """SFT over the base checkpoint.

  Architecture and tokenizer are inherited from the base checkpoint, so only
  the optimisation and run blocks differ. LR is one order of magnitude below
  the pretraining peak. Three epochs over the ~80M-token packed corpus.

  `base` is copied, not adopted. A caller holding a `pretrain_config()` config and
  passing it here would otherwise find its learning rate silently rewritten.
  """
  cfg = copy.deepcopy(base) if base is not None else pretrain_config()
  cfg.notes = "SFT over the 103M base checkpoint"

  # Cap the counterfactual category at what its spec table can distinguish, plus
  # a margin for teacher rejects. Imported here rather than at module load: the
  # capacity is computed from the spec table, and `config` sits below `sft` in
  # the import graph. The value is memoised, so this costs seconds once.
  from .sft.conflict import recommended_conflict_pairs
  from .sft.prompts import recommended_examples_per_seed

  cfg.sft.conflict_max_pairs = recommended_conflict_pairs()
  # And cap the unpaired categories at what their seed pools can realise, the same
  # asymmetry the paired cap fixes: without this, every cell plans thousands of
  # examples from a pool of tens and the teacher repeats itself up to the dedup pass.
  cfg.sft.max_examples_per_seed = recommended_examples_per_seed()

  cfg.optim.lr_peak = 5e-5
  cfg.optim.lr_min = 5e-6
  cfg.optim.warmup_steps = 200
  cfg.run.micro_batch = 4
  # 12 sequences x 4096 ctx = 49,152 tokens per optimizer step.
  #
  # This was 16 accumulation steps, sized for the ~80M-token corpus that
  # `SFT_README.md` planned and that its "~920 steps total" follows from. The
  # realised corpus is 14.7M tokens: the per-seed cap cut the plan from 80,000
  # requests to 36,640, and dedup then removed 43.9% of what passed
  # validation. At 262,144 tokens per
  # step that leaves 168 optimizer steps for three epochs, which is fewer than
  # `warmup_steps` above. The fine-tune would have spent its entire length in
  # linear warmup, reached 84% of the peak rate on its final step, and never
  # decayed at all, which is the worst point at which to stop.
  #
  # Three restores the intended schedule rather than the intended batch: 899
  # steps over three epochs, warmup completing at step 199, and a full cosine
  # decay to `lr_min`. Peak memory is set by `micro_batch` and is unchanged.
  cfg.run.grad_accum = 3
  cfg.run.use_grad_checkpt = True
  cfg.run.save_every_steps = 100
  cfg.run.eval_every_steps = 50
  cfg.run.keep_last_n_ckpts = 3
  cfg.run.decile_milestones = False
  # target_tokens is set by the SFT notebook from the packed corpus size
  # times the epoch count; there is no fixed 10B budget here.
  return cfg



