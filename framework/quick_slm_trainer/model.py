"""Model construction, and the device / dtype / compiler preparation step.

Two entry points, one per stage. `build_model` makes a randomly initialised
network from a `Config`, which is what pretraining starts from.
`load_pretrained` reads a HuggingFace directory, which is what SFT starts from.
Both hand off to `prepare`, so a checkpoint reaches the training loop through
exactly one code path regardless of where it came from.
"""

from __future__ import annotations

from pathlib import Path


def unwrap(model):
  """The underlying module beneath `torch.compile`.

  `torch.compile` returns an `OptimizedModule` whose `state_dict` keys are all
  prefixed with `_orig_mod.`. Saving those keys produces a checkpoint that
  `from_pretrained` cannot read, and the failure surfaces hours later at load
  time. Every save and load goes through here.
  """
  return getattr(model, "_orig_mod", model)


def param_count(model) -> int:
  return sum(p.numel() for p in unwrap(model).parameters())


def build_llama_config(cfg, *, vocab_size: int, tok=None):
  """A `LlamaConfig` from this package's `ModelConfig`.

  `use_cache=False` because a KV cache during training is wasted memory, and
  because gradient checkpointing warns and disables it anyway.
  """
  from transformers import LlamaConfig

  m = cfg.model
  kwargs = dict(
    vocab_size=vocab_size,
    hidden_size=m.hidden_size,
    intermediate_size=m.intermediate_size,
    num_hidden_layers=m.num_hidden_layers,
    num_attention_heads=m.num_attention_heads,
    num_key_value_heads=m.num_key_value_heads,
    head_dim=m.head_dim,
    max_position_embeddings=m.max_position_embeddings,
    rope_theta=m.rope_theta,
    tie_word_embeddings=m.tie_word_embeddings,
    rms_norm_eps=m.rms_norm_eps,
    hidden_act=m.hidden_act,
    attention_dropout=m.attention_dropout,
    attn_implementation=m.attn_implementation,
    use_cache=False,
  )
  if tok is not None:
    kwargs.update(
      pad_token_id=tok.pad_token_id,
      bos_token_id=tok.bos_token_id,
      eos_token_id=tok.eos_token_id,
    )
  return LlamaConfig(**kwargs)


def build_model(cfg, *, vocab_size: int, tok=None):
  """A fresh, randomly initialised model. The pretraining entry point."""
  from transformers import LlamaForCausalLM

  return LlamaForCausalLM(build_llama_config(cfg, vocab_size=vocab_size, tok=tok))


def _from_pretrained(hf_dir: str | Path, *, dtype, attn_implementation: str):
  """`from_pretrained`, tolerating the `torch_dtype` to `dtype` rename.

  transformers renamed the keyword and deprecated the old spelling. Colab
  pins whatever version it pins, and a run should not die on a keyword.
  """
  from transformers import AutoModelForCausalLM

  common = dict(attn_implementation=attn_implementation)
  try:
    return AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=dtype, **common)
  except TypeError:
    return AutoModelForCausalLM.from_pretrained(str(hf_dir), torch_dtype=dtype, **common)


def load_pretrained(hf_dir: str | Path, *, dtype, attn_implementation: str = "sdpa", tok=None):
  """Load a saved checkpoint. The SFT entry point.

  When `tok` is supplied its vocabulary is checked against the checkpoint's
  embedding rows. SFT introduces `<think>` and the ChatML pair, all of which
  were reserved before pretraining; a mismatch here means the tokenizer being
  used is not the tokenizer the base model was trained with, and every token
  id in the corpus would be off.
  """
  model = _from_pretrained(hf_dir, dtype=dtype, attn_implementation=attn_implementation)
  model.config.use_cache = False
  if tok is not None:
    embed_rows = model.get_input_embeddings().weight.shape[0]
    if len(tok) != embed_rows:
      raise ValueError(
        f"tokenizer has {len(tok):,} tokens but the checkpoint at {hf_dir} has "
        f"{embed_rows:,} embedding rows. Resizing here would shift token ids "
        "relative to the base model; load the tokenizer that trained it instead."
      )
  return model


def _enable_fp8(model) -> bool:
  """Convert Linear layers to torchao FP8. Returns whether anything converted.

  Off by default. At 103M the per-call cast overhead exceeds the FP8 matmul
  win, and recovering it needs `torch.compile` to fuse the casts.
  """
  import torch.nn as nn

  try:
    from torchao.float8 import convert_to_float8_training
  except ImportError:
    print("fp8 (torchao)   : not installed; staying on BF16")
    return False

  n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
  try:
    convert_to_float8_training(model)
    from torchao.float8.float8_linear import Float8Linear

    n_fp8 = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
  except Exception as e: # noqa: BLE001 - a failed conversion must not kill a run
    print(f"fp8 (torchao)   : failed ({type(e).__name__}: {e}); staying on BF16")
    return False

  if n_fp8 == 0:
    print(f"fp8 (torchao)   : converted 0/{n_linear} Linear layers; staying on BF16")
    return False
  print(f"fp8 (torchao)   : enabled, {n_fp8}/{n_linear} Linear layers")
  return True


def prepare(model, run, *, device: str | None = None, dtype=None) -> tuple[object, bool]:
  """Move to device, cast, and apply the optional accelerators.

  Returns `(model, fp8_active)`. The flag is not cosmetic: it doubles the
  tensor-core peak used as the MFU denominator, and reporting BF16 MFU against
  an FP8 run would overstate utilisation by 2x.
  """
  import torch

  device = device or run.device
  if dtype is None:
    dtype = torch.bfloat16 if run.dtype == "bfloat16" else torch.float16

  model = model.to(device=device, dtype=dtype)

  if run.use_grad_checkpt:
    model.config.use_cache = False
    # Non-reentrant is the supported path for tied embeddings and does not
    # require any input tensor to carry requires_grad.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpoint: enabled")

  fp8_active = _enable_fp8(model) if run.use_fp8 else False

  if run.use_torch_compile:
    try:
      model = torch.compile(model)
      print("torch.compile   : enabled, the first step will trace")
    except Exception as e: # noqa: BLE001
      print(f"torch.compile   : skipped ({type(e).__name__}: {e})")

  return model, fp8_active


def set_seed(seed: int) -> None:
  import numpy as np
  import torch

  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def enable_tf32() -> None:
  """TF32 for the fp32 matmuls that remain outside autocast."""
  import torch

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.set_float32_matmul_precision("high")
