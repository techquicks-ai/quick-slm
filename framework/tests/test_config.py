from __future__ import annotations

import pytest

from quick_slm_trainer.config import Config, ModelConfig, pretrain_config, sft_config


def test_round_trips_through_json(tmp_path):
  cfg = pretrain_config()
  back = Config.load(cfg.save(tmp_path / "config.json"))
  assert back.to_dict() == cfg.to_dict()


def test_tuple_fields_survive_the_round_trip(tmp_path):
  cfg = Config.load(pretrain_config().save(tmp_path / "c.json"))
  assert isinstance(cfg.optim.betas, tuple)
  assert isinstance(cfg.data.starcoder_langs, tuple)


def test_unknown_keys_are_ignored_so_old_checkpoints_still_load():
  # A field removed from ModelConfig must not break a checkpoint that has it.
  d = ModelConfig().to_dict() | {"a_field_that_no_longer_exists": 7}
  assert ModelConfig.from_dict(d).hidden_size == 576


def test_attention_head_dims_are_consistent():
  m = ModelConfig()
  assert m.num_attention_heads * m.head_dim == m.hidden_size
  assert m.num_attention_heads % m.num_key_value_heads == 0


def test_v1_records_what_ran_not_what_was_right():
  # `docs/important_notes.md` argues 6e-4 to 1e-3 was the right band.
  # Correcting this constant would falsify the record of the base checkpoint.
  cfg = pretrain_config()
  assert cfg.optim.lr_peak == 3e-4
  assert cfg.run.effective_batch == 260
  assert cfg.model.attn_implementation == "sdpa"
  assert cfg.run.use_grad_checkpt is False


def test_v1_step_count_matches_the_run_that_happened():
  cfg = pretrain_config()
  assert cfg.run.tokens_per_step(cfg.data.ctx) == 260 * 4096
  assert cfg.run.total_steps(cfg.data.ctx) == 9_390



def test_data_budgets_sum_to_the_advertised_shares():
  shares = pretrain_config().data.shares()
  assert shares["fineweb_edu"] == pytest.approx(0.775)
  assert shares["finemath"] == pytest.approx(0.10)
  assert shares["tool_calling"] == pytest.approx(0.025)
  assert shares["starcoder"] == pytest.approx(0.10)
  assert sum(shares.values()) == pytest.approx(1.0)


def test_sft_config_does_not_mutate_the_config_it_was_given():
  base = pretrain_config()
  sft_config(base)
  assert base.optim.lr_peak == 3e-4, "sft_config rewrote its caller's learning rate"


def test_sft_config_lowers_the_learning_rate_well_below_the_pretraining_peak():
  cfg = sft_config()
  assert cfg.optim.lr_peak == 5e-5
  assert cfg.optim.lr_peak < pretrain_config().optim.lr_peak / 5
  assert cfg.optim.lr_min == cfg.optim.lr_peak / 10
  assert cfg.run.use_grad_checkpt is True


def test_the_sft_schedule_fits_inside_the_corpus_it_will_actually_see():
  # The failure this exists for is silent. `warmup_steps` longer than the run
  # means the learning rate ramps for the whole fine-tune, never reaches its
  # peak and never decays, and nothing raises. its 16-step accumulation was
  # sized for the ~80M-token corpus SFT_README planned; the realised corpus is
  # 14.7M, which left 168 steps against a 200-step warmup. The run that
  # produced the evaluated checkpoint logged exactly these 900 steps and
  # 44,236,800 tokens seen.
  from quick_slm_trainer.schedule import cosine_with_warmup

  PACKED_TRAIN_TOKENS = 14_745_600  # measured: 3,600 windows x 4,096
  EPOCHS = 3

  cfg = sft_config()
  # Pinned because the comment beside `grad_accum` once claimed 491,520 here,
  # which is 's figure; 12 sequences of 4,096 is 49,152. The run confirms it:
  # 900 steps x 49,152 is the 44,236,800 tokens the training log reported.
  assert cfg.run.tokens_per_step(cfg.data.ctx) == 49_152 == 4 * 3 * 4096
  steps = (PACKED_TRAIN_TOKENS * EPOCHS) // cfg.run.tokens_per_step(cfg.data.ctx)
  assert steps > cfg.optim.warmup_steps, (
    f"warmup ({cfg.optim.warmup_steps}) is not shorter than the run ({steps} steps); "
    "the schedule would never leave warmup"
  )
  # And the run must actually anneal, not stop at the top of the cosine.
  last = cosine_with_warmup(steps - 1, lr_peak=cfg.optim.lr_peak, lr_min=cfg.optim.lr_min,
               warmup_steps=cfg.optim.warmup_steps, total_steps=steps)
  assert last == pytest.approx(cfg.optim.lr_min, rel=0.05), "run does not decay to lr_min"
  assert steps == 900
