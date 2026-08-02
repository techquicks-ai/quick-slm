"""One training loop, both stages.

The loop only ever sees `(input_ids, labels)`. Pretraining passes
`labels = input_ids`; SFT passes labels with everything outside an assistant
turn set to `IGNORE_INDEX`. Nothing else about the two stages differs, so
nothing else about them is forked.

Gradient accumulation, and why it is not a division by `grad_accum`
------------------------------------------------------------------
`LlamaForCausalLM(labels=...)` returns the *mean* cross-entropy over the scored
tokens in that micro-batch: `loss_i = sum_i / n_i`. The conventional
`loss_i / grad_accum` therefore accumulates

  (1/A) * sum_i (sum_i / n_i)

which is the mean of per-micro-batch means. That equals the true mean only when
every `n_i` is identical. Under pretraining it is: every token is scored, so
`n_i = B * (ctx - 1)` for all i, and the conventional form is exact. Under SFT
it is not, because the fraction of a packed window that lies inside an assistant
turn swings from a few percent to most of it. The conventional form would then
weight a sparsely-scored micro-batch as heavily as a densely-scored one, which
upweights whichever windows happen to be mostly context.

Scaling instead by `n_i / N`, where `N = sum_i n_i` over the step, accumulates

  sum_i (sum_i / n_i) * (n_i / N) = (sum_i sum_i) / N

which is the exact token-mean over the whole effective batch. It is an identity,
needs nothing from the transformers loss internals, and reduces to the
conventional form whenever the `n_i` are equal. The summed micro-batch losses
are also, for free, the correct step loss to report.

Collecting the micro-batches before the backward pass is what makes `N` known in
advance. They are CPU tensors at that point; at the shape that is 8.5 MB.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import checkpoint as ckpt
from .checkpoint import TrainState
from .config import Config
from .evaluate import scored_tokens
from .logging import JsonlLogger, detect_bf16_peak_tflops, fmt_eta, model_flops_per_token
from .model import param_count, unwrap
from .schedule import make_lr_fn


@dataclass
class Trainer:
  """A resumable loop over `(input_ids, labels)` batches.

  `make_loader(start_window, seed)` must return a fresh DataLoader. It is
  called once at start and again whenever the corpus is exhausted, which under
  SFT is once per epoch.
  """

  config: Config
  model: object
  optimizer: object
  make_loader: Callable[[int, int], object]
  total_steps: int
  ctx: int
  ckpt_dir: Path
  log_path: Path
  tokenizer: object = None
  evaluate: Callable[[], dict] | None = None
  state: TrainState = None
  title: str = "quick-slm"
  fp8_active: bool = False
  save_hf: bool = True

  def __post_init__(self) -> None:
    import torch

    self.ckpt_dir = Path(self.ckpt_dir)
    self.log_path = Path(self.log_path)
    self.state = self.state or TrainState()
    self.n_params = param_count(self.model)

    run = self.config.run
    self.device = run.device
    self.dtype = torch.bfloat16 if run.dtype == "bfloat16" else torch.float16
    self.tokens_per_step = run.tokens_per_step(self.ctx)

    peak = detect_bf16_peak_tflops(
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    # FP8 doubles the tensor-core peak on Blackwell and Hopper, so the MFU
    # denominator has to double with it.
    self.gpu_peak_tflops = peak * (2.0 if self.fp8_active else 1.0)
    self.peak_label = "FP8" if self.fp8_active else "BF16"

  # ------------------------------------------------------------------
  def _autocast(self):
    import torch

    device_type = "cuda" if self.device.startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type, dtype=self.dtype, enabled=device_type == "cuda")

  def _banner(self) -> None:
    run = self.config.run
    print("=" * 72)
    print(f" {self.title}")
    print(f" params    : {self.n_params / 1e6:.2f}M")
    print(
      f" effective bs : {run.effective_batch} seq x {self.ctx} ctx "
      f"= {self.tokens_per_step:,} tok/step"
    )
    print(f" total steps  : {self.total_steps:,}")
    print(f" resume step  : {self.state.step:,} (tokens seen: {self.state.tokens_seen:,})")
    print(f" lr      : {self.config.optim.lr_peak:.2e} peak -> {self.config.optim.lr_min:.2e} min")
    print(f" warmup    : {self.config.optim.warmup_steps:,} steps")
    print(f" save / log / eval every: {run.save_every_steps} / {run.log_every_steps} / {run.eval_every_steps}")
    print(f" {self.peak_label} peak   : {self.gpu_peak_tflops:.0f} TFLOPS (MFU divisor)")
    print("=" * 72)

  def _gpu_util(self) -> int:
    import torch

    try:
      return int(torch.cuda.utilization())
    except Exception: # noqa: BLE001 - a flaky NVML must never abort a run
      return 0

  # ------------------------------------------------------------------
  def train(self) -> TrainState:
    import torch
    from tqdm.auto import tqdm

    run = self.config.run
    lr_fn = make_lr_fn(self.config.optim, self.total_steps)
    state = self.state

    self._banner()

    loader = self.make_loader(state.next_window, run.seed + state.step)
    iterator = iter(loader)

    self.model.train()
    self.optimizer.zero_grad(set_to_none=True)

    logger = JsonlLogger(self.log_path)
    pbar = tqdm(
      total=self.total_steps,
      initial=state.step,
      desc="train",
      unit="step",
      smoothing=0.05,
      dynamic_ncols=True,
    )

    t_start = time.time()
    step_t0 = time.time()
    loss_ema, loss_alpha = None, 0.05
    util, gpu_mem_gb = 0, 0.0

    try:
      for step in range(state.step, self.total_steps):
        lr = lr_fn(step)
        for g in self.optimizer.param_groups:
          g["lr"] = lr

        # --- collect the step's micro-batches ------------------
        micro = []
        for _ in range(run.grad_accum):
          try:
            batch = next(iterator)
          except StopIteration:
            state.epoch += 1
            state.next_window = 0
            loader = self.make_loader(0, run.seed + step + 1)
            iterator = iter(loader)
            batch = next(iterator)
          micro.append(batch)
          state.next_window += int(batch[0].size(0))

        counts = [scored_tokens(y) for _, y in micro]
        total_items = sum(counts)
        if total_items == 0:
          raise RuntimeError(
            f"step {step}: no scored tokens in the effective batch. "
            "Every label is IGNORE_INDEX, so there is nothing to learn from."
          )

        # --- forward / backward --------------------------------
        accum_loss_t = torch.zeros((), device=self.device)
        for (x, y), n in zip(micro, counts):
          if n == 0:
            continue # an all-context window contributes no gradient
          x = x.to(self.device, non_blocking=True)
          y = y.to(self.device, non_blocking=True)
          with self._autocast():
            out = self.model(input_ids=x, labels=y)
          loss = out.loss * (n / total_items)
          loss.backward()
          accum_loss_t += loss.detach()

        grad_norm_t = torch.nn.utils.clip_grad_norm_(
          unwrap(self.model).parameters(), self.config.optim.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        accum_loss = float(accum_loss_t.item())
        grad_norm = float(grad_norm_t.item())

        # --- metrics -------------------------------------------
        state.step = step + 1
        state.tokens_seen += self.tokens_per_step
        step_dt = time.time() - step_t0
        step_t0 = time.time()
        tps = self.tokens_per_step / max(step_dt, 1e-9)
        # The 6N approximation ignores attention and the recompute that
        # gradient checkpointing adds, so this is a lower bound.
        achieved_tflops = model_flops_per_token(self.n_params) * tps / 1e12
        mfu = 100.0 * achieved_tflops / self.gpu_peak_tflops
        loss_ema = accum_loss if loss_ema is None else (1 - loss_alpha) * loss_ema + loss_alpha * accum_loss

        pbar.update(1)
        pbar.set_postfix(
          {
            "loss": f"{loss_ema:.3f}",
            "lr": f"{lr:.2e}",
            "tok/s": f"{tps / 1e3:.1f}k",
            "mfu": f"{mfu:.1f}%",
            "gpu": f"{gpu_mem_gb:.1f}GB",
          }
        )

        if state.step % run.log_every_steps == 0:
          util = self._gpu_util()
          gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
          eta = (self.total_steps - state.step) * step_dt
          tqdm.write(
            f"step {state.step:>6d}/{self.total_steps} "
            f"loss {loss_ema:.3f} lr {lr:.2e} tok/s {tps / 1e3:.1f}k "
            f"TFLOPS {achieved_tflops:.0f} util {util:d}% mfu {mfu:.1f}% "
            f"grad {grad_norm:.2f} tokens {state.tokens_seen / 1e9:.2f}B eta {fmt_eta(eta)}"
          )
          logger.log(
            {
              "step": state.step,
              "loss": accum_loss,
              "loss_ema": loss_ema,
              "lr": lr,
              "grad_norm": grad_norm,
              "scored_tokens": total_items,
              "tokens_seen": state.tokens_seen,
              "tokens_per_sec": tps,
              "wall": time.time() - t_start,
              "achieved_tflops": achieved_tflops,
              "mfu_pct": mfu,
              "gpu_peak_tflops": self.gpu_peak_tflops,
              "gpu_util_pct": util,
              "gpu_mem_gb": gpu_mem_gb,
              "epoch": state.epoch,
              "fp8_active": self.fp8_active,
            }
          )

        if self.evaluate is not None and state.step % run.eval_every_steps == 0:
          if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
          metrics = self.evaluate()
          state.val_loss = metrics["loss"]
          tqdm.write(f" eval@{state.step}: loss={metrics['loss']:.4f} ppl={metrics['ppl']:.2f}")
          logger.log({"step": state.step, "eval_loss": metrics["loss"], "eval_ppl": metrics["ppl"]})

        if state.step % run.save_every_steps == 0 or state.step == self.total_steps:
          path = self.save(state.step)
          tqdm.write(f" saved {path}")
          logger.log({"step": state.step, "checkpoint": str(path)})
    finally:
      pbar.close()
      logger.close()

    print("Training complete.")
    return state

  # ------------------------------------------------------------------
  def save(self, step: int) -> Path:
    run = self.config.run
    path = ckpt.save(
      self.ckpt_dir,
      step=step,
      state=self.state,
      model=self.model,
      optimizer=self.optimizer,
      config=self.config,
      tokenizer=self.tokenizer,
      save_hf=self.save_hf,
    )
    ckpt.prune(
      self.ckpt_dir,
      keep_last_n=run.keep_last_n_ckpts,
      total_steps=self.total_steps,
      save_every=run.save_every_steps,
      deciles=run.decile_milestones,
    )
    return path


def resume_if_possible(ckpt_dir, model, optimizer) -> TrainState:
  """Load the highest-step checkpoint under `ckpt_dir`, or start fresh."""
  latest = ckpt.find_latest(ckpt_dir)
  if latest is None:
    print("No checkpoint found; starting from scratch.")
    return TrainState()
  print(f"Resuming from {latest}")
  state = ckpt.load_for_resume(latest, model, optimizer)
  print(
    f" step={state.step:,} tokens_seen={state.tokens_seen:,} "
    f"next_window={state.next_window:,} epoch={state.epoch}"
  )
  return state


def steps_for_epochs(n_windows: int, effective_batch: int, epochs: int) -> int:
  """Total optimizer steps to make `epochs` passes over `n_windows` windows."""
  return max(1, math.floor(n_windows * epochs / effective_batch))
