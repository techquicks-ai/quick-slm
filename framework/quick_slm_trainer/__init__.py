"""quick-slm-trainer: pretraining, SFT, and evaluation for Quick SLM.

Nothing imported here pulls in torch or transformers. That is deliberate: the
config, the template, the loss mask, the LR schedule, and the combine allocation
are the four places where a silent mistake corrupts an entire run, and all four
must be testable on a laptop with nothing installed.

The package is one shared core plus one subpackage per stage. `pretraining` owns
the corpus and the window it reads; `sft` owns the teacher, the filters, and the
masked window. Everything both stages use, which is most of it, stays at the
root: `config`, `template`, `tokenizer`, `model`, `optim`, `schedule`,
`checkpoint`, `evaluate`, `loader`, and the one `Trainer` that serves both.

The torch-dependent halves live one import deeper, and each is loaded only when
asked for:

  from quick_slm_trainer.model import build_model, prepare
  from quick_slm_trainer.trainer import Trainer
  from quick_slm_trainer.loader import make_loader
  from quick_slm_trainer.pretraining import combine_sources, WindowedMemmapDataset
  from quick_slm_trainer.sft import plan_requests, MaskedWindowDataset
"""

from __future__ import annotations

from . import template
from .config import (
  Config,
  DataConfig,
  ModelConfig,
  OptimConfig,
  RunConfig,
  SFTConfig,
  pretrain_config,
  sft_config,
)
from .logging import (
  JsonlLogger,
  detect_bf16_peak_tflops,
  fmt_eta,
  model_flops_per_token,
)
from .manifest import Manifest
from .paths import SOURCE_KEYS, Layout
from .schedule import cosine_with_warmup, make_lr_fn
from .support import SupportError, check_framework, require_framework
from .template import IGNORE_INDEX

#: Semantic, with one project-specific rule: **a major bump means a training
#: version lost support.** 1.x runs `training`; the release that drops it is
#: 2.0.0. Minor and patch releases must leave every supported version's
#: `framework.json` window satisfied, which `test_support_window.py` enforces.
#: See SUPPORT.md for the matrix and the deprecation history.
__version__ = "1.0.0"

__all__ = [
  "Config",
  "DataConfig",
  "IGNORE_INDEX",
  "JsonlLogger",
  "Layout",
  "Manifest",
  "ModelConfig",
  "OptimConfig",
  "RunConfig",
  "SFTConfig",
  "SOURCE_KEYS",
  "SupportError",
  "check_framework",
  "cosine_with_warmup",
  "detect_bf16_peak_tflops",
  "fmt_eta",
  "make_lr_fn",
  "model_flops_per_token",
  "pretrain_config",
  "require_framework",
  "sft_config",
  "template",
  "__version__",
]
