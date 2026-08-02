#!/usr/bin/env python3
"""Record what the framework currently resolves its presets to.

  python scripts/freeze_config.py

Writes `frozen_config.json` beside this file. `framework/tests/test_frozen.py`
compares the framework's live output against it on every run.

The run is finished. Its checkpoints exist, and its paper reports numbers
measured from them. The framework, meanwhile, keeps moving: will want new
tools, new sub-types, new presets. Those are additive and safe. What is not safe
is a change that reaches backwards, and the ways that happens are quiet ones. A
default is edited in `SFTConfig`. `recommended_examples_per_seed()` is retuned
for 's larger pools. A category share is adjusted. None of these touch a file
under `training/`, and all of them change what `sft_config()` produces, which is
what the notebooks call.

Running the notebooks would eventually reveal it, on Colab, hours in. This turns
that into a failing test at commit time.

**Regenerating is an act of judgement, not a fix.** If the test fails, the
default is that the framework change is wrong and should be made additive
instead. Re-run this script only when the difference is understood and
deliberate, and say in the commit message what moved and why tolerates it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO / "framework"))

from quick_slm_trainer.config import pretrain_config, sft_config # noqa: E402
from quick_slm_trainer.sft.generate import plan_requests # noqa: E402

OUT = HERE / "frozen_config.json"


def snapshot() -> dict:
  sft = sft_config()
  return {
    "_comment": (
      "Frozen record of what the framework resolves its presets to. Written by "
      "scripts/freeze_config.py, checked by framework/tests/test_frozen.py. "
      "A diff here means a framework change reached backwards into a finished run."
    ),
    "pretrain_config": pretrain_config().to_dict(),
    "sft_config": sft.to_dict(),
    "derived": {
      "_comment": (
        "Not configuration, but computed from it. The planned request count is "
        "what the corpus size and the teacher bill follow from, and it moves when "
        "a cap or a share does, so it is worth pinning separately."
      ),
      "planned_sft_requests": len(plan_requests(sft.sft)),
    },
  }


def main() -> None:
  OUT.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
  print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
  main()
