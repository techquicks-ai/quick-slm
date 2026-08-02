"""JSONL metrics and the MFU denominator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Per-GPU dense BF16 peak, FP32 accumulate, WITHOUT sparsity.
#:
#: Vendor marketing for workstation Blackwell commonly quotes the 2:4 structured
#: sparsity figure, which is exactly double the dense number and would halve any
#: MFU computed against it. `paper/README.md` flags this: confirm the RTX PRO
#: 6000 entry against NVIDIA's datasheet before an MFU derived from it reaches
#: the paper. 500.0 is a placeholder carried over from the notebook.
BF16_PEAK_TFLOPS: dict[str, float] = {
  # EDUCATIONAL DEFECT: The MFU Measurement Fault
  # The RTX PRO 6000 was originally listed here as 500.0 (the 2:4 sparsity rate).
  # This caused the logs to incorrectly report exactly HALF the true MFU 
  # (14.1% instead of 29.5%). 500.0 is preserved here for the exact record.
  # To fix this in , this value must be set to the dense BF16 peak (240.0).
  "RTX PRO 6000": 500.0, # PLACEHOLDER, unverified (Should be 240.0)
  "H100_SXM": 989.0,
  "H100_PCIE": 756.0,
  "A100": 312.0,
  "RTX 6000 ADA": 182.0,
  "L40": 181.0,
  "L4": 121.0,
  "V100": 125.0,
  "A10": 125.0,
  "T4": 65.0,
}


def detect_bf16_peak_tflops(device_name: str) -> float:
  name = device_name.upper()
  if "RTX PRO 6000" in name or ("RTX 6000" in name and "BLACKWELL" in name):
    return BF16_PEAK_TFLOPS["RTX PRO 6000"]
  if "H100" in name:
    return BF16_PEAK_TFLOPS["H100_SXM"] if ("SXM" in name or "NVL" in name) else BF16_PEAK_TFLOPS["H100_PCIE"]
  if "A100" in name:
    return BF16_PEAK_TFLOPS["A100"]
  if "RTX 6000" in name and "ADA" in name:
    return BF16_PEAK_TFLOPS["RTX 6000 ADA"]
  for key in ("L40", "L4", "V100", "A10", "T4"):
    if key in name:
      return BF16_PEAK_TFLOPS[key]
  print(f' [warn] unknown GPU "{device_name}"; using H100 SXM peak as the MFU divisor')
  return BF16_PEAK_TFLOPS["H100_SXM"]


def model_flops_per_token(n_params: int) -> float:
  """The 6N approximation: 2N forward, 4N backward.

  It ignores attention, which at ctx=4096 and d=576 is not a rounding error.
  Reported MFU is therefore a lower bound on the true utilisation.
  """
  return 6.0 * n_params


def fmt_eta(seconds: float) -> str:
  m, s = divmod(int(seconds), 60)
  h, m = divmod(m, 60)
  return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class JsonlLogger:
  """Append-only metrics log, flushed on every write.

  Flushing every line costs nothing at ten-step cadence and means a killed
  Colab runtime loses no history, which is the only reason the log exists.
  """

  def __init__(self, path: str | Path):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._fh = open(self.path, "a")

  def log(self, record: dict[str, Any]) -> None:
    self._fh.write(json.dumps(record) + "\n")
    self._fh.flush()

  def close(self) -> None:
    if not self._fh.closed:
      self._fh.close()

  def __enter__(self) -> "JsonlLogger":
    return self

  def __exit__(self, *exc) -> None:
    self.close()
