"""The four pretraining corpora, as (iterator, text_fn) pairs.

Filters are stated as configuration rather than buried in a closure, because
the paper quotes them and a reviewer will want to check the number against the
code that produced the corpus.

A caution on both FineWeb-Edu and FineMath: the schema carries two score
columns. `score` is the classifier's continuous output, roughly 0 to 5.
`int_score` is the rounded integer tier, 1 to 5, and it is the one to compare
against. Filtering on `score` silently admits a different corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from ..config import DataConfig
from ..template import render_pretrain_example


@dataclass
class Source:
  key: str
  documents: Iterable
  text_fn: Callable[[object], str | None]


# --------------------------------------------------------------------------
# FineWeb-Edu: tier 4 and above
# --------------------------------------------------------------------------
def _fineweb_edu(cfg: DataConfig) -> Source:
  """Educational web text, the language base at 77.5% of the mixture.

  Tier-4+ rather than strict tier-5. Strict tier-5 is roughly 3% of the
  corpus, so the streamer downloads and discards ~97% of every byte, which
  projected to a ~24-day wall-clock for the 7.75B budget. Tier-4+ takes the
  hit rate to ~25% and the wall-clock to ~3 days. At 103M the quality gap
  between tier 4 and tier 5 is smaller than the volume gain.
  """
  from datasets import load_dataset

  stream = (
    load_dataset("HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True)
    # Tell the parquet reader to skip url/dump/language/etc. Every row
    # download becomes several times smaller, and it costs no bandwidth to ask.
    .select_columns(["text", "int_score"])
    # Interleave shards so the stream does not sit inside one CommonCrawl
    # dump for hours. The buffer is small enough for a standard runtime.
    .shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)
  )
  threshold = cfg.fineweb_min_int_score

  def text_fn(row):
    return row.get("text") if (row.get("int_score") or 0) >= threshold else None

  return Source("fineweb_edu", stream, text_fn)


# --------------------------------------------------------------------------
# FineMath: strict tier 5, inside the already-filtered 4plus config
# --------------------------------------------------------------------------
def _finemath(cfg: DataConfig) -> Source:
  """Numeric reasoning, at 10%.

  Strict tier-5 here, unlike FineWeb-Edu, because `finemath-4plus` is small
  enough that strict filtering still finishes fast. Math quality matters
  disproportionately for tool arguments: many are integers, and a model that
  cannot reliably decide whether 23 exceeds 25 will bind the wrong one.
  """
  from datasets import load_dataset

  stream = load_dataset(
    "HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True
  ).select_columns(["text", "int_score"])
  threshold = cfg.finemath_min_int_score

  def text_fn(row):
    return row.get("text") if (row.get("int_score") or 0) >= threshold else None

  return Source("finemath", stream, text_fn)


# --------------------------------------------------------------------------
# Tool calling: xLAM for `xlam_epochs` passes, then Glaive fills the budget
# --------------------------------------------------------------------------
def format_xlam(row) -> str:
  """Render one xLAM row into the flat pretraining template.

  Delegates to `template.render_pretrain_example` so the corpus and the
  capability probes cannot disagree about the surface form.
  """
  return render_pretrain_example(
    query=row.get("query") or "",
    tools=row.get("tools"),
    answers=row.get("answers"),
  )


def format_glaive(row) -> str | None:
  """Glaive is yielded verbatim, in its own `<functioncall>` chat format.

  Deliberately not normalised into the xLAM template. Two surface styles
  teach the model that the *structure* of a tool call generalises, rather than
  letting it memorise one arrangement of characters. The canonical template
  is imposed later, at SFT.
  """
  system = (row.get("system") or "").strip()
  chat = (row.get("chat") or "").strip()
  if not chat:
    return None
  return f"{system}\n\n{chat}" if system else chat


def _tool_calling(cfg: DataConfig) -> Source:
  """Sequential, not round-robin.

  Alternating the two sources would let Glaive's longer documents dominate
  the token count regardless of how many rows each contributed. Running xLAM
  to completion first makes the split predictable: three passes over xLAM's
  ~30M unique tokens give ~90M, comfortably inside the 4-epoch repetition
  ceiling (Muennighoff et al., 2023), and Glaive fills the remaining ~160M.
  """
  from datasets import load_dataset

  xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
  print(f"xLAM rows: {len(xlam):,} ({cfg.xlam_epochs} epoch(s) before Glaive)")

  def documents() -> Iterator[tuple[str, object]]:
    for _ in range(cfg.xlam_epochs):
      for row in xlam:
        yield ("xlam", row)
    glaive = load_dataset("glaiveai/glaive-function-calling-", split="train", streaming=True)
    for row in glaive:
      yield ("glaive", row)

  def text_fn(item):
    origin, row = item
    if origin == "xlam":
      return format_xlam(row)
    if origin == "glaive":
      return format_glaive(row)
    return None

  return Source("tool_calling", documents(), text_fn)


# --------------------------------------------------------------------------
# StarCoder: Python only
# --------------------------------------------------------------------------
def _starcoder(cfg: DataConfig) -> Source:
  """Bracket discipline and procedural surface, at 10%. One language, to keep
  the run focused."""
  from datasets import load_dataset

  langs = cfg.starcoder_langs

  def documents():
    for lang in langs:
      ds = load_dataset("bigcode/starcoderdata", data_dir=lang, split="train", streaming=True)
      yield from ds

  return Source("starcoder", documents(), lambda row: row.get("content"))


SOURCES: dict[str, Callable[[DataConfig], Source]] = {
  "fineweb_edu": _fineweb_edu,
  "finemath": _finemath,
  "tool_calling": _tool_calling,
  "starcoder": _starcoder,
}


def build_source(key: str, cfg: DataConfig) -> Source:
  if key not in SOURCES:
    raise KeyError(f"unknown source {key!r}; known: {sorted(SOURCES)}")
  return SOURCES[key](cfg)
