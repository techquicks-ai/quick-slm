"""The paper's architecture figure, against the config it claims to describe.

Figure 1 states the hidden size, the head counts, the FFN width and the whole
parameter ledger. A diagram is the one part of a paper nobody re-derives while
reading, so a stale one is believed. These tests fail if the committed SVG stops
matching `ModelConfig`, which is the only way the two can be kept honest without
a human comparing eleven numbers by eye.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from quick_slm_trainer.config import ModelConfig
from quick_slm_trainer.tokenizer import SPECIAL_TOKENS

# Found rather than counted. This file has already moved once, from tests/ to
# framework/tests/, and a hardcoded `parents[n]` is the kind of thing that then
# resolves to a directory that happens to exist and fails somewhere less obvious.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

#: The figure belongs to a training version, not to the framework: will bring
#: its own paper, its own architecture, and its own Figure 1.
PAPER_DIR = REPO / "training" / "" / "paper"
SVG = PAPER_DIR / "figures" / "architecture.svg"
PAPER = PAPER_DIR / "index.html"
SCRIPT = PAPER_DIR / "scripts" / "plot_architecture.py"


@pytest.fixture(scope="module")
def plot():
  spec = importlib.util.spec_from_file_location("plot_architecture", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def vocab() -> int:
  return 32_000 + len(SPECIAL_TOKENS)


def test_the_committed_figure_is_what_the_script_emits(plot, vocab):
  # Regenerate with: python paper/scripts/plot_architecture.py
  assert SVG.exists(), f"{SVG.relative_to(REPO)} is missing; run the script"
  assert SVG.read_text() == plot.build(ModelConfig(), vocab), (
    "paper/figures/architecture.svg is stale; "
    "re-run python paper/scripts/plot_architecture.py"
  )


def test_the_figure_inlined_in_the_paper_is_the_same_figure(plot, vocab):
  # The markup now lives in two files, which is exactly the drift this suite
  # exists to prevent. index.html is what a reader sees, so it is the copy that
  # matters most; the standalone .svg only feeds the PNG export.
  paper = PAPER.read_text()
  assert paper == plot.inline_into(paper, plot.build(ModelConfig(), vocab)), (
    "the figure inlined in paper/index.html is stale; "
    "re-run python paper/scripts/plot_architecture.py"
  )


def test_inlining_is_idempotent(plot, vocab):
  # The script rewrites the region between two comments on every run. If a run
  # left the markers doubled or consumed them, the next run would corrupt the
  # paper instead of updating it.
  svg = plot.build(ModelConfig(), vocab)
  once = plot.inline_into(PAPER.read_text(), svg)
  assert plot.inline_into(once, svg) == once
  assert once.count(plot.BEGIN) == 1 and once.count(plot.END) == 1


def test_the_inline_figure_defines_no_bare_ids(plot, vocab):
  # Inline SVG shares the document's id namespace. An unprefixed id="arrow"
  # would be claimed by whichever element reached it first the moment the paper
  # grew a second diagram, and the arrowheads would silently point at the wrong
  # marker.
  svg = plot.build(ModelConfig(), vocab)
  ids = re.findall(r'\sid="([^"]+)"', svg)
  assert ids, "the figure defines no ids at all; this test has stopped checking anything"
  for name in ids:
    assert name.startswith(plot.IDP), f"unprefixed id in the figure: {name!r}"
  # And every reference resolves to one of them.
  for ref in re.findall(r"url\(#([^)]+)\)", svg):
    assert ref in ids, f"dangling marker reference: {ref!r}"


def test_the_ledger_sums_to_the_total_it_prints(plot, vocab):
  # The figure's own arithmetic, checked independently of how it is drawn. The
  # ledger is two additions, not one: the indented rows decompose a block, and
  # the flush rows are what reaches the total. Both are asserted, because a
  # ledger whose column does not add up is worse than no ledger.
  cfg = ModelConfig()
  d, L = cfg.hidden_size, cfg.num_hidden_layers
  p_attn = 2 * d * d + 2 * d * cfg.num_key_value_heads * cfg.head_dim
  p_ffn = 3 * d * cfg.intermediate_size
  p_norm = 2 * d
  p_block = p_attn + p_ffn + p_norm
  p_total = vocab * d + L * p_block + d

  svg = plot.build(cfg, vocab)
  shown = {int(n.replace(",", "")) for n in re.findall(r">([\d,]+)<", svg)}

  assert p_attn + p_ffn + p_norm == p_block
  assert vocab * d + L * p_block + d == p_total
  for value in (p_attn, p_ffn, p_norm, p_block, vocab * d, L * p_block, d, p_total):
    assert value in shown, f"{value:,} is not printed in the ledger"


def test_the_figure_does_not_contradict_section_3_6_on_the_8_3_rule(plot, vocab):
  # Section 3.6 turns on the ratio being exact at this width: three gated
  # matrices cost precisely what two ungated ones cost at 4d, so SwiGLU is
  # adopted at zero parameter cost. A draft of the figure said d_ff was "8/3 x
  # 576 rounded to a multiple of 64", which asserts the opposite of the section
  # it sits beside. If the width ever moves off the exact ratio, this fails here
  # rather than in a reader's head.
  cfg = ModelConfig()
  d, d_ff = cfg.hidden_size, cfg.intermediate_size
  assert d_ff * 3 == 8 * d, "d_ff is no longer exactly 8/3 of d"
  assert 3 * d * d_ff == 2 * d * (4 * d)

  svg = plot.build(cfg, vocab)
  assert "rounded" not in svg
  assert f"8/3 × {d} = {d_ff:,}" in svg


def test_the_figure_agrees_with_the_papers_parameter_count(plot, vocab):
  # Section 3.1 of index.html states this number three times, in the prose, in
  # Table 1 and in the equation. The figure is the fourth place it appears.
  svg = plot.build(ModelConfig(), vocab)
  paper = PAPER.read_text()
  assert "103,402,944" in svg
  assert "103,402,944" in paper


def test_every_hyperparameter_the_figure_states_comes_from_the_config(plot, vocab):
  cfg = ModelConfig()
  svg = plot.build(cfg, vocab)
  for value in (
    f"{vocab:,} × {cfg.hidden_size}",      # embedding shape
    f"× {cfg.num_hidden_layers}",        # depth
    f"{cfg.num_attention_heads} Q / {cfg.num_key_value_heads} KV heads",
    f"d_h {cfg.head_dim}",
    f"d_ff {cfg.intermediate_size:,}",
    f"{cfg.max_position_embeddings:,}",     # context
  ):
    assert value in svg, value


def test_a_changed_config_changes_the_figure(plot, vocab):
  # The guard only means something if the output actually tracks the input.
  deeper = ModelConfig(num_hidden_layers=ModelConfig().num_hidden_layers + 1)
  assert plot.build(deeper, vocab) != plot.build(ModelConfig(), vocab)


def test_no_left_anchored_label_runs_into_the_number_to_its_right(plot, vocab):
  # The ledger is three columns faked with text anchors, so nothing stops a long
  # expression from being drawn straight through the value beside it. Spelling
  # out "884,736 + 2,654,208 + 1,152" did exactly that. Georgia at 9pt averages
  # well under 0.5em per character; 0.5 is used as a deliberately generous bound,
  # so this flags real overlap rather than tight-but-fine spacing.
  svg = plot.build(ModelConfig(), vocab)
  drawn = re.findall(
    r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*'
    r'text-anchor="(\w+)"[^>]*>([^<]*)</text>',
    svg,
  )
  starts = [(float(x), float(y), float(s), t) for x, y, s, a, t in drawn if a == "start"]
  ends = [(float(x), float(y), t) for x, y, _s, a, t in drawn if a == "end"]

  for sx, sy, size, text in starts:
    width = 0.5 * size * len(text)
    for ex, ey, etext in ends:
      if abs(ey - sy) < 1 and ex > sx:      # same line, to the right
        right_edge = ex - 0.5 * 9.5 * len(etext)
        assert sx + width <= right_edge, (
          f"{text!r} (ends ~{sx + width:.0f}) overlaps {etext!r} (starts ~{right_edge:.0f})"
        )


def test_the_figure_is_well_formed_and_inside_its_canvas(plot, vocab):
  import xml.etree.ElementTree as ET

  svg = plot.build(ModelConfig(), vocab)
  ET.fromstring(svg) # raises if malformed
  # Content that runs off the canvas is clipped by the browser, silently.
  coords = [float(v) for v in re.findall(r'(?:x|y|x1|y1|x2|y2|cx|cy)="(-?[\d.]+)"', svg)]
  assert min(coords) >= 0, "figure content sits outside the top or left edge"


def test_the_paper_carries_the_figure_rather_than_linking_it(plot):
  # index.html is meant to be one self-contained file: opening it from a copy
  # with no `figures/` directory beside it must still show Figure 1.
  paper = PAPER.read_text()
  assert "<strong>Figure 1.</strong>" in paper
  assert "<svg" in paper and f'id="{plot.IDP}arrow"' in paper
  assert 'src="figures/' not in paper, "the figure is linked again rather than inlined"
