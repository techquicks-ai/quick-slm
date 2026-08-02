"""Every framework symbol its notebooks name must still exist.

The other half of the freeze. `test_v1_frozen.py` pins what the presets
*produce*; this pins the API surface the notebooks *call*. A function renamed
while building , or an argument dropped from a signature, does not change any
config value and so slips past the frozen snapshot entirely. It surfaces instead
on Colab, after Drive is mounted and the package is installed, which is the
worst place and the latest time to learn it.

Resolution is static and cheap: parse each notebook's code cells, then import
each `quick_slm_trainer` module they name and getattr every symbol off it. No
notebook is executed and no GPU is involved.
"""

from __future__ import annotations

import ast
import glob
import importlib
import json
import types
from pathlib import Path

import pytest

from quick_slm_trainer.support import load_window

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
NOTEBOOKS = sorted(glob.glob(str(REPO / "training" / "" / "notebooks" / "*.ipynb")))

# Same gate as test_v1_frozen: the API surface is pinned only while is
# supported. After that the notebooks refuse to run against this framework
# anyway, from the same framework.json, so pinning it here would forbid renames
# for a version that can no longer use them.
_WINDOW = load_window("", REPO)
pytestmark = pytest.mark.skipif(
  _WINDOW.is_end_of_life,
  reason=(
    f"training is end-of-life (last framework {_WINDOW.last_supported_framework}, "
    f"archived at {_WINDOW.archive_tag}); its notebooks pin to that build. See SUPPORT.md."
  ),
)

#: Modules that pull torch at import. The suite is required to run without torch
#: (pyproject says so, and CI has no GPU), so these are resolved only when torch
#: happens to be installed. Anything skipped is reported, so the set cannot
#: quietly grow to cover the whole package.
TORCH_BACKED = {"quick_slm_trainer.sft.dataset", "quick_slm_trainer.pretraining.dataset"}


def _code_of(path: str) -> str:
  """The notebook's Python, with Colab-only lines and IPython magics dropped."""
  cells = json.load(open(path))["cells"]
  return "\n".join(
    "\n".join(
      line
      for line in "".join(c["source"]).splitlines()
      if "google.colab" not in line and not line.lstrip().startswith(("!", "%"))
    )
    for c in cells
    if c["cell_type"] == "code"
  )


def _imports(tree: ast.AST) -> list[tuple[str, str]]:
  """(module, symbol) for every `from quick_slm_trainer... import symbol`."""
  out = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quick_slm_trainer"):
      out.extend((node.module, a.name) for a in node.names)
  return out


def _module_aliases(tree: ast.AST) -> dict[str, str]:
  """Local name -> module path, for `import x as y` and `from p import module`."""
  aliases = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        if a.name.startswith("quick_slm_trainer"):
          aliases[a.asname or a.name.split(".")[0]] = a.name
    elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quick_slm_trainer"):
      for a in node.names:
        full = f"{node.module}.{a.name}"
        try:
          if isinstance(importlib.import_module(full), types.ModuleType):
            aliases[a.asname or a.name] = full
        except ImportError:
          continue
  return aliases


def _resolves(module, symbol: str) -> bool | None:
  """True, False, or None when torch is needed to tell.

  `hasattr` is not enough here. `pretraining` and `sft` expose their Dataset
  through a module `__getattr__` that imports torch on first touch, which is
  what lets notebooks 01 and 04 run on a CPU runtime. `hasattr` propagates that
  ImportError rather than returning False, so a plain check reports a missing
  symbol on any machine without torch.
  """
  try:
    return hasattr(module, symbol)
  except ImportError as e:
    if "torch" in str(e).lower():
      return None
    raise


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_every_symbol_the_notebook_imports_still_exists(path):
  tree = ast.parse(_code_of(path))
  skipped, missing = [], []

  for module_name, symbol in _imports(tree):
    try:
      module = importlib.import_module(module_name)
    except ImportError as e:
      assert module_name in TORCH_BACKED or "torch" in str(e).lower(), (
        f"{module_name} failed to import for a reason unrelated to torch: {e}"
      )
      skipped.append(module_name)
      continue
    present = _resolves(module, symbol)
    if present is None:
      skipped.append(f"{module_name}.{symbol}")
    elif not present:
      missing.append(f"{module_name}.{symbol}")

  assert not missing, (
    f"{Path(path).name} imports names the framework no longer provides:\n "
    + "\n ".join(sorted(set(missing)))
    + "\nv1 is frozen: add the new form alongside rather than renaming."
  )
  if skipped:
    print(f"{Path(path).name}: {len(set(skipped))} torch-backed module(s) not resolved")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_every_module_attribute_the_notebook_reads_still_exists(path):
  # Catches `G.distinct_turns_per_seed` style access, which the import check
  # above cannot see because only the module was imported.
  tree = ast.parse(_code_of(path))
  aliases = _module_aliases(tree)
  missing = []
  for node in ast.walk(tree):
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
      continue
    if node.value.id not in aliases:
      continue
    module_name = aliases[node.value.id]
    if _resolves(importlib.import_module(module_name), node.attr) is False:
      missing.append(f"{module_name.split('.')[-1]}.{node.attr}")
  assert not missing, (
    f"{Path(path).name} reads attributes the framework no longer provides:\n "
    + "\n ".join(sorted(set(missing)))
  )


def test_the_notebooks_are_actually_being_checked():
  # Guards the guard. If the notebooks move again and the glob silently returns
  # nothing, every parametrised test above vanishes and the suite still passes.
  assert len(NOTEBOOKS) >= 8, f"expected its eight notebooks, found {len(NOTEBOOKS)}"
  names = {Path(p).name for p in NOTEBOOKS}
  assert {"01_data_preparation.ipynb", "04_sft_data.ipynb", "05_sft_train.ipynb"} <= names


# --------------------------------------------------------------------------
# Structural invariants
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_every_cell_stores_its_source_as_a_list_of_lines(path):
  # nbformat permits `source` to be a plain string, and these notebooks carried
  # both forms. That is not cosmetic. Any tool written as `for line in
  # cell["source"]` iterates *characters* on a string cell and silently matches
  # nothing, which is how 05_sft_train.ipynb kept a `REPO_DIR / "src"` that no
  # longer resolves, through both a rewrite and the check meant to catch it.
  # One representation removes the whole class of failure.
  cells = json.load(open(path))["cells"]
  strings = [i for i, c in enumerate(cells) if isinstance(c["source"], str)]
  assert not strings, (
    f"{Path(path).name}: cells {strings} store source as a string rather than a "
    "list of lines; line-oriented tooling will silently skip them"
  )
  ragged = [
    i for i, c in enumerate(cells)
    if "".join(c["source"]).splitlines(keepends=True) != c["source"]
  ]
  assert not ragged, f"{Path(path).name}: cells {ragged} do not split one line per element"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_a_notebook_that_installs_the_framework_checks_its_support_window(path):
  # The repo-side tests cannot reach a Colab runtime. This check is the only
  # thing standing between a frozen notebook and a future framework that would
  # build a different corpus without complaining.
  text = "\n".join("".join(c["source"]) for c in json.load(open(path))["cells"])
  if "REPO_DIR" not in text:
    pytest.skip("self-contained notebook; it never installs the framework")
  assert "require_framework" in text, (
    f"{Path(path).name} installs the framework but never checks its support "
    "window; see SUPPORT.md"
  )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: Path(p).name)
def test_no_notebook_references_the_pre_split_layout(path):
  text = "\n".join("".join(c["source"]) for c in json.load(open(path))["cells"])
  for stale in ("REPO_DIR / 'src'", 'REPO_DIR / "src"', "training/colab/"):
    assert stale not in text, f"{Path(path).name} still references {stale!r}"
