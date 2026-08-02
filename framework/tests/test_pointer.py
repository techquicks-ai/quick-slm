"""JSON pointers and the structural diff.

`diff_paths` is the assertion that the two branches of a counterfactual pair are
identical outside the decisive field. If it misses a difference, a spurious cue
survives into the corpus and the pair stops measuring state-grounding.
"""

from __future__ import annotations

import pytest

from quick_slm_trainer.sft.pointer import (
  PointerError,
  diff_paths,
  escape,
  exists,
  get,
  parse,
  unescape,
  with_value,
)

DOC = {
  "buildings": [{"id": "b_004", "paused": False, "progress": 0.62}],
  "grid": {"generation": 100, "consumption": 60},
}


# --------------------------------------------------------------------------
# parse / escape
# --------------------------------------------------------------------------
def test_the_empty_pointer_is_the_whole_document():
  assert parse("") == []


def test_a_pointer_must_start_with_a_slash():
  with pytest.raises(PointerError, match="start with"):
    parse("buildings/0")


def test_escaping_round_trips():
  for token in ("a/b", "a~b", "~01", "plain", "a~1b"):
    assert unescape(escape(token)) == token


def test_tilde_one_is_unescaped_before_tilde_zero():
  # The other order turns `~01` into `/`, per RFC 6901.
  assert parse("/~01") == ["~1"]


# --------------------------------------------------------------------------
# get
# --------------------------------------------------------------------------
def test_get_walks_dicts_and_lists():
  assert get(DOC, "/buildings/0/id") == "b_004"
  assert get(DOC, "/grid/consumption") == 60
  assert get(DOC, "") == DOC


def test_get_raises_on_an_unresolved_pointer():
  with pytest.raises(PointerError, match="does not resolve"):
    get(DOC, "/buildings/0/nope")


def test_get_returns_the_default_instead_of_raising():
  assert get(DOC, "/nope", default=None) is None
  assert get(DOC, "/buildings/9/id", default="fallback") == "fallback"


def test_get_refuses_to_index_a_list_with_a_word():
  assert get(DOC, "/buildings/first", default=None) is None


def test_exists():
  assert exists(DOC, "/buildings/0/paused")
  assert not exists(DOC, "/buildings/0/absent")


# --------------------------------------------------------------------------
# with_value
# --------------------------------------------------------------------------
def test_with_value_returns_a_copy_and_leaves_the_original_alone():
  out = with_value(DOC, "/buildings/0/paused", True)
  assert out["buildings"][0]["paused"] is True
  assert DOC["buildings"][0]["paused"] is False


def test_with_value_writes_into_a_list_element():
  out = with_value(DOC, "/buildings/0/progress", 0.1)
  assert out["buildings"][0]["progress"] == 0.1


def test_with_value_refuses_to_create_a_missing_key():
  # A typo'd decisive_path would otherwise add a field no oracle reads, and the
  # pair would be rejected downstream for the wrong reason while the typo lived.
  with pytest.raises(PointerError, match="refusing to create"):
    with_value(DOC, "/buildings/0/pausd", True)


def test_with_value_refuses_the_document_root():
  with pytest.raises(PointerError, match="root"):
    with_value(DOC, "", {})


def test_with_value_rejects_an_out_of_range_index():
  with pytest.raises(PointerError, match="out of range"):
    with_value(DOC, "/buildings/7", {})


def test_with_value_rejects_a_path_through_a_missing_element():
  with pytest.raises(PointerError, match="does not resolve"):
    with_value(DOC, "/buildings/7/id", "x")


def test_with_value_accepts_none_as_a_value():
  # `research_start` flips `/research/active` between None and a tech id.
  doc = {"research": {"active": "smelting_2"}}
  assert with_value(doc, "/research/active", None)["research"]["active"] is None


# --------------------------------------------------------------------------
# diff_paths
# --------------------------------------------------------------------------
def test_identical_documents_differ_nowhere():
  assert diff_paths(DOC, {**DOC}) == set()


def test_a_single_flip_reports_a_single_pointer():
  assert diff_paths(DOC, with_value(DOC, "/grid/consumption", 140)) == {"/grid/consumption"}


def test_a_flip_inside_a_list_reports_the_indexed_pointer():
  assert diff_paths(DOC, with_value(DOC, "/buildings/0/paused", True)) == {"/buildings/0/paused"}


def test_two_flips_report_two_pointers():
  other = with_value(with_value(DOC, "/grid/consumption", 1), "/buildings/0/id", "b_999")
  assert diff_paths(DOC, other) == {"/grid/consumption", "/buildings/0/id"}


def test_a_key_present_on_one_side_only_is_reported():
  assert diff_paths({"a": 1}, {"a": 1, "b": 2}) == {"/b"}
  assert diff_paths({"a": 1, "b": 2}, {"a": 1}) == {"/b"}


def test_lists_of_different_lengths_differ_at_the_list():
  assert diff_paths({"xs": [1, 2]}, {"xs": [1, 2, 3]}) == {"/xs"}


def test_true_does_not_equal_one():
  # `True == 1` in Python. A boolean decisive field flipped against an integer
  # variant would otherwise read as unchanged.
  assert diff_paths({"on": True}, {"on": 1}) == {"/on"}
  assert diff_paths({"on": False}, {"on": 0}) == {"/on"}


def test_none_differs_from_a_string():
  assert diff_paths({"active": None}, {"active": "x"}) == {"/active"}


def test_a_dict_differs_from_a_list_at_that_pointer():
  assert diff_paths({"x": {}}, {"x": []}) == {"/x"}


def test_nested_equality_descends_all_the_way():
  a = {"p": {"q": [{"r": 1}]}}
  b = {"p": {"q": [{"r": 2}]}}
  assert diff_paths(a, b) == {"/p/q/0/r"}


def test_a_slash_in_a_key_is_escaped_in_the_reported_pointer():
  assert diff_paths({"a/b": 1}, {"a/b": 2}) == {"/a~1b"}
