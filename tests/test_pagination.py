from __future__ import annotations

from papercut.models.baselines.tfidf_xgb_layout import _pagination, _pagination_pair_features


def test_reads_separated_and_worded_forms() -> None:
    assert _pagination("Rechnung 1/2 Vodafone") == (True, 1, 2)
    assert _pagination("Seite 3 von 12 unten") == (True, 3, 12)
    assert _pagination("page 2 of 2") == (True, 2, 2)
    assert _pagination("Blatt 1 - 4 Anlage") == (True, 1, 4)


def test_rejects_pairs_that_cannot_be_pagination() -> None:
    assert _pagination("no numbers here") == (False, 0, 0)
    assert _pagination("Betrag 250 von 100 Euro") == (False, 0, 0)
    assert _pagination("Kundennummer 1234567 von 9999999") == (False, 0, 0)


def test_only_head_and_foot_are_read() -> None:
    body = "x" * 2000
    assert _pagination(f"head{body}1 von 2{body}tail", window=50) == (False, 0, 0)


def test_last_page_followed_by_a_first_page_is_marked() -> None:
    features = _pagination_pair_features("Seite 2 von 2", "Seite 1 von 2")
    assert features[2] == 1.0  # same total
    assert features[4] == 1.0  # count restarts
    assert features[5] == 1.0  # previous page closed its document


def test_continuation_inside_one_document_is_marked() -> None:
    features = _pagination_pair_features("Seite 1 von 3", "Seite 2 von 3")
    assert features[3] == 1.0  # consecutive
    assert features[4] == 0.0
    assert features[5] == 0.0


def test_missing_pagination_yields_zeros_beyond_the_flags() -> None:
    assert _pagination_pair_features("nothing", "Seite 1 von 2") == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
