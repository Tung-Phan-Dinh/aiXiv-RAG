import json

import pandas as pd
import pytest

from rageblate import ingest


def test_has_target_exact_token_match():
    assert ingest.has_target("physics.ao-ph")
    assert ingest.has_target("cs.LG physics.geo-ph")
    assert not ingest.has_target("astro-ph.EP")
    assert not ingest.has_target("physics.ao-ph.extra")  # substring match would wrongly accept this
    assert not ingest.has_target("")


def test_first_year_takes_earliest_even_if_unordered():
    versions = [
        {"version": "v2", "created": "Tue, 24 Jul 2008 20:10:27 GMT"},
        {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
    ]
    assert ingest.first_year(versions) == 2007


def test_first_year_empty_raises():
    with pytest.raises(ValueError):
        ingest.first_year([])


def test_clean_text_collapses_whitespace_but_keeps_math():
    raw = "  We describe the $(k,\\ell)$-pebble game\nwith  colors.\n"
    assert ingest.clean_text(raw) == "We describe the $(k,\\ell)$-pebble game with colors."


def _line(**over):
    rec = {
        "id": "0704.0001",
        "title": "A\n  title",
        "abstract": "  Some\nabstract.\n",
        "categories": "physics.ao-ph cs.LG",
        "license": None,
        "update_date": "2008-11-26",
        "versions": [{"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}],
    }
    rec.update(over)
    return json.dumps(rec)


def test_parse_record_keeps_and_cleans():
    row, reason = ingest.parse_record(_line())
    assert reason is None
    assert row["title"] == "A title" and row["abstract"] == "Some abstract."
    assert row["year"] == 2007 and row["update_date"] == "2008-11-26"


@pytest.mark.parametrize(
    "line, reason",
    [
        ("{not json", "bad_json"),
        (_line(categories="hep-ph"), "not_target"),
        (_line(versions=[]), "bad_version"),
        (_line(versions=None), "bad_version"),
        (_line(abstract="   \n "), "empty_abstract"),
    ],
)
def test_parse_record_skip_reasons(line, reason):
    row, why = ingest.parse_record(line)
    assert row is None and why == reason


def test_dedup_latest_keeps_latest_update_date_and_breaks_ties_by_file_order():
    df = pd.DataFrame(
        [
            {"arxiv_id": "a", "update_date": "2008-01-01", "categories": "old"},
            {"arxiv_id": "b", "update_date": "2009-01-01", "categories": "only"},
            {"arxiv_id": "a", "update_date": "2010-01-01", "categories": "newer"},
            {"arxiv_id": "c", "update_date": "2009-01-01", "categories": "first"},
            {"arxiv_id": "c", "update_date": "2009-01-01", "categories": "second"},
        ]
    )
    out, dropped = ingest.dedup_latest(df)
    assert dropped == 2 and out["arxiv_id"].is_unique
    got = dict(zip(out["arxiv_id"], out["categories"]))
    assert got == {"a": "newer", "b": "only", "c": "second"}
