import numpy as np
import pytest

from rageblate.config import list_snapshot_names, load_questions, load_snapshot_config, load_topics
from rageblate.snapshot import resolve_snapshot
from rageblate.topics import topic_mask


def test_shipped_configs_are_valid():
    topics = load_topics()
    for name in list_snapshot_names():
        cfg = load_snapshot_config(name)
        for t in [cfg["exclude"].get("topic"), (cfg["exclude"].get("random") or {}).get("matched_to_topic")]:
            for tt in ([t] if isinstance(t, str) else (t or [])):
                assert tt in topics, f"{name} references unknown topic {tt}"
    topic_names = set(topics)
    for q in load_questions():
        assert q["topic"] is None or q["topic"] in topic_names


def test_real_topic_rules_match_expected_spellings():
    topics = load_topics()
    import pandas as pd

    def hit(topic, text):
        df = pd.DataFrame({"title": [text], "abstract": [""]})
        return bool(topic_mask(df, topics[topic])[0])

    assert hit("enso", "Impacts of El Niño on rainfall")
    assert hit("enso", "El Nino and La Nina")
    assert hit("enso", r"strong El Ni\~no events")
    assert hit("enso", "the ENSO cycle")
    assert not hit("enso", "a new sensor design")  # \bENSO\b must not match inside words
    assert hit("sea_ice", "Arctic Sea-Ice decline")
    assert not hit("sea_ice", "a nice device")


def test_topic_mask_is_case_insensitive_and_deterministic(ctx, topics):
    a, b = topic_mask(ctx.df, topics["sea_ice"]), topic_mask(ctx.df, topics["sea_ice"])
    assert (a == b).all() and a.sum() > 0


def test_year_cutoff(ctx, topics, make_cfg):
    snap = resolve_snapshot(ctx.df, make_cfg(include={"year_max": 2010}), topics)
    years = ctx.df["year"].to_numpy()[snap.rows]
    assert len(snap.rows) > 0 and years.max() <= 2010
    assert (np.diff(snap.rows) > 0).all()  # sorted, unique


def test_year_range_and_categories(ctx, topics, make_cfg):
    snap = resolve_snapshot(ctx.df, make_cfg(include={"year_min": 2005, "year_max": 2015, "categories_any": ["physics.geo-ph"]}), topics)
    sub = ctx.df.iloc[snap.rows]
    assert sub["year"].between(2005, 2015).all()
    assert sub["categories"].str.contains("physics.geo-ph").all()


def test_topic_removal_removes_exactly_the_topic(ctx, topics, make_cfg):
    full = resolve_snapshot(ctx.df, make_cfg(), topics)
    snap = resolve_snapshot(ctx.df, make_cfg(exclude={"topic": "sea_ice"}), topics)
    in_topic = topic_mask(ctx.df, topics["sea_ice"])
    assert full.n_docs == len(ctx.df)
    assert snap.n_docs == len(ctx.df) - in_topic.sum()
    assert not in_topic[snap.rows].any()


def test_random_control_matches_size_never_touches_topic_and_is_seeded(ctx, topics, make_cfg):
    in_topic = topic_mask(ctx.df, topics["sea_ice"])
    removal = resolve_snapshot(ctx.df, make_cfg(exclude={"topic": "sea_ice"}), topics)

    def control(seed):
        cfg = make_cfg(exclude={"random": {"matched_to_topic": "sea_ice", "seed": seed}})
        return resolve_snapshot(ctx.df, cfg, topics)

    c0, c0b, c1 = control(0), control(0), control(1)
    assert c0.n_docs == removal.n_docs  # same number removed
    assert in_topic[c0.rows].sum() == in_topic.sum()  # all topic docs survive
    assert (c0.rows == c0b.rows).all() and c0.ids_hash == c0b.ids_hash  # deterministic
    assert not (len(c0.rows) == len(c1.rows) and (c0.rows == c1.rows).all())  # seed matters


def test_cfg_hash_covers_topic_rules(ctx, make_cfg):
    cfg = make_cfg(exclude={"topic": "sea_ice"})
    a = resolve_snapshot(ctx.df, cfg, {"sea_ice": {"any_of": ["sea ice"]}})
    b = resolve_snapshot(ctx.df, cfg, {"sea_ice": {"any_of": ["sea ice", "glacier"]}})
    assert a.cfg_hash != b.cfg_hash


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ninclude: {year_mx: 2010}\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_snapshot_config(str(bad))
    bad.write_text("name: bad\nexclude: {random: {n: 3, frac: 0.1}}\n")
    with pytest.raises(ValueError, match="exactly one"):
        load_snapshot_config(str(bad))
