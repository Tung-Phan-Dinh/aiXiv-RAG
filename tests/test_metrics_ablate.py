import json

import numpy as np
import pandas as pd
import pytest

from rageblate import embed as embed_mod
from rageblate import metrics, report
from rageblate.ablate import run
from rageblate.rag import ExtractiveAnswerer
from rageblate.snapshot import resolve_snapshot


def test_jaccard():
    assert metrics.jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert metrics.jaccard([1, 2], [3, 4]) == 0.0
    assert metrics.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert metrics.jaccard([], []) == 1.0


def _toy():
    def row(snap, q, ids, years, abstained=False, cited=(), invalid=(), topic="t1", answer="x"):
        return {"snapshot": snap, "question_id": q, "question_topic": topic, "k": len(ids), "retrieved_ids": ids,
                "retrieved_years": years, "answer": answer, "abstained": abstained, "cited_ids": list(cited),
                "invalid_cited_ids": list(invalid)}

    res = pd.DataFrame([
        row("full", "q1", ["a", "b", "c"], [2001, 2005, 2010], answer="alpha"),
        row("full", "q2", ["d", "e", "f"], [2011, 2012, 2013], topic=None, answer="beta"),
        row("rm", "q1", ["x", "y", "c"], [2002, 2003, 2010], abstained=True, answer="INSUFFICIENT_EVIDENCE"),
        row("rm", "q2", ["d", "e", "f"], [2011, 2012, 2013], topic=None, cited=["d"], invalid=["zzz"], answer="beta"),
    ])
    ids = {"full": set("abcdef") | {"x", "y"}, "rm": {"c", "d", "e", "f", "x", "y"}}
    return res, ids


def test_stability_and_monotonicity():
    res, ids = _toy()
    df = metrics.attach_reference(res, "full", ids)
    q1 = df[(df.snapshot == "rm") & (df.question_id == "q1")].iloc[0]
    q2 = df[(df.snapshot == "rm") & (df.question_id == "q2")].iloc[0]
    assert q1.ref_missing == 2 and q1.affected and q1.jaccard == pytest.approx(1 / 5)
    assert q2.ref_missing == 0 and not q2.affected and q2.jaccard == 1.0
    tab = metrics.stability_table(df, "full").iloc[0]
    assert tab.monotonicity_violations == 0 and tab.share_affected == 0.5
    assert all(ok for _, ok, _ in metrics.sanity_checks(df, "full", ids))


def test_sanity_checks_catch_leaks_and_violations():
    res, ids = _toy()
    res.at[3, "retrieved_ids"] = ["d", "e", "ZZ"]  # row 3 = (rm, q2): a leaked doc, and a changed top-k
    df = metrics.attach_reference(res, "full", ids)
    failed = {name for name, ok, _ in metrics.sanity_checks(df, "full", ids) if not ok}
    assert "every retrieved doc is inside its snapshot" in failed
    assert "monotonicity: unaffected questions keep an identical top-k" in failed


def test_temporal_profile_and_histogram():
    res, _ = _toy()
    prof = metrics.temporal_profile(res).set_index("snapshot")
    assert prof.loc["full", "median_year"] == 2010.5
    hist = metrics.year_histogram(res)
    assert hist.loc["full"].sum() == pytest.approx(1.0) and hist.loc["rm", 2003] > 0


def test_abstention_table():
    res, _ = _toy()
    tab = metrics.abstention_table(res, {"rm": ["t1"]}).set_index("snapshot")
    assert tab.loc["rm", "n_targeted"] == 1 and tab.loc["rm", "abstain_rate_targeted"] == 1.0
    assert tab.loc["rm", "abstain_rate_nontargeted"] == 0.0 and tab.loc["rm", "invalid_citation_rate"] == 0.5
    assert tab.loc["full", "n_targeted"] == 0


def test_answer_drift(embedder):
    res, _ = _toy()
    out = metrics.answer_drift(res, "full", lambda texts: embedder.encode(texts))
    rm = out[out.snapshot == "rm"].set_index("question_id")
    assert np.isnan(rm.loc["q1", "answer_drift"]) and bool(rm.loc["q1", "flipped"])  # abstained: not comparable
    assert rm.loc["q2", "answer_drift"] == pytest.approx(0.0, abs=1e-5)  # identical answer
    assert (out[out.snapshot == "full"]["answer_drift"].dropna() < 1e-5).all()


def test_end_to_end_run_and_report(ctx, topics, make_cfg, questions, embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(report, "load_corpus", lambda columns=None: ctx.df[["arxiv_id"]])
    cfgs = [
        make_cfg("full"),
        make_cfg("upto_2010", include={"year_max": 2010}),
        make_cfg("remove_sea_ice", exclude={"topic": "sea_ice"}),
        make_cfg("control_sea_ice", exclude={"random": {"matched_to_topic": "sea_ice", "seed": 0}}),
    ]
    snaps = [resolve_snapshot(ctx.df, c, topics) for c in cfgs]
    qvecs = embedder.encode([q["question"] for q in questions], kind="query")
    out = tmp_path / "run"
    k = 5
    res = run(ctx, snaps, questions, qvecs, ExtractiveAnswerer(0.0), k=k, backend="numpy", out_dir=out,
              meta_extra={"run_id": "test", "reference_snapshot": "full"})

    assert len(res) == len(snaps) * len(questions) and (res["k"] == k).all()
    meta = json.loads((out / "run.json").read_text())
    assert set(meta["snapshots"]) == {c["name"] for c in cfgs} and meta["snapshots"]["remove_sea_ice"]["removed_topics"] == ["sea_ice"]
    assert (out / "results.parquet").exists() and (out / "snapshot_rows" / "full.npy").exists()

    # deterministic: a second run gives the identical table
    res2 = run(ctx, snaps, questions, qvecs, ExtractiveAnswerer(0.0), k=k, backend="numpy")
    pd.testing.assert_frame_equal(res, res2)

    path = report.build_report(out, with_drift=True)
    text = path.read_text()
    assert "FAIL" not in text and "PASS" in text
    assert (out / "metrics" / "stability.png").exists() and (out / "metrics" / "drift.csv").exists()
    stab = pd.read_csv(out / "metrics" / "stability.csv").set_index("snapshot")
    assert stab.loc["remove_sea_ice", "jaccard_affected"] < 1.0  # removing the topic changes retrieval
    assert stab["monotonicity_violations"].sum() == 0
    # the experiment's core contrast: removing the topic changes the sea-ice question's top-k,
    # while the size-matched random control (which never touches topic docs) leaves it identical
    _, res_back, sids = report.load_run(out)
    dfm = metrics.attach_reference(res_back, "full", sids)
    tq = dfm[dfm.question_topic == "sea_ice"].set_index("snapshot")
    assert tq.loc["remove_sea_ice", "jaccard"] < 1.0
    assert tq.loc["control_sea_ice", "jaccard"] == 1.0


def test_ask_answers_from_the_snapshot_it_was_given(ctx, topics, make_cfg, embedder):
    from rageblate.ask import answer_question
    from rageblate.index import build_index

    snap = resolve_snapshot(ctx.df, make_cfg("upto_2010", include={"year_max": 2010}), topics)
    question = "arctic sea ice melt"
    qvec = embedder.encode([question], kind="query")[0]
    hits, ans = answer_question(ctx, build_index(ctx, snap, "numpy"), question, qvec, ExtractiveAnswerer(0.0), k=5)

    assert len(hits) == 5 and all(h.year <= 2010 for h in hits)
    assert not ans.abstained and ans.cited_ids and not ans.invalid_cited_ids
