"""Tests for the metrics added to answer the 'weak control / no noise floor / narrow metrics' critique."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rageblate import compare, judge, metrics, rag, report
from rageblate import embed as embed_mod
from rageblate.ablate import run
from rageblate.config import load_snapshot_config
from rageblate.llm import OpenAILLM
from rageblate.snapshot import resolve_snapshot
from rageblate.topics import topic_mask

# ------------------------------------------------------------------ helpers


def test_bootstrap_ci():
    m, lo, hi = metrics.bootstrap_ci([1.0] * 10)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    m, lo, hi = metrics.bootstrap_ci([0, 1, 0, 1, 0, 1, 0, 1])
    assert m == 0.5 and lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert metrics.bootstrap_ci([0.3, 0.7], seed=1) == metrics.bootstrap_ci([0.3, 0.7], seed=1)  # seeded
    one = metrics.bootstrap_ci([0.4])
    assert one[0] == 0.4 and np.isnan(one[1]) and np.isnan(one[2])
    assert all(np.isnan(x) for x in metrics.bootstrap_ci([np.nan, np.nan]))
    assert metrics.bootstrap_ci([1.0, np.nan, 3.0])[0] == 2.0


def test_primary_handles_old_results_without_sample_column():
    old = pd.DataFrame({"snapshot": ["a", "b"], "question_id": ["q", "q"]})
    assert len(metrics.primary(old)) == 2
    new = old.assign(sample=[0, 1])
    assert metrics.primary(new)["snapshot"].tolist() == ["a"]


def test_scale_drift_ruler():
    assert metrics.scale_drift(0.1, 0.1, 0.5) == 0.0
    assert metrics.scale_drift(0.5, 0.1, 0.5) == 1.0
    assert metrics.scale_drift(0.3, 0.1, 0.5) == pytest.approx(0.5)
    assert metrics.scale_drift(0.0, 0.1, 0.5) < 0  # quieter than noise is allowed to show
    assert np.isnan(metrics.scale_drift(0.3, 0.5, 0.5)) and np.isnan(metrics.scale_drift(0.3, 0.5, 0.1))


def test_answer_change_covers_flips_and_double_abstentions(embedder):
    def row(snap, q, abstained, answer):
        return {"snapshot": snap, "question_id": q, "question_topic": "t", "k": 1, "retrieved_ids": ["a"],
                "retrieved_years": [2000], "answer": answer, "abstained": abstained, "cited_ids": [], "invalid_cited_ids": []}

    res = pd.DataFrame([
        row("full", "answered_then_flipped", False, "alpha beta"), row("rm", "answered_then_flipped", True, "INSUFFICIENT_EVIDENCE"),
        row("full", "abstained_both", True, "INSUFFICIENT_EVIDENCE"), row("rm", "abstained_both", True, "INSUFFICIENT_EVIDENCE"),
        row("full", "same", False, "gamma delta"), row("rm", "same", False, "gamma delta"),
    ])
    out = metrics.answer_drift(res, "full", lambda t: embedder.encode(t)).set_index(["snapshot", "question_id"])
    assert out.loc[("rm", "answered_then_flipped"), "answer_change"] == 1.0
    assert out.loc[("rm", "abstained_both"), "answer_change"] == 0.0
    assert out.loc[("rm", "same"), "answer_change"] == pytest.approx(0.0, abs=1e-5)
    assert np.isnan(out.loc[("rm", "answered_then_flipped"), "answer_drift"])  # old metric still skips it


# ------------------------------------------------------------------ noise floor


def _draws(rows):
    return pd.DataFrame([
        {"snapshot": "full", "question_id": q, "question_topic": "t", "sample": s, "abstained": ab, "answer": txt,
         "cited_ids": cites, "retrieved_ids": ["a", "b"]}
        for q, s, ab, txt, cites in rows
    ])


def test_noise_floor_counts_pairs_flips_and_drift(embedder):
    res = _draws([
        ("q1", 0, False, "ocean heat transport", ["a"]), ("q1", 1, False, "ocean heat transport", ["a"]),
        ("q2", 0, False, "sea ice melt", ["a"]), ("q2", 1, True, "INSUFFICIENT_EVIDENCE", []), ("q2", 2, False, "sea ice melt", ["a"]),
    ])
    per_q, summ = metrics.noise_floor(res, "full", lambda t: embedder.encode(t))
    per_q = per_q.set_index("question_id")
    assert per_q.loc["q1", "n_pairs"] == 1 and per_q.loc["q1", "abstain_disagreement"] == 0 and per_q.loc["q1", "mean_drift"] == pytest.approx(0, abs=1e-5)
    assert per_q.loc["q2", "n_pairs"] == 3 and per_q.loc["q2", "abstain_disagreement"] == pytest.approx(2 / 3)
    assert per_q.loc["q2", "mean_change"] == pytest.approx((1 + 1 + 0) / 3, abs=1e-5)  # two flips, one identical pair
    assert summ["n_questions"] == 2 and summ["abstain_disagreement"] == pytest.approx(1 / 3)


def test_noise_floor_needs_replicates(embedder):
    assert metrics.noise_floor(_draws([("q1", 0, False, "x", [])]), "full", lambda t: embedder.encode(t))[1] == {}
    assert metrics.noise_floor(_draws([("q1", 0, False, "x", [])]).drop(columns="sample"), "full", None)[1] == {}


def test_replicate_checks_flag_diverging_retrieval():
    res = _draws([("q1", 0, False, "x", []), ("q1", 1, False, "y", [])])
    assert metrics.replicate_checks(res)[0][1] is True
    res.at[1, "retrieved_ids"] = ["a", "ZZ"]
    assert metrics.replicate_checks(res)[0][1] is False
    assert metrics.replicate_checks(res.drop(columns="sample")) == []


# ------------------------------------------------------------------ the harder control


def test_pool_all_control_can_hit_the_topic_and_validates(ctx, topics, make_cfg, tmp_path):
    in_topic = topic_mask(ctx.df, topics["sea_ice"])
    removal = resolve_snapshot(ctx.df, make_cfg(exclude={"topic": "sea_ice"}), topics)
    cfg = make_cfg(exclude={"random": {"matched_to_topic": "sea_ice", "pool": "all", "seed": 3}})
    c = resolve_snapshot(ctx.df, cfg, topics)
    assert c.n_docs == removal.n_docs  # same number removed
    assert in_topic[c.rows].sum() < in_topic.sum()  # ...but topic documents are fair game
    assert c.cfg_hash != resolve_snapshot(ctx.df, make_cfg(exclude={"random": {"matched_to_topic": "sea_ice", "seed": 3}}), topics).cfg_hash

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nexclude: {random: {matched_to_topic: sea_ice, pool: nowhere}}\n")
    with pytest.raises(ValueError, match="pool"):
        load_snapshot_config(str(bad))
    bad.write_text("name: bad\nexclude: {random: {n: 5, pool: all}}\n")
    with pytest.raises(ValueError, match="matched_to_topic"):
        load_snapshot_config(str(bad))


def test_snapshot_kind():
    k = metrics.snapshot_kind
    assert k({"exclude": {"topic": "x"}}) == "topic_removal"
    assert k({"exclude": {"random": {"matched_to_topic": "x"}}}) == "control_outside"
    assert k({"exclude": {"random": {"matched_to_topic": "x", "pool": "all"}}}) == "control_all"
    assert k({"exclude": {"random": {"n": 3}}}) == "random"
    assert k({"include": {"year_max": 2010}, "exclude": {}}) == "time_slice"
    assert k({"include": {}, "exclude": {}}) == "other"


# ------------------------------------------------------------------ end to end: replicates + report + judge + compare


@pytest.fixture
def e2e(ctx, topics, make_cfg, questions, embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(report, "load_corpus", lambda columns=None: ctx.df[["arxiv_id"]] if columns else ctx.df)
    cfgs = [
        make_cfg("full"),
        make_cfg("upto_2010", include={"year_max": 2010}),
        make_cfg("remove_sea_ice", exclude={"topic": "sea_ice"}),
        make_cfg("control_sea_ice", exclude={"random": {"matched_to_topic": "sea_ice", "seed": 0}}),
        make_cfg("control_all_sea_ice_s1", exclude={"random": {"matched_to_topic": "sea_ice", "pool": "all", "seed": 1}}),
        make_cfg("control_all_sea_ice_s2", exclude={"random": {"matched_to_topic": "sea_ice", "pool": "all", "seed": 2}}),
        make_cfg("remove_earthquakes", exclude={"topic": "earthquakes"}),
        make_cfg("control_earthquakes", exclude={"random": {"matched_to_topic": "earthquakes", "seed": 0}}),
    ]
    snaps = [resolve_snapshot(ctx.df, c, topics) for c in cfgs]
    qvecs = embedder.encode([q["question"] for q in questions], kind="query")
    out = tmp_path / "run"
    res = run(ctx, snaps, questions, qvecs, rag.ExtractiveAnswerer(0.5), k=5, backend="numpy", out_dir=out,
              meta_extra={"run_id": "test", "reference_snapshot": "full"}, repeats=3, repeat_snapshots={"full"})
    return SimpleNamespace(out=out, res=res, snaps=snaps, ctx=ctx, questions=questions)


def test_run_records_repeats_in_results_and_meta(e2e):
    meta = json.loads((e2e.out / "run.json").read_text())
    assert meta["repeats"] == 3 and meta["repeat_snapshots"] == ["full"] and meta["prompt_version"] == "v1"
    assert set(e2e.res[e2e.res.snapshot == "full"]["sample"]) == {0, 1, 2}
    assert set(e2e.res[e2e.res.snapshot != "full"]["sample"]) == {0}


def test_report_has_new_sections_and_files(e2e):
    text = report.build_report(e2e.out, with_drift=True).read_text()
    for needle in ("Headline: topic removal vs size-matched controls", "Evidence loss", "Noise floor", "scaled_drift_affected",
                   "5. Answer quality", "Not judged", "replicate draws share identical retrieval"):
        assert needle in text, needle
    assert "FAIL" not in text
    m = e2e.out / "metrics"
    for f in ("topic_effect.csv", "dose_response.csv", "noise_floor.csv", "citations.csv", "topic_effect.png", "dose_response.png"):
        assert (m / f).exists(), f
    eff = pd.read_csv(m / "topic_effect.csv")
    assert {"outside", "all"} <= set(eff["control"]) and "ALL TOPICS" in set(eff["topic"])
    row = eff[(eff.topic == "sea_ice") & (eff.control == "all")].iloc[0]
    assert row["n_control_snapshots"] == 2  # two seeds averaged
    assert row["jaccard_gap"] == pytest.approx(row["jaccard_control"] - row["jaccard_removed"])  # orientation: control - removed
    assert row["abstain_gap"] == pytest.approx(row["abstain_removed"] - row["abstain_control"])  # orientation: removed - control
    assert np.isnan(row["jaccard_gap_lo"])  # one question for this topic in the fixture: no interval, said honestly
    pooled = eff[(eff.topic == "ALL TOPICS") & (eff.control == "outside")].iloc[0]  # sea_ice + earthquakes
    assert pooled["n_questions"] == 2 and pooled["jaccard_gap_lo"] <= pooled["jaccard_gap"] <= pooled["jaccard_gap_hi"]


def test_report_banner_without_replicates(e2e, ctx, topics, make_cfg, questions, embedder, tmp_path, monkeypatch):
    snaps = [resolve_snapshot(ctx.df, make_cfg("full"), topics), resolve_snapshot(ctx.df, make_cfg("rm", exclude={"topic": "sea_ice"}), topics)]
    qv = embedder.encode([q["question"] for q in questions], kind="query")
    out = tmp_path / "single"
    run(ctx, snaps, questions, qv, rag.ExtractiveAnswerer(0.5), k=5, backend="numpy", out_dir=out, meta_extra={"run_id": "s", "reference_snapshot": "full"})
    text = report.build_report(out).read_text()
    assert "No noise floor measured" in text and "Small sample" in text  # one question per topic in the fixture
    assert "Headline" not in text  # no control snapshots -> no contrast, and no crash


def test_report_reads_results_from_before_replicates_existed(e2e):
    res = pd.read_parquet(e2e.out / "results.parquet")
    res = res[res["sample"] == 0].drop(columns="sample")
    res.to_parquet(e2e.out / "results.parquet", index=False)
    meta = json.loads((e2e.out / "run.json").read_text())
    meta.pop("repeats"), meta.pop("repeat_snapshots")
    (e2e.out / "run.json").write_text(json.dumps(meta))
    assert "No noise floor measured" in report.build_report(e2e.out).read_text()


def test_dose_response_bins_by_evidence_lost(e2e):
    meta, res, ids = report.load_run(e2e.out)
    df = metrics.attach_reference(metrics.primary(res), "full", ids)
    df["abstained"] = df["abstained"].astype(bool)
    tab = metrics.dose_response(df, meta["snapshots"], "full")
    assert {"evidence_lost", "kind", "n", "abstain_rate"} <= set(tab.columns)
    assert tab["n"].sum() == (df["snapshot"] != "full").sum()
    assert "topic_removal" in set(tab["kind"]) and "control_all" in set(tab["kind"])


# ------------------------------------------------------------------ judge


class _JudgeResp:
    status = "completed"
    model = "judge-dated"

    def __init__(self, text):
        self.output_text = text
        self.usage = SimpleNamespace(input_tokens=500, output_tokens=40, output_tokens_details=None)


class _JudgeClient:
    def __init__(self):
        self.calls, self.responses = 0, self

    def create(self, **kw):
        self.calls += 1
        if "abstentions" in kw["instructions"]:
            return _JudgeResp('```json\n{"verdict": "justified", "reason": "no relevant abstract"}\n```')
        return _JudgeResp('{"verdict": "supported", "reason": "stated in [0704.0001]"}')


def test_parse_verdict_is_tolerant_and_never_raises():
    assert judge.parse_verdict('{"verdict": "Partial", "reason": "r"}', "support") == ("partial", "r")
    assert judge.parse_verdict('noise {"verdict": "justified", "reason": "r"} noise', "abstain")[0] == "justified"
    assert judge.parse_verdict('{"verdict": "supported"}', "abstain")[0] == "unparsed"  # wrong vocabulary for the kind
    assert judge.parse_verdict("not json", "support")[0] == "unparsed"
    assert judge.parse_verdict('{"verdict": ', "support")[0] == "unparsed"


def test_judge_pipeline_end_to_end(e2e, tmp_path, monkeypatch):
    _, res, _ = report.load_run(e2e.out)
    qtext = {q["id"]: q["question"] for q in e2e.questions}
    items = judge.judge_items(res, e2e.ctx.df, qtext, None)
    n_primary = len(metrics.primary(res))
    assert len(items) == n_primary and {i[2] for i in items} <= {"support", "abstain"}

    client = _JudgeClient()
    llm = OpenAILLM("judge-model", cache_dir=tmp_path / "jc", client=client)
    out = judge.run_judge(llm, items, plan_only=True)
    assert out is None and client.calls == 0  # --plan-only spends nothing
    with pytest.raises(SystemExit, match="paid judge calls"):
        judge.run_judge(llm, items, confirm_above=0)
    j = judge.run_judge(llm, items, yes=True, workers=2)
    assert len(j) == len(items) and (j["verdict"] != "unparsed").all()
    assert set(j.loc[j.kind == "abstain", "verdict"]) <= {"justified"} and set(j.loc[j.kind == "support", "verdict"]) <= {"supported"}
    calls_after = client.calls
    judge.run_judge(llm, items, yes=True)  # everything cached now
    assert client.calls == calls_after

    j.to_parquet(e2e.out / "judge.parquet", index=False)
    text = report.build_report(e2e.out).read_text()
    assert "LLM judge" in text and "unvalidated" in text and "Not judged" not in text
    assert (e2e.out / "metrics" / "judge_by_kind.csv").exists()

    n = judge.export_labels(j, items, 6, e2e.out / "judge_labels.csv")
    lab = pd.read_csv(e2e.out / "judge_labels.csv")
    assert n == len(lab) > 0 and (lab["your_verdict"].isna() | (lab["your_verdict"] == "")).all()
    lab["your_verdict"] = lab["judge_verdict"]
    lab.at[0, "your_verdict"] = "unsupported" if lab.at[0, "kind"] == "support" else "unjustified"
    lab.to_csv(e2e.out / "judge_labels.csv", index=False)
    sc = judge.score_labels(e2e.out / "judge_labels.csv")
    assert sc["n"] == n and sc["agreement"] == pytest.approx((n - 1) / n)


def test_cohens_kappa():
    assert judge.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert judge.cohens_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == -1.0
    assert np.isnan(judge.cohens_kappa([], []))
    assert np.isnan(judge.cohens_kappa(["a", "a"], ["a", "a"]))  # no variation: undefined, not 1


# ------------------------------------------------------------------ prompt v2 and compare-runs


def test_prompt_v2_has_its_own_sentinel_and_cache(tmp_path):
    assert rag.ABSTAIN_V2 in rag.SYSTEM_PROMPT_V2 and rag.ABSTAIN not in rag.SYSTEM_PROMPT_V2
    assert rag.is_abstain("NO_SUPPORTING_EVIDENCE\nnothing", rag.ABSTAIN_V2)
    assert not rag.is_abstain("INSUFFICIENT_EVIDENCE\nnothing", rag.ABSTAIN_V2)

    class C:
        def __init__(self):
            self.responses = self

        def create(self, **kw):
            return _JudgeResp("NO_SUPPORTING_EVIDENCE\nmissing")

    llm = OpenAILLM("m", cache_dir=tmp_path, client=C())
    v1, v2 = rag.OpenAIAnswerer(llm, "v1"), rag.OpenAIAnswerer(llm, "v2")
    assert v1.name.endswith(":v1") and v2.name.endswith(":v2")
    hits = [rag.Hit(1, 1, "0704.0001", 2010, 0.9, "t", "a")]
    assert v1.prompt("q", hits)[0] != v2.prompt("q", hits)[0]
    assert v2.answer("q", hits).abstained and not v1.answer("q", hits).abstained  # v1 does not know v2's sentinel
    with pytest.raises(ValueError, match="unknown prompt version"):
        rag.OpenAIAnswerer(llm, "v9")


def test_compare_flags_sign_and_interval_overlap():
    def tab(gap, lo, hi):
        return pd.DataFrame([{"topic": "enso", "control": "all", "abstain_gap": gap, "abstain_gap_lo": lo, "abstain_gap_hi": hi}])

    same = compare.compare(tab(0.5, 0.2, 0.8), tab(0.4, 0.1, 0.7)).iloc[0]
    assert bool(same["same_sign"]) and bool(same["intervals_overlap"])
    flip = compare.compare(tab(0.5, 0.4, 0.6), tab(-0.5, -0.6, -0.4)).iloc[0]
    assert not bool(flip["same_sign"]) and not bool(flip["intervals_overlap"])
    assert compare.compare(pd.DataFrame(), tab(0.1, 0, 0.2)).empty


def test_compare_runs_on_two_real_runs(e2e):
    t, info = compare.effect_for(e2e.out)
    assert info["n_questions"] == len(e2e.questions) and not t.empty
    out = compare.compare(t, t, "jaccard_gap")
    assert out["same_sign"].dropna().all() and out["intervals_overlap"].all()
