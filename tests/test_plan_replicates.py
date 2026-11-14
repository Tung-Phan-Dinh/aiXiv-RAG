import numpy as np
import pytest

from rageblate import plan as plan_mod
from rageblate import rag
from rageblate.ablate import iter_cells, retrieve_all, run, sample_count
from rageblate.audit import audit_questions
from rageblate.llm import OpenAILLM
from rageblate.retrieve import Hit
from rageblate.snapshot import resolve_snapshot
from rageblate.topics import topic_overlap


def hit(i, aid=None):
    return Hit(i, i, aid or f"0704.000{i}", 2010, 0.9, f"title {i}", f"First sentence {i}.")


class _Resp:
    status = "completed"
    model = "m-dated"

    def __init__(self, text):
        self.output_text = text
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 200, "output_tokens_details": None})()


class _Client:
    def __init__(self):
        self.calls = 0
        self.responses = self

    def create(self, **kw):
        self.calls += 1
        return _Resp(f"answer {self.calls} [0704.0001]")


# ------------------------------------------------------------------ replicate samples in the cache


def test_sample_zero_keeps_the_pre_replicate_key_and_replicates_differ(tmp_path):
    llm = OpenAILLM("m", cache_dir=tmp_path, client=_Client())
    import hashlib
    import json

    legacy = hashlib.sha256(
        json.dumps(
            {"provider": "openai", "model": "m", "system": "s", "user": "u", "max_output_tokens": 3000,
             "temperature": 0.0, "reasoning_effort": None},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert llm.key("s", "u") == llm.key("s", "u", 0) == legacy  # old caches stay valid
    assert len({llm.key("s", "u", i) for i in range(4)}) == 4


def test_replicates_are_separate_paid_calls_and_each_is_cached(tmp_path):
    client = _Client()
    llm = OpenAILLM("m", cache_dir=tmp_path, client=client)
    texts = [llm.complete("s", "u", i).text for i in (0, 1, 2)]
    assert len(set(texts)) == 3 and client.calls == 3
    assert llm.complete("s", "u", 1).cached and client.calls == 3
    assert llm.is_cached("s", "u", 2) and not llm.is_cached("s", "u", 3)


# ------------------------------------------------------------------ plan


def _answerer(tmp_path, client=None):
    return rag.OpenAIAnswerer(OpenAILLM("m", cache_dir=tmp_path, client=client or _Client()))


def test_plan_dedupes_identical_prompts_and_skips_cached(tmp_path):
    a = _answerer(tmp_path)
    h1, h2 = [hit(1)], [hit(2)]
    calls = [("q", h1, 0), ("q", h1, 0), ("q", h2, 0), ("q", h1, 1)]  # 4 cells, 3 distinct prompts
    p = plan_mod.build_plan(a, calls, {"m": {"input": 2.0, "output": 10.0}})
    assert (p.n_cells, p.n_unique, p.n_cached, p.n_new) == (4, 3, 0, 3)
    a.answer("q", h1)  # pay for one of them
    p2 = plan_mod.build_plan(a, calls, {"m": {"input": 2.0, "output": 10.0}})
    assert (p2.n_unique, p2.n_cached, p2.n_new) == (3, 1, 2)
    # estimate learns from the one cached response: 1000 in + 200 out tokens per call
    assert p2.est_input_tokens == 2000 and p2.est_output_tokens == 400
    assert p2.est_cost_usd == pytest.approx((2000 * 2.0 + 400 * 10.0) / 1e6)
    assert "1 cached, 2 to pay for" in p2.describe("m")


def test_plan_without_price_or_llm(tmp_path):
    a = _answerer(tmp_path)
    assert plan_mod.build_plan(a, [("q", [hit(1)], 0)], {}).est_cost_usd is None
    p = plan_mod.build_plan(rag.ExtractiveAnswerer(), [("q", [hit(1)], 0)] * 3)
    assert p.n_new == 0 and p.n_cells == 3


def test_execute_pays_for_every_job_concurrently_then_run_reads_cache(tmp_path):
    client = _Client()
    a = _answerer(tmp_path, client)
    calls = [("q", [hit(i)], 0) for i in range(1, 6)]
    p = plan_mod.build_plan(a, calls)
    plan_mod.execute(a.llm, p.jobs, workers=3)
    assert client.calls == 5 and plan_mod.build_plan(a, calls).n_new == 0


def test_execute_stops_and_raises_on_first_failure(tmp_path):
    class Boom(_Client):
        def create(self, **kw):
            raise RuntimeError("boom")

    a = _answerer(tmp_path, Boom())
    p = plan_mod.build_plan(a, [("q", [hit(i)], 0) for i in range(1, 4)])
    with pytest.raises(RuntimeError):
        plan_mod.execute(a.llm, p.jobs, workers=1)


# ------------------------------------------------------------------ repeats in run()


def test_run_repeats_only_on_requested_snapshots_and_paid_keys(ctx, topics, make_cfg, questions, embedder, tmp_path):
    full = resolve_snapshot(ctx.df, make_cfg("full"), topics)
    rm = resolve_snapshot(ctx.df, make_cfg("rm", exclude={"topic": "sea_ice"}), topics)
    qv = embedder.encode([q["question"] for q in questions], kind="query")
    client = _Client()
    a = _answerer(tmp_path, client)
    assert sample_count(full, 3, {"full"}) == 3 and sample_count(rm, 3, {"full"}) == 1
    res = run(ctx, [full, rm], questions, qv, a, k=5, backend="numpy", repeats=3, repeat_snapshots={"full"})
    assert len(res) == len(questions) * (3 + 1)
    assert set(res[res.snapshot == "full"]["sample"]) == {0, 1, 2} and set(res[res.snapshot == "rm"]["sample"]) == {0}
    # replicates of one prompt are different draws, but retrieval is identical across them
    q0 = res[(res.snapshot == "full") & (res.question_id == questions[0]["id"])]
    assert q0["answer"].nunique() == 3 and q0["retrieved_ids"].map(tuple).nunique() == 1
    assert client.calls >= 3  # at least the three draws of one prompt were paid separately


def test_run_without_repeats_has_sample_zero_and_no_extra_calls(ctx, topics, make_cfg, questions, embedder):
    full = resolve_snapshot(ctx.df, make_cfg("full"), topics)
    qv = embedder.encode([q["question"] for q in questions], kind="query")
    res = run(ctx, [full], questions, qv, rag.ExtractiveAnswerer(0.0), k=5, backend="numpy")
    assert (res["sample"] == 0).all() and len(res) == len(questions)
    cells = list(iter_cells(ctx, [full], questions, retrieve_all(ctx, [full], qv, 5, "numpy")))
    assert len(cells) == len(questions)


# ------------------------------------------------------------------ topic overlap and question audit


def test_topic_overlap_is_a_share_matrix(ctx, topics):
    ov = topic_overlap(ctx.df, topics)
    assert list(ov.index) == list(ov.columns) == ["sea_ice", "earthquakes"]
    assert ov.loc["sea_ice", "sea_ice"] == 1.0 and ov.loc["sea_ice", "earthquakes"] == 0.0


def test_audit_flags_weakly_targeted_and_duplicate_questions(ctx, topics, make_cfg, embedder):
    full = resolve_snapshot(ctx.df, make_cfg("full"), topics)
    qs = [
        {"id": "a", "topic": "sea_ice", "question": "arctic sea ice melt"},
        {"id": "b", "topic": "sea_ice", "question": "earthquake fault rupture"},  # retrieves earthquakes
        {"id": "c", "topic": "earthquakes", "question": "earthquake fault rupture"},  # duplicate of b
        {"id": "d", "topic": None, "question": "arctic sea ice melt"},  # control that leans on sea_ice
    ]
    qv = embedder.encode([q["question"] for q in qs], kind="query")
    tab = audit_questions(ctx, full, qs, qv, topics, k=5, backend="numpy", min_score=0.0).set_index("id")
    assert tab.loc["a", "own_share"] == 1.0 and "weakly targeted" not in tab.loc["a", "flags"]
    assert "weakly targeted" in tab.loc["b", "flags"]
    assert "near-duplicate of c" in tab.loc["b", "flags"] and "near-duplicate of a" in tab.loc["d", "flags"]
    assert "control leans on sea_ice" in tab.loc["d", "flags"]
    assert np.isnan(tab.loc["d", "own_share"])
