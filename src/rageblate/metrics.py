"""The four metrics, as small pure functions over the results table.

Reference snapshot = the full corpus. For every (snapshot, question):

  1. Retrieval stability   Jaccard(top-k ids in snapshot, top-k ids in reference).
  2. Answer drift          1 - cosine(embedding of answer, embedding of reference answer).
  3. Temporal profile      distribution of publication years among retrieved documents.
  4. Abstention / support  did the model abstain, and do its citations point at retrieved docs?

Beyond the four, and added because a single number per cell is not enough to trust:

  * bootstrap CIs over questions on every headline contrast (`bootstrap_ci`);
  * `answer_change`: one number defined for EVERY pair - 1.0 if abstention flipped, 0.0 if both
    abstained, else the embedding drift - so drift no longer silently ignores the interesting cells;
  * a noise floor (`noise_floor`): how much two draws of the SAME prompt differ, and a scale
    (`scale_drift`) that expresses drift relative to that floor and to unrelated-question distance;
  * citation overlap and lost-evidence share (`attach_citations`): did the answer lean on documents
    the snapshot removed;
  * a dose-response over evidence loss (`dose_response`), so topic removal and random removal are
    compared at the same amount of lost evidence, not just the same corpus size.

Everything except `noise_floor` and `replicate_checks` takes the PRIMARY rows only (sample == 0);
`primary()` selects them and also accepts results from before replicates existed.

Built-in sanity property ("monotonicity"): if none of the reference top-k documents were removed
from a snapshot, exact search must return the identical top-k. `ref_missing == 0` therefore
implies `jaccard == 1`; any violation means a bug (misalignment, index built wrongly, ties).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def primary(res: pd.DataFrame) -> pd.DataFrame:
    """Rows of draw 0 (the answer every metric uses). Old results without a `sample` column pass through."""
    if "sample" not in res.columns:
        return res.reset_index(drop=True)
    return res[res["sample"] == 0].reset_index(drop=True)


def bootstrap_ci(values, n_boot: int = 4000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) of the mean, percentile bootstrap over the items (questions). NaNs are dropped.
    With fewer than 2 items there is no interval: lo = hi = NaN. Honest, not clever: with 8 questions
    per topic the intervals are wide, and they should be."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) < 2:
        return (float(v.mean()), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return (float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 1.0


def attach_reference(res: pd.DataFrame, ref_name: str, snapshot_ids: dict[str, set[str]]) -> pd.DataFrame:
    """Add jaccard / ref_missing / retained_frac / affected columns relative to the reference."""
    ref = res[res["snapshot"] == ref_name].set_index("question_id")
    if ref.empty:
        raise ValueError(f"reference snapshot {ref_name!r} not in results")
    out = res.copy()
    out["ref_ids"] = out["question_id"].map(lambda q: list(ref.loc[q, "retrieved_ids"]))
    out["jaccard"] = [jaccard(list(a), b) for a, b in zip(out["retrieved_ids"], out["ref_ids"])]
    out["ref_missing"] = [
        sum(1 for i in ids if i not in snapshot_ids[s]) for ids, s in zip(out["ref_ids"], out["snapshot"])
    ]
    out["retained_frac"] = 1 - out["ref_missing"] / out["ref_ids"].map(len).clip(lower=1)
    out["affected"] = out["ref_missing"] > 0
    return out


def stability_table(df: pd.DataFrame, ref_name: str) -> pd.DataFrame:
    rows = []
    for snap, g in df.groupby("snapshot", sort=False):
        if snap == ref_name:
            continue
        hit, safe = g[g["affected"]], g[~g["affected"]]
        rows.append(
            {
                "snapshot": snap,
                "n_questions": len(g),
                "mean_jaccard": g["jaccard"].mean(),
                "mean_retained": g["retained_frac"].mean(),
                "share_affected": g["affected"].mean(),
                "jaccard_affected": hit["jaccard"].mean() if len(hit) else np.nan,
                "jaccard_unaffected": safe["jaccard"].mean() if len(safe) else np.nan,
                "monotonicity_violations": int((safe["jaccard"] < 1 - 1e-9).sum()),
            }
        )
    return pd.DataFrame(rows)


def _years(series: pd.Series) -> np.ndarray:
    if len(series) == 0:
        return np.array([], dtype=int)
    return np.concatenate([np.asarray(list(y), dtype=int) for y in series])


def temporal_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for snap, g in df.groupby("snapshot", sort=False):
        y = _years(g["retrieved_years"])
        rows.append(
            {
                "snapshot": snap,
                "mean_year": float(y.mean()),
                "median_year": float(np.median(y)),
                "p10_year": float(np.percentile(y, 10)),
                "p90_year": float(np.percentile(y, 90)),
            }
        )
    return pd.DataFrame(rows)


def year_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = snapshots, columns = years, values = share of retrieved docs from that year."""
    counts = {}
    for snap, g in df.groupby("snapshot", sort=False):
        counts[snap] = pd.Series(_years(g["retrieved_years"])).value_counts()
    table = pd.DataFrame(counts).T.fillna(0.0)
    table = table.reindex(columns=range(int(table.columns.min()), int(table.columns.max()) + 1), fill_value=0.0)
    return table.div(table.sum(axis=1), axis=0)


def answer_drift(df: pd.DataFrame, ref_name: str, embed_fn: Callable[[list[str]], np.ndarray]) -> pd.DataFrame:
    """Add answer_drift (NaN if either answer is an abstention) and flipped (abstention status changed)."""
    ref = df[df["snapshot"] == ref_name].set_index("question_id")
    out = df.copy()
    ref_text = out["question_id"].map(ref["answer"])
    ref_abs = out["question_id"].map(ref["abstained"]).astype(bool)
    out["flipped"] = out["abstained"].astype(bool) != ref_abs

    usable = (~out["abstained"].astype(bool)) & (~ref_abs)
    texts = sorted(set(out.loc[usable, "answer"]) | set(ref_text[usable]))
    vec = dict(zip(texts, embed_fn(texts))) if texts else {}
    drift = [
        max(0.0, 1.0 - float(np.dot(vec[a], vec[b]))) if u else np.nan  # clip float noise below 0
        for a, b, u in zip(out["answer"], ref_text, usable)
    ]
    out["answer_drift"] = drift
    both_abstained = out["abstained"].astype(bool) & ref_abs
    # defined for every pair: a flip is the largest possible change, two abstentions are identical
    out["answer_change"] = np.where(out["flipped"], 1.0, np.where(both_abstained, 0.0, out["answer_drift"]))
    return out


def drift_table(df: pd.DataFrame, ref_name: str, noise: float | None = None, between: float | None = None) -> pd.DataFrame:
    """Per snapshot. `*_affected` restricts to questions whose top-k lost a document: for the others the
    prompt is byte-identical to the reference, so the cached answer is too and drift is exactly 0 - that
    is not stability evidence about the model, it is the cache. With `noise` (median same-prompt drift)
    and `between` (mean drift between unrelated questions) the scaled column puts drift on a 0..1 ruler:
    0 = as different as two draws of the same prompt, 1 = as different as answers to unrelated questions."""
    rows = []
    for snap, g in df.groupby("snapshot", sort=False):
        if snap == ref_name:
            continue
        aff = g[g["affected"]]
        row = {
            "snapshot": snap,
            "mean_drift": g["answer_drift"].mean(),
            "median_drift": g["answer_drift"].median(),
            "n_compared": int(g["answer_drift"].notna().sum()),
            "n_flipped": int(g["flipped"].sum()),
            "mean_drift_affected": aff["answer_drift"].mean() if len(aff) else np.nan,
            "mean_change_affected": aff["answer_change"].mean() if len(aff) and "answer_change" in aff else np.nan,
        }
        if noise is not None and between is not None:
            row["scaled_drift_affected"] = scale_drift(row["mean_drift_affected"], noise, between)
        rows.append(row)
    return pd.DataFrame(rows)


def scale_drift(drift, noise: float, between: float):
    """(drift - noise) / (between - noise): 0 at the same-prompt noise floor, 1 at unrelated-question
    distance. Not clipped (negative = quieter than noise). NaN if the ruler is degenerate."""
    if between is None or noise is None or between <= noise:
        return np.nan
    return (drift - noise) / (between - noise)


def citation_table(df: pd.DataFrame, ref_name: str) -> pd.DataFrame:
    """Per snapshot: how much the answer's cited evidence moved relative to the reference answer."""
    rows = []
    for snap, g in df.groupby("snapshot", sort=False):
        if snap == ref_name:
            continue
        rows.append({
            "snapshot": snap,
            "mean_cited_overlap": g["cited_overlap"].mean(),
            "n_overlap": int(g["cited_overlap"].notna().sum()),
            "ref_cited_lost": g["ref_cited_lost"].mean(),
        })
    return pd.DataFrame(rows)


def between_question_distance(df: pd.DataFrame, ref_name: str, embed_fn: Callable[[list[str]], np.ndarray]) -> float:
    """Mean 1-cosine between the reference answers to DIFFERENT questions (answered ones only):
    the 'completely different answer' end of the drift ruler for this embedder."""
    ref = df[(df["snapshot"] == ref_name) & (~df["abstained"].astype(bool))]
    if len(ref) < 2:
        return np.nan
    v = np.asarray(embed_fn(list(ref["answer"])))
    sim = v @ v.T
    iu = np.triu_indices(len(ref), k=1)
    return float(np.clip(1.0 - sim[iu], 0.0, None).mean())


def attach_citations(df: pd.DataFrame, ref_name: str, snapshot_ids: dict[str, set[str]]) -> pd.DataFrame:
    """Add cited_overlap (Jaccard of cited ids vs the reference answer; NaN unless both answered and cite
    something) and ref_cited_lost (share of the reference answer's citations that the snapshot removed)."""
    ref = df[df["snapshot"] == ref_name].set_index("question_id")
    out = df.copy()
    overlap, lost = [], []
    for q, s, ab, cited in zip(out["question_id"], out["snapshot"], out["abstained"].astype(bool), out["cited_ids"]):
        rc = list(ref.loc[q, "cited_ids"]) if not bool(ref.loc[q, "abstained"]) else []
        cc = list(cited)
        overlap.append(jaccard(cc, rc) if (rc and cc and not ab) else np.nan)
        lost.append(sum(1 for c in rc if c not in snapshot_ids[s]) / len(rc) if rc else np.nan)
    out["cited_overlap"] = overlap
    out["ref_cited_lost"] = lost
    return out


def snapshot_kind(cfg: dict) -> str:
    """topic_removal | control_outside | control_all | random | time_slice | other, from a snapshot config."""
    exc, inc = cfg.get("exclude") or {}, cfg.get("include") or {}
    rnd = exc.get("random")
    if exc.get("topic"):
        return "topic_removal"
    if rnd:
        if "matched_to_topic" in rnd:
            return "control_all" if rnd.get("pool") == "all" else "control_outside"
        return "random"
    return "time_slice" if ("year_min" in inc or "year_max" in inc) else "other"


def _per_question(df: pd.DataFrame, snapshots: list[str], qids: set[str], col: str) -> pd.Series:
    """Mean of `col` per question over the given snapshots (several control seeds -> their average)."""
    g = df[df["snapshot"].isin(snapshots) & df["question_id"].isin(qids)]
    return g.groupby("question_id")[col].mean()


def paired_topic_effect(df: pd.DataFrame, snapshots_meta: dict, ref_name: str, seed: int = 0) -> pd.DataFrame:
    """The headline contrast, one row per (topic, control kind) plus a pooled row per control kind.

    For topic T take the questions ABOUT T and compare their outcome under `remove_T` with the same
    questions under the control(s) of that kind, paired by question. Every `*_gap` is oriented so that
    POSITIVE means "removing the topic hurt more than the control did":
      jaccard_gap = jaccard_control - jaccard_removed
      abstain_gap = abstain_removed - abstain_control      (same for change_gap, from answer_change)
    Intervals are percentile bootstrap over questions. Control kinds:
      outside   size-matched random removal drawn only from outside T (cannot touch T's evidence)
      all       size-matched random removal drawn from the whole corpus (the fairer baseline)
    With several seeds of one kind the controls are averaged per question first.
    """
    ctrl: dict[tuple[str, str], list[str]] = {}
    for name, s in snapshots_meta.items():
        kind = snapshot_kind(s["cfg"])
        if kind in ("control_outside", "control_all"):
            ctrl.setdefault((s["cfg"]["exclude"]["random"]["matched_to_topic"], kind.split("_", 1)[1]), []).append(name)

    answered = "abstained" in df and df["abstained"].notna().all()
    work = df.copy()
    if answered:
        work["abstained_f"] = work["abstained"].astype(float)
    has_change = "answer_change" in work

    rows = []
    pooled: dict[str, dict[str, list]] = {}
    for name, s in snapshots_meta.items():
        if len(s["removed_topics"]) != 1 or name == ref_name or snapshot_kind(s["cfg"]) != "topic_removal":
            continue
        topic = s["removed_topics"][0]
        qids = set(work.loc[(work["snapshot"] == name) & (work["question_topic"] == topic), "question_id"])
        if not qids:
            continue
        for kind in ("outside", "all"):
            names = ctrl.get((topic, kind))
            if not names:
                continue
            measures = {"jaccard": +1}  # +1: gap = control - removed
            if answered:
                measures["abstained_f"] = -1  # -1: gap = removed - control
            if has_change:
                measures["answer_change"] = -1
            row = {"topic": topic, "control": kind, "n_questions": len(qids), "n_control_snapshots": len(names)}
            for col, sign in measures.items():
                a = _per_question(work, [name], qids, col)
                b = _per_question(work, names, qids, col)
                diff = (b - a).dropna() if sign > 0 else (a - b).dropna()  # positive = topic removal hurt more
                mean, lo, hi = bootstrap_ci(diff.to_numpy(), seed=seed)
                short = {"jaccard": "jaccard", "abstained_f": "abstain", "answer_change": "change"}[col]
                row[f"{short}_removed"], row[f"{short}_control"] = a.mean(), b.mean()
                row[f"{short}_gap"], row[f"{short}_gap_lo"], row[f"{short}_gap_hi"] = mean, lo, hi
                pooled.setdefault(kind, {}).setdefault(col, []).extend(diff.to_numpy().tolist())
                pooled[kind].setdefault(f"_{col}_a", []).extend(a.reindex(diff.index).to_numpy().tolist())
                pooled[kind].setdefault(f"_{col}_b", []).extend(b.reindex(diff.index).to_numpy().tolist())
            rows.append(row)
    for kind, cols in pooled.items():
        row = {"topic": "ALL TOPICS", "control": kind,
               "n_questions": len(cols["jaccard"]), "n_control_snapshots": np.nan}
        for col in [c for c in cols if not c.startswith("_")]:
            short = {"jaccard": "jaccard", "abstained_f": "abstain", "answer_change": "change"}[col]
            mean, lo, hi = bootstrap_ci(cols[col], seed=seed)
            row[f"{short}_removed"] = float(np.nanmean(cols[f"_{col}_a"]))
            row[f"{short}_control"] = float(np.nanmean(cols[f"_{col}_b"]))
            row[f"{short}_gap"], row[f"{short}_gap_lo"], row[f"{short}_gap_hi"] = mean, lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["n_control_snapshots"] = out["n_control_snapshots"].astype("Int64")
    order = {"outside": 0, "all": 1}
    out["_pooled"] = out["topic"] == "ALL TOPICS"
    out["_k"] = out["control"].map(order)
    return out.sort_values(["_k", "_pooled", "topic"], kind="stable").drop(columns=["_pooled", "_k"]).reset_index(drop=True)


def dose_response(df: pd.DataFrame, snapshots_meta: dict, ref_name: str) -> pd.DataFrame:
    """Outcome by how much of the reference top-k was lost, split by what kind of removal caused it.

    Topic removal and random removal take away the same number of documents but very different amounts
    of *evidence*. Binning on evidence lost makes them comparable: at the same loss, does the kind of
    removal still matter? Sparse cells (n < 5) are shown but should not be over-read."""
    kinds = {n: snapshot_kind(s["cfg"]) for n, s in snapshots_meta.items()}
    d = df[df["snapshot"] != ref_name].copy()
    d["kind"] = d["snapshot"].map(kinds)
    lost = 1.0 - d["retained_frac"]
    labels = ["none lost", "<=1/3 lost", "1/3-2/3 lost", ">2/3 lost", "all lost"]
    d["evidence_lost"] = pd.cut(lost, [-1e-9, 1e-9, 1 / 3 + 1e-9, 2 / 3 + 1e-9, 1 - 1e-9, 1 + 1e-9], labels=labels)
    agg = {"n": ("question_id", "size")}
    if "abstained" in d and d["abstained"].notna().all():
        d["abstained_f"] = d["abstained"].astype(float)
        agg["abstain_rate"] = ("abstained_f", "mean")
    if "answer_change" in d:
        agg["mean_change"] = ("answer_change", "mean")
    out = d.groupby(["evidence_lost", "kind"], observed=True).agg(**agg).reset_index()
    return out.sort_values(["evidence_lost", "kind"]).reset_index(drop=True)


def noise_floor(res: pd.DataFrame, ref_name: str, embed_fn: Callable[[list[str]], np.ndarray]):
    """Run-to-run noise of the answerer itself: for every question with >= 2 draws on the reference
    snapshot, compare all pairs of draws of the SAME prompt (retrieval is identical, so any difference
    is the model). Returns (per_question table, summary dict), or (empty, {}) without replicates."""
    if "sample" not in res.columns:
        return pd.DataFrame(), {}
    ref = res[res["snapshot"] == ref_name]
    if ref["sample"].nunique() < 2:
        return pd.DataFrame(), {}
    texts = sorted(set(ref.loc[~ref["abstained"].astype(bool), "answer"]))
    vec = dict(zip(texts, embed_fn(texts))) if texts else {}
    rows = []
    for q, g in ref.groupby("question_id", sort=False):
        g = g.sort_values("sample")
        recs = list(zip(g["abstained"].astype(bool), g["answer"], g["cited_ids"]))
        changes, drifts, flips, cj = [], [], 0, []
        pairs = 0
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                (ai, ti, ci), (aj, tj, cj_) = recs[i], recs[j]
                pairs += 1
                if ai != aj:
                    flips += 1
                    changes.append(1.0)
                elif ai and aj:
                    changes.append(0.0)
                else:
                    d = max(0.0, 1.0 - float(np.dot(vec[ti], vec[tj])))
                    drifts.append(d)
                    changes.append(d)
                    if list(ci) or list(cj_):
                        cj.append(jaccard(list(ci), list(cj_)))
        rows.append({
            "question_id": q, "question_topic": g["question_topic"].iloc[0], "n_draws": len(g), "n_pairs": pairs,
            "abstain_disagreement": flips / pairs, "mean_drift": float(np.mean(drifts)) if drifts else np.nan,
            "mean_change": float(np.mean(changes)), "cited_jaccard": float(np.mean(cj)) if cj else np.nan,
        })
    per_q = pd.DataFrame(rows)
    summary = {
        "n_questions": len(per_q),
        "draws_per_question": float(per_q["n_draws"].mean()),
        "median_drift": float(per_q["mean_drift"].median()),
        "mean_drift": float(per_q["mean_drift"].mean()),
        "mean_change": float(per_q["mean_change"].mean()),
        "abstain_disagreement": float(per_q["abstain_disagreement"].mean()),
        "cited_jaccard": float(per_q["cited_jaccard"].mean()),
    }
    return per_q, summary


def replicate_checks(res: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Draws of one (snapshot, question) must share their retrieval exactly; differences are the model's alone."""
    if "sample" not in res.columns or res["sample"].max() == 0:
        return []
    varying = res.assign(_t=res["retrieved_ids"].map(tuple)).groupby(["snapshot", "question_id"])["_t"].nunique()
    bad = int((varying > 1).sum())
    return [("replicate draws share identical retrieval", bad == 0, f"{bad} cells differ")]


def abstention_table(df: pd.DataFrame, removed_topics: dict[str, list[str]]) -> pd.DataFrame:
    """Per snapshot. 'targeted' = questions whose evidence topic was removed from that snapshot."""
    rows = []
    for snap, g in df.groupby("snapshot", sort=False):
        g = g.copy()
        g["abstained"] = g["abstained"].astype(bool)
        targeted = g["question_topic"].isin(set(removed_topics.get(snap, [])))
        t, n = g[targeted], g[~targeted]
        rows.append(
            {
                "snapshot": snap,
                "abstain_rate": g["abstained"].mean(),
                "n_targeted": len(t),
                "abstain_rate_targeted": t["abstained"].mean() if len(t) else np.nan,
                "answered_targeted": int((~t["abstained"]).sum()) if len(t) else 0,
                "abstain_rate_nontargeted": n["abstained"].mean() if len(n) else np.nan,
                "invalid_citation_rate": float((g["invalid_cited_ids"].map(len) > 0).mean()),
                "mean_citations": float(g["cited_ids"].map(len).mean()),
            }
        )
    return pd.DataFrame(rows)


def sanity_checks(df: pd.DataFrame, ref_name: str, snapshot_ids: dict[str, set[str]]) -> list[tuple[str, bool, str]]:
    """Cheap invariants that must hold for any correct run: (name, passed, detail)."""
    checks = []
    leaks = sum(
        sum(1 for i in ids if i not in snapshot_ids[s]) for ids, s in zip(df["retrieved_ids"], df["snapshot"])
    )
    checks.append(("every retrieved doc is inside its snapshot", leaks == 0, f"{leaks} leaked docs"))

    ref_rows = df[df["snapshot"] == ref_name]
    ok_ref = bool((ref_rows["jaccard"] == 1.0).all())
    checks.append(("reference vs itself has Jaccard 1", ok_ref, ""))

    safe = df[(df["snapshot"] != ref_name) & (~df["affected"])]
    bad = int((safe["jaccard"] < 1 - 1e-9).sum())
    checks.append(("monotonicity: unaffected questions keep an identical top-k", bad == 0, f"{bad} violations of {len(safe)}"))

    short = int((df["k"] < df["k"].max()).sum())
    checks.append(("every query returned the full k documents", short == 0, f"{short} short result lists"))
    return checks
