"""Turn a run directory into tables, figures and runs/<id>/report.md.

    aixiv-rag report --run 20261015T120000Z

Imports no FAISS (it may import torch to embed answers for the drift metric).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics
from .data import load_corpus
from .embed import vectors
from .paths import RUNS

BLUE, ORANGE = "#2a6fbb", "#d9822b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "font.size": 9})


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if v is pd.NA:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    res = pd.read_parquet(run_dir / "results.parquet")
    ids = load_corpus(columns=["arxiv_id"])["arxiv_id"].to_numpy()
    snapshot_ids = {
        name: set(ids[np.load(run_dir / "snapshot_rows" / f"{name}.npy")]) for name in meta["snapshots"]
    }
    return meta, res, snapshot_ids


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_topic_effect(tab: pd.DataFrame, path: Path) -> None:
    """Left: retrieval overlap by topic under removal vs each control. Right: the gap with its 95% CI
    (positive = removing the topic hurt more than the control). One marker per control kind."""
    topics = [t for t in tab["topic"].unique() if t != "ALL TOPICS"] + (["ALL TOPICS"] if (tab["topic"] == "ALL TOPICS").any() else [])
    kinds = [k for k in ("outside", "all") if (tab["control"] == k).any()]
    colour = {"outside": BLUE, "all": "#4a4a4a"}
    label = {"outside": "control: random outside topic", "all": "control: random anywhere"}
    gap_col = "abstain_gap" if "abstain_gap" in tab else "jaccard_gap"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(9, 0.8 * len(topics) + 6), 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(topics))
    first = tab[tab["control"] == kinds[0]].set_index("topic")
    ax1.bar(x - 0.27, [first.loc[t, "jaccard_removed"] if t in first.index else np.nan for t in topics], 0.27, color=ORANGE, label="topic removed")
    for i, k in enumerate(kinds):
        sub = tab[tab["control"] == k].set_index("topic")
        ax1.bar(x + 0.0 + 0.27 * i, [sub.loc[t, "jaccard_control"] if t in sub.index else np.nan for t in topics], 0.27, color=colour[k], label=label[k])
    ax1.set_xticks(x, topics, rotation=40, ha="right")
    ax1.set_ylabel("Jaccard of top-k vs full\n(questions about the topic)")
    ax1.set_ylim(0, 1.25)
    ax1.legend(frameon=False, loc="upper center", ncol=1, fontsize=7)
    y = np.arange(len(topics))[::-1]
    for i, k in enumerate(kinds):
        sub = tab[tab["control"] == k].set_index("topic")
        for yy, t in zip(y, topics):
            if t not in sub.index:
                continue
            r = sub.loc[t]
            off = (i - (len(kinds) - 1) / 2) * 0.25
            ax2.plot([r[f"{gap_col}_lo"], r[f"{gap_col}_hi"]], [yy + off] * 2, color=colour[k], lw=1.5)
            ax2.plot(r[gap_col], yy + off, "o", color=colour[k], ms=4)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y, topics)
    ax2.set_xlabel(f"{gap_col.replace('_', ' ')}  (>0: topic removal hurts more)\n95% bootstrap CI over questions")
    _save(fig, path)


def fig_dose_response(tab: pd.DataFrame, path: Path) -> None:
    ycol = "abstain_rate" if "abstain_rate" in tab else "mean_change"
    order = list(tab["evidence_lost"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colours = {"topic_removal": ORANGE, "control_outside": BLUE, "control_all": "#4a4a4a", "time_slice": "#7a9e3f", "random": "#8a5fb0"}
    for kind, g in tab.groupby("kind"):
        g = g.assign(_x=g["evidence_lost"].astype(str).map(order.index)).sort_values("_x")
        ax.plot(g["_x"], g[ycol], "o-", color=colours.get(kind, "gray"), label=kind)
        for _, r in g.iterrows():
            ax.annotate(int(r["n"]), (r["_x"], r[ycol]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
    ax.set_ylabel(ycol.replace("_", " "))
    ax.set_xlabel("share of the reference top-k removed from the snapshot  (numbers = n)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path)


def fig_stability(df: pd.DataFrame, ref: str, path: Path) -> None:
    names = [s for s in df["snapshot"].unique() if s != ref]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(names) + 2), 3.8))
    rng = np.random.default_rng(0)
    for x, s in enumerate(names):
        g = df[df["snapshot"] == s]
        for flag, colour in ((False, BLUE), (True, ORANGE)):
            sub = g[g["affected"] == flag]
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(sub)), sub["jaccard"], s=14, color=colour, alpha=0.8)
        ax.hlines(g["jaccard"].mean(), x - 0.3, x + 0.3, color="black", lw=1.2)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_ylabel(f"Jaccard of top-k vs {ref}")
    ax.set_ylim(-0.03, 1.05)
    ax.scatter([], [], color=BLUE, s=14, label="top-k untouched")
    ax.scatter([], [], color=ORANGE, s=14, label="top-k lost documents")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def fig_temporal(hist: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(hist) + 1)))
    im = ax.imshow(hist.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(hist)), hist.index)
    step = max(1, -(-len(hist.columns) // 8))  # ceil: at most ~8 year labels
    ax.set_xticks(range(0, len(hist.columns), step), [int(c) for c in hist.columns[::step]])
    ax.set_xlabel("publication year of retrieved documents")
    fig.colorbar(im, ax=ax, label="share of retrieved docs")
    _save(fig, path)


def fig_abstention(tab: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(tab) + 2), 3.8))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["abstain_rate_targeted"].fillna(0), 0.4, color=ORANGE, label="targeted questions")
    ax.bar(x + 0.2, tab["abstain_rate_nontargeted"].fillna(0), 0.4, color=BLUE, label="other questions")
    ax.set_xticks(x, tab["snapshot"], rotation=60, ha="right")
    ax.set_ylabel("abstention rate")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    _save(fig, path)


def fig_drift(df: pd.DataFrame, ref: str, path: Path) -> None:
    names = [s for s in df["snapshot"].unique() if s != ref]
    data = [df.loc[df["snapshot"] == s, "answer_drift"].dropna().to_numpy() for s in names]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(names) + 2), 3.8))
    ax.boxplot([d if len(d) else [np.nan] for d in data], tick_labels=names, medianprops={"color": ORANGE})
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel(f"answer drift vs {ref} (1 - cosine)")
    _save(fig, path)


def build_report(run_dir: Path, with_drift: bool = True) -> Path:
    meta, res_all, snapshot_ids = load_run(run_dir)
    ref = meta["reference_snapshot"]
    out_dir = run_dir / "metrics"
    out_dir.mkdir(exist_ok=True)

    res = metrics.primary(res_all)  # draw 0: the answer every metric uses; extra draws only feed the noise floor
    df = metrics.attach_reference(res, ref, snapshot_ids)
    has_answers = meta["answerer"] != "none"
    removed = {n: s["removed_topics"] for n, s in meta["snapshots"].items()}
    notes: list[str] = []

    stab = metrics.stability_table(df, ref)
    stab.to_csv(out_dir / "stability.csv", index=False)
    fig_stability(df, ref, out_dir / "stability.png")

    temporal = metrics.temporal_profile(df)
    temporal.to_csv(out_dir / "temporal.csv", index=False)
    fig_temporal(metrics.year_histogram(df), out_dir / "temporal.png")

    abst = drift = cites = noise_tab = None
    noise: dict = {}
    between = None
    if has_answers:
        df = metrics.attach_citations(df, ref, snapshot_ids)
        abst = metrics.abstention_table(df, removed)
        abst.to_csv(out_dir / "abstention.csv", index=False)
        fig_abstention(abst, out_dir / "abstention.png")
        cites = metrics.citation_table(df, ref)
        cites.to_csv(out_dir / "citations.csv", index=False)
        if with_drift:
            try:
                tag = meta["embedder"]["tag"]
                embed_fn = lambda t: vectors(t, kind="doc", tag=tag)
                df = metrics.answer_drift(df, ref, embed_fn)
                between = metrics.between_question_distance(df, ref, embed_fn)
                noise_tab, noise = metrics.noise_floor(res_all, ref, embed_fn)
                drift = metrics.drift_table(df, ref, noise.get("median_drift"), between)
                drift.to_csv(out_dir / "drift.csv", index=False)
                fig_drift(df, ref, out_dir / "drift.png")
                if len(noise_tab):
                    noise_tab.to_csv(out_dir / "noise_floor.csv", index=False)
            except Exception as e:  # noqa: BLE001 - e.g. embedding model unavailable offline
                notes.append(f"> Answer drift skipped: {type(e).__name__}: {e}\n")

    effect = metrics.paired_topic_effect(df, meta["snapshots"], ref)
    if not effect.empty:
        effect.to_csv(out_dir / "topic_effect.csv", index=False)
        fig_topic_effect(effect, out_dir / "topic_effect.png")
    dose = metrics.dose_response(df, meta["snapshots"], ref) if has_answers else pd.DataFrame()
    if len(dose):
        dose.to_csv(out_dir / "dose_response.csv", index=False)
        fig_dose_response(dose, out_dir / "dose_response.png")

    checks = metrics.sanity_checks(df, ref, snapshot_ids) + metrics.replicate_checks(res_all)

    emb = meta["embedder"]
    banners = []
    if emb.get("backend") == "hash":
        banners.append("**Embedder is the offline `hash` baseline - a pipeline check, not a research result.**")
    if str(meta["answerer"]).startswith("extractive"):
        banners.append("**Answers come from the extractive baseline, not an LLM.**")
    if not has_answers:
        banners.append("Retrieval-only run: no answers, so abstention and drift are not available.")
    per_topic = df[df["snapshot"] == ref].groupby("question_topic").size()
    if len(per_topic) and per_topic.min() < 6:
        banners.append(
            f"**Small sample: {int(per_topic.min())} question(s) for the thinnest topic.** Read the intervals, "
            "not the point estimates; add questions in configs/questions.yaml."
        )
    if has_answers and not noise:
        banners.append(
            "**No noise floor measured** (single draw per prompt): a difference between two snapshots cannot be told "
            "apart from the model's own run-to-run variation. Re-run with `--repeats 5`."
        )

    snap_tab = pd.DataFrame(
        [
            {
                "snapshot": n,
                "kind": metrics.snapshot_kind(s["cfg"]),
                "n_docs": s["n_docs"],
                "removed_topics": ", ".join(s["removed_topics"]) or "-",
                "cfg": s["cfg_hash"][:8],
            }
            for n, s in meta["snapshots"].items()
        ]
    )

    lines = [
        f"# Ablation report - run {meta['run_id']}",
        "",
        *[b + "\n" for b in banners],
        f"- created: {meta['created_utc']}  |  answerer: `{meta['answerer']}`  |  embedder: `{emb['tag']}` ({emb.get('model') or emb.get('backend')})",
        (f"- k = {meta['k']}, backend = {meta['backend']}, reference snapshot = `{ref}`, git = {meta.get('git_commit') or 'n/a'}"
         f"{' **+ uncommitted changes: ' + ', '.join(meta['git_dirty']) + '**' if meta.get('git_dirty') else ''}, "
         f"questions = {df['question_id'].nunique()}, draws/prompt on the reference = {meta.get('repeats', 1)}, "
         f"abstention rule = `{meta.get('abstain_rule', 'first-line')}`"),
        "",
        "## Sanity checks",
        *[f"- {'PASS' if ok else '**FAIL**'} - {name} ({detail})" for name, ok, detail in checks],
        "",
    ]
    if not effect.empty:
        lines += [
            "## Headline: topic removal vs size-matched controls",
            ("For each topic, the questions *about that topic*, under `remove_<topic>` and under two random controls that remove "
             "the same number of documents: `outside` draws only from outside the topic (it cannot touch the topic's evidence, "
             "so it is a gentle baseline - a large gap against it is almost guaranteed), `all` draws from the whole corpus "
             "(the fair baseline: topic documents can be hit, just not on purpose). Every `*_gap` is signed so that "
             "**positive = removing the topic hurt more than the control**, with a 95% bootstrap interval over questions. "
             "An interval that includes 0 means this run cannot tell the topic from chance."),
            "",
            md_table(effect),
            "",
            "![topic effect](metrics/topic_effect.png)",
            "",
        ]
    if len(dose):
        lines += [
            "## Evidence loss: topic removal vs random removal at equal loss",
            ("Cells are grouped by how much of the *reference* top-k a snapshot removed, whatever the snapshot was. "
             "If topic removal and random removal behave alike at the same loss, the effect is about lost evidence, "
             "not about the topic. Small cells (n < 5) are noise."),
            "",
            md_table(dose),
            "",
            "![dose response](metrics/dose_response.png)",
            "",
        ]
    lines += [
        "## Snapshots",
        md_table(snap_tab),
        "",
        "## 1. Retrieval stability",
        "Jaccard overlap of top-k ids with the reference. `affected` = at least one reference top-k doc was removed.",
        "",
        md_table(stab),
        "",
        "![stability](metrics/stability.png)",
        "",
    ]
    if noise:
        lines += [
            "## Noise floor: the answerer against itself",
            (f"{noise['n_questions']} reference questions, {noise['draws_per_question']:.1f} independent draws each of the *same* prompt. "
             "Anything a corpus change does has to be read against this."),
            "",
            md_table(pd.DataFrame([noise])),
            "",
        ]
    if drift is not None:
        ruler = ""
        if between is not None and not np.isnan(between) and noise:
            ruler = (f" `scaled_drift_affected` = (drift - noise) / (between - noise) with noise = {noise['median_drift']:.3f} "
                     f"(median same-prompt drift) and between = {between:.3f} (mean drift between answers to *different* questions): "
                     "0 = indistinguishable from the model's own noise, 1 = as different as an unrelated answer.")
        lines += [
            "## 2. Answer drift",
            ("1 - cosine between embedded answers, non-abstained pairs only; `n_flipped` = abstention status changed vs the reference. "
             "`mean_change_affected` counts a flip as 1.0 and two abstentions as 0, so it covers every pair. "
             "Unaffected questions have a byte-identical prompt, hence a cached identical answer and drift exactly 0 - that is "
             "the cache, not stability, which is why the `_affected` columns are the ones to read." + ruler),
            "",
            md_table(drift),
            "",
            "![drift](metrics/drift.png)",
            "",
        ]
    lines += [
        "## 3. Temporal profile of retrieved documents",
        md_table(temporal),
        "",
        "![temporal](metrics/temporal.png)",
        "",
    ]
    if abst is not None:
        lines += [
            "## 4. Abstention and citation support",
            ("`targeted` = questions whose evidence topic was removed from that snapshot. "
             "`answered_targeted` > 0 means the system answered although its supporting topic was removed "
             "(candidate fabrication - read those answers, or see the judge section). "
             "`invalid_citation_rate` = answers citing ids that were not in the context. "
             "Abstention here is the sentinel string `INSUFFICIENT_EVIDENCE` from one prompt and one model: "
             "a different prompt or model can abstain differently (see `compare-runs`)."),
            "",
            md_table(abst),
            "",
            "![abstention](metrics/abstention.png)",
            "",
        ]
        if cites is not None and len(cites):
            lines += [
                ("Did the answer keep leaning on the same documents? `mean_cited_overlap` = Jaccard of cited ids against the "
                 "reference answer (both answered); `ref_cited_lost` = share of the reference answer's citations the snapshot removed."),
                "",
                md_table(cites),
                "",
            ]
    jpath = run_dir / "judge.parquet"
    if jpath.exists():
        from . import judge

        lines += judge.report_section(pd.read_parquet(jpath), df, meta, out_dir)
    elif has_answers:
        lines += [
            "## 5. Answer quality",
            ("Not judged. Everything above measures *change*, not *correctness*: an answer can drift and be fine, or stay "
             "put and be wrong. `aixiv-rag judge --run <id>` adds an independent LLM judge of support and of whether each "
             "abstention was justified (see AGENTS.md; it costs money and is itself unvalidated until you hand-check a sample)."),
            "",
        ]
    lines += notes

    report = run_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the report for one run.")
    parser.add_argument("--run", required=True, help="run id (folder under runs/) or a path")
    parser.add_argument("--no-drift", action="store_true", help="skip the answer-drift metric (needs the embedder)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).is_dir() else RUNS / args.run
    path = build_report(run_dir, with_drift=not args.no_drift)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
