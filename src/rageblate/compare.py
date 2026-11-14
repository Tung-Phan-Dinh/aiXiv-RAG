"""Do two runs tell the same story?  The conclusions of one run rest on one prompt, one model, one
abstention sentinel. Re-run with a changed ingredient (`ablate --prompt-version v2`, `--model`, `--effort`,
or a different embedder) and ask whether the *effect* survives, not whether the raw numbers match:

    aixiv-rag compare-runs <run_a> <run_b>

Per topic, against the harder (`all`) random control when both runs have it: the abstention gap and the
Jaccard gap with their bootstrap intervals in each run, whether the sign agrees, and whether the intervals
overlap. Absolute abstention rates will differ between prompts - that is the point of checking.
Only questions present in both runs are compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .paths import RUNS
from .report import load_run, md_table


def effect_for(run_dir: Path, questions: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    meta, res_all, ids = load_run(run_dir)
    res = metrics.primary(res_all)
    if questions is not None:
        res = res[res["question_id"].isin(questions)]
    df = metrics.attach_reference(res, meta["reference_snapshot"], ids)
    tab = metrics.paired_topic_effect(df, meta["snapshots"], meta["reference_snapshot"])
    ref = df[df["snapshot"] == meta["reference_snapshot"]]
    info = {
        "answerer": meta["answerer"],
        "prompt_version": meta.get("prompt_version"),
        "embedder": meta["embedder"]["tag"],
        "ref_abstain_rate": float(ref["abstained"].astype(float).mean()) if ref["abstained"].notna().all() else np.nan,
        "n_questions": df["question_id"].nunique(),
    }
    return tab, info


def compare(tab_a: pd.DataFrame, tab_b: pd.DataFrame, metric: str = "abstain_gap") -> pd.DataFrame:
    def pick(t):
        if t.empty:
            return t
        kind = "all" if (t["control"] == "all").any() else "outside"
        return t[t["control"] == kind].set_index("topic")

    a, b = pick(tab_a), pick(tab_b)
    if a.empty or b.empty or metric not in a or metric not in b:
        return pd.DataFrame()
    rows = []
    for topic in [t for t in a.index if t in b.index]:
        ra, rb = a.loc[topic], b.loc[topic]
        overlap = not (ra[f"{metric}_hi"] < rb[f"{metric}_lo"] or rb[f"{metric}_hi"] < ra[f"{metric}_lo"])
        rows.append({
            "topic": topic,
            f"{metric}_A": ra[metric], "A_lo": ra[f"{metric}_lo"], "A_hi": ra[f"{metric}_hi"],
            f"{metric}_B": rb[metric], "B_lo": rb[f"{metric}_lo"], "B_hi": rb[f"{metric}_hi"],
            "same_sign": bool(np.sign(ra[metric]) == np.sign(rb[metric])) if ra[metric] and rb[metric] else np.nan,
            "intervals_overlap": overlap,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the topic-removal effect between two runs.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--metric", default="abstain_gap", choices=["abstain_gap", "jaccard_gap", "change_gap"])
    args = parser.parse_args(argv)
    dirs = [Path(r) if Path(r).is_dir() else RUNS / r for r in (args.run_a, args.run_b)]

    qs = [set(pd.read_parquet(d / "results.parquet")["question_id"]) for d in dirs]
    common = qs[0] & qs[1]
    if common != qs[0] or common != qs[1]:
        print(f"note: runs differ in questions ({len(qs[0])} vs {len(qs[1])}); comparing the {len(common)} they share")
    (ta, ia), (tb, ib) = (effect_for(d, common) for d in dirs)
    for tag, info in (("A", ia), ("B", ib)):
        print(f"{tag}: {info['answerer']}  embedder={info['embedder']}  reference abstain rate={info['ref_abstain_rate']:.2f}")
    out = compare(ta, tb, args.metric)
    if out.empty:
        raise SystemExit(f"nothing to compare for {args.metric} (needs answers and a control in both runs)")
    print(out.round(3).to_string(index=False))
    pooled = out[out["topic"] == "ALL TOPICS"]
    if len(pooled):
        p = pooled.iloc[0]
        print(f"\npooled {args.metric}: A={p.iloc[1]:.3f} B={p.iloc[4]:.3f}; same sign={p['same_sign']}; intervals overlap={p['intervals_overlap']}")
    (dirs[1] / f"compare_vs_{dirs[0].name}.md").write_text(
        f"# {dirs[1].name} vs {dirs[0].name} ({args.metric})\n\nA = {ia}\n\nB = {ib}\n\n" + md_table(out) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
