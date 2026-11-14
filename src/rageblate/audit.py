"""Audit the evaluation questions before spending API money on them.

    aixiv-rag audit-questions [--embedder TAG] [--snapshot full] [--min-score 0.72]

For every question: its top-1 cosine score, what share of its top-k documents lie inside each topic,
and whether it is near-duplicate of another question. A good topic-tagged question draws most of
its evidence from its own topic (otherwise "remove the topic" does not target it); a good control
question draws it from no single removable topic. Retrieval only - no LLM calls, no cost.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_questions, load_rag_config, load_topics
from .data import Context, load_context
from .embed import vectors
from .index import build_index
from .snapshot import Snapshot, resolve_by_name
from .topics import topic_mask


def audit_questions(
    ctx: Context,
    snapshot: Snapshot,
    questions: list[dict],
    qvecs: np.ndarray,
    topics: dict[str, dict],
    k: int = 10,
    backend: str = "numpy",
    min_score: float = 0.72,
    own_share_min: float = 0.6,
    other_share_max: float = 0.5,
    dup_cosine: float = 0.9,
) -> pd.DataFrame:
    scores, rows = build_index(ctx, snapshot, backend).search(qvecs, k)
    masks = {t: topic_mask(ctx.df, spec) for t, spec in topics.items()}
    sim = qvecs @ qvecs.T
    np.fill_diagonal(sim, -1.0)

    out = []
    for i, q in enumerate(questions):
        share = {t: float(m[rows[i]].mean()) for t, m in masks.items()}
        own = q.get("topic")
        others = {t: s for t, s in share.items() if t != own}
        best_other, best_other_share = max(others.items(), key=lambda kv: kv[1])
        flags = []
        if own and share[own] < own_share_min:
            flags.append(f"weakly targeted ({share[own]:.0%} of top-{k} in {own})")
        if scores[i][0] < min_score:
            flags.append(f"low top score ({scores[i][0]:.2f})")
        if own is None and best_other_share >= other_share_max:
            flags.append(f"control leans on {best_other} ({best_other_share:.0%})")
        elif own and best_other_share >= other_share_max:
            flags.append(f"also depends on {best_other} ({best_other_share:.0%})")
        j = int(np.argmax(sim[i]))
        if sim[i, j] >= dup_cosine:
            flags.append(f"near-duplicate of {questions[j]['id']} ({sim[i, j]:.2f})")
        out.append(
            {
                "id": q["id"],
                "topic": own or "-",
                "top_score": float(scores[i][0]),
                "own_share": share[own] if own else np.nan,
                "best_other": best_other if best_other_share > 0 else "-",
                "other_share": best_other_share,
                "flags": "; ".join(flags),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check that questions retrieve what they are supposed to.")
    parser.add_argument("--snapshot", default="full")
    parser.add_argument("--embedder")
    parser.add_argument("--min-score", type=float, default=0.72)
    parser.add_argument("--out", help="also write the table to this CSV path")
    args = parser.parse_args(argv)

    rag = load_rag_config()
    ctx = load_context(args.embedder)
    topics, questions = load_topics(), load_questions()
    snap = resolve_by_name(ctx.df, args.snapshot, topics)
    qvecs = vectors([q["question"] for q in questions], kind="query", tag=ctx.tag, subprocess_ok=True)
    table = audit_questions(ctx, snap, questions, qvecs, topics, k=rag["k"], min_score=args.min_score)

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(table.round(2).to_string(index=False))
    per_topic = table[table["topic"] != "-"].groupby("topic").agg(
        n=("id", "size"), mean_own_share=("own_share", "mean"), mean_top_score=("top_score", "mean")
    )
    print("\nper topic:\n" + per_topic.round(2).to_string())
    flagged = table[table["flags"] != ""]
    print(f"\n{len(flagged)} of {len(table)} questions flagged; embedder={ctx.tag}, snapshot={snap.name}")
    if ctx.meta.get("backend") == "hash":
        print("NOTE: hash embedder - retrieval quality is not representative; run this with the real embedder.")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
