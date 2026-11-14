"""Query a snapshot: `aixiv-rag retrieve --snapshot upto_2010 --query "ocean heat uptake"`."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_rag_config
from .data import load_context
from .embed import vectors
from .index import build_index
from .snapshot import resolve_by_name


@dataclass(frozen=True)
class Hit:
    rank: int  # 1-based
    row: int  # row in the corpus parquet
    arxiv_id: str
    year: int
    score: float
    title: str
    abstract: str


def to_hits(df: pd.DataFrame, scores: np.ndarray, rows: np.ndarray) -> list[Hit]:
    """Turn one query's (scores, corpus rows) into Hit objects."""
    return [
        Hit(
            rank=i + 1,
            row=int(r),
            arxiv_id=str(df["arxiv_id"].iat[r]),
            year=int(df["year"].iat[r]),
            score=float(s),
            title=str(df["title"].iat[r]),
            abstract=str(df["abstract"].iat[r]),
        )
        for i, (s, r) in enumerate(zip(scores, rows))
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k documents from a snapshot.")
    parser.add_argument("--snapshot", default="full")
    parser.add_argument("--query", action="append", required=True, help="repeatable")
    parser.add_argument("-k", type=int)
    parser.add_argument("--backend", choices=["faiss", "numpy"])
    parser.add_argument("--embedder")
    args = parser.parse_args(argv)

    rag = load_rag_config()
    k, backend = args.k or rag["k"], args.backend or rag["backend"]
    ctx = load_context(args.embedder)
    snap = resolve_by_name(ctx.df, args.snapshot)
    qvecs = vectors(args.query, kind="query", tag=ctx.tag, subprocess_ok=True)
    scores, rows = build_index(ctx, snap, backend).search(qvecs, k)

    print(f"snapshot={snap.name} docs={snap.n_docs:,} embedder={ctx.tag} backend={backend}")
    for qi, query in enumerate(args.query):
        print(f"\nQ: {query}")
        for h in to_hits(ctx.df, scores[qi], rows[qi]):
            print(f"  {h.rank:>2}  {h.score:.3f}  {h.year}  {h.arxiv_id:<18} {h.title[:80]}")


if __name__ == "__main__":
    main()
