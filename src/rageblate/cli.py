"""`aixiv-rag <command>` - one entry point for the whole pipeline.

    aixiv-rag ingest      raw arXiv dump -> data/corpus.parquet
    aixiv-rag topics      how many docs each topic rule matches
    aixiv-rag embed       corpus -> data/embeddings/<tag>/
    aixiv-rag check       verify corpus/embeddings alignment
    aixiv-rag snapshots   list snapshots and their sizes
    aixiv-rag retrieve    query one snapshot
    aixiv-rag ask         answer your own question (one or more snapshots, or interactive)
    aixiv-rag audit-questions  check that questions retrieve what they should (no API cost)
    aixiv-rag ablate      run questions x snapshots
    aixiv-rag judge       LLM-judge support and abstention justification for a run
    aixiv-rag compare-runs  does the effect survive a different prompt/model/embedder?
    aixiv-rag report      metrics + figures for a run
"""

from __future__ import annotations

import sys


def _ingest(rest: list[str]) -> None:
    from . import ingest

    ingest.main()


def _check(rest: list[str]) -> None:
    from .data import load_context

    ctx = load_context(rest[0] if rest else None)
    print(f"OK  corpus rows={len(ctx.df):,}  embeddings={ctx.emb.shape}  embedder={ctx.tag}  model={ctx.meta.get('model')}")
    print(f"    years {ctx.df['year'].min()}-{ctx.df['year'].max()}  device={ctx.meta.get('device')}  created={ctx.meta.get('created_utc')}")


def _dispatch():
    from . import ablate, ask, audit, compare, embed, judge, report, retrieve, snapshot, topics

    return {
        "ingest": _ingest,
        "topics": topics.main,
        "embed": embed.main,
        "check": _check,
        "snapshots": snapshot.main,
        "retrieve": retrieve.main,
        "ask": ask.main,
        "audit-questions": audit.main,
        "ablate": ablate.main,
        "judge": judge.main,
        "compare-runs": compare.main,
        "report": report.main,
    }


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    commands = _dispatch()
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in commands:
        print(__doc__)
        raise SystemExit(0 if not argv or argv[0] in {"-h", "--help"} else 2)
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
