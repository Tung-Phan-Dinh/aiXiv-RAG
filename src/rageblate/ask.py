"""Ask your own question: `aixiv-rag ask "What drives ENSO variability?"`.

    aixiv-rag ask "ocean heat uptake"                              # answer from the full corpus
    aixiv-rag ask "ocean heat uptake" --snapshot full --snapshot upto_2010   # same question, two corpora
    aixiv-rag ask                                                  # interactive: type questions, empty line quits
    aixiv-rag ask "..." --answerer extractive                      # offline, no API cost
    aixiv-rag ask "..." --answerer ollama                          # local model (configs/rag.yaml `ollama:`)

Exploration only: nothing is written to runs/. OpenAI answers still go through the on-disk cache,
so asking the same question on the same snapshot twice is free.
"""

from __future__ import annotations

import argparse
import textwrap

import numpy as np

from .ablate import build_answerer
from .config import load_rag_config
from .data import Context, load_context
from .embed import vectors
from .index import build_index
from .llm import IncompleteResponseError, LocalServerError, MissingAPIKeyError
from .rag import PROMPT_VERSION, PROMPTS, Answer, Answerer
from .retrieve import Hit, to_hits
from .snapshot import resolve_by_name


def answer_question(ctx: Context, index, question: str, qvec: np.ndarray, answerer: Answerer, k: int) -> tuple[list[Hit], Answer]:
    """Retrieve top-k from one snapshot's index, then answer from those hits. Pure, so tests can call it."""
    scores, rows = index.search(qvec.reshape(1, -1), k)
    hits = to_hits(ctx.df, scores[0], rows[0])
    return hits, answerer.answer(question, hits)


def _print_answer(snap_name: str, hits: list[Hit], ans: Answer, show_sources: bool) -> None:
    status = "ABSTAINED" if ans.abstained else "answered"
    print(f"\n--- {snap_name}  ({status}{', cached' if ans.cached else ''}) ---")
    for para in ans.text.splitlines():
        print(textwrap.fill(para, width=100) if para.strip() else "")
    if ans.invalid_cited_ids:
        print(f"  ! cites ids not in the context: {', '.join(ans.invalid_cited_ids)}")
    if show_sources:
        cited = set(ans.cited_ids)
        print("  sources (* = cited):")
        for h in hits:
            mark = "*" if h.arxiv_id in cited else " "
            print(f"   {mark}{h.rank:>2}  {h.score:.3f}  {h.year}  {h.arxiv_id:<18} {h.title[:70]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask your own question against one or more snapshots.")
    parser.add_argument("question", nargs="?", help="omit to type questions interactively")
    parser.add_argument("--snapshot", action="append", help="repeatable (default: full)")
    parser.add_argument("--answerer", choices=["openai", "ollama", "extractive"], default="openai")
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS), default=PROMPT_VERSION)
    parser.add_argument("--model", help="override openai.model from configs/rag.yaml")
    parser.add_argument("--effort", help="override openai.reasoning_effort from configs/rag.yaml")
    parser.add_argument("-k", type=int)
    parser.add_argument("--embedder")
    parser.add_argument("--no-sources", action="store_true", help="hide the retrieved documents")
    args = parser.parse_args(argv)

    rag = load_rag_config()
    k = args.k or rag["k"]
    ctx = load_context(args.embedder)
    answerer = build_answerer(args.answerer, rag, ctx.tag, args.prompt_version, args.model, args.effort)

    # A fresh index per snapshot (never a global index filtered afterwards), built once and reused.
    indexes = {}
    for name in args.snapshot or [rag["reference_snapshot"]]:
        snap = resolve_by_name(ctx.df, name)
        indexes[snap.name] = build_index(ctx, snap, rag["backend"])
        print(f"snapshot={snap.name} docs={snap.n_docs:,}")
    print(f"answerer={answerer.name} embedder={ctx.tag} k={k}")

    def ask_once(question: str) -> None:
        qvec = vectors([question], kind="query", tag=ctx.tag, subprocess_ok=True)[0]
        for name, index in indexes.items():
            hits, ans = answer_question(ctx, index, question, qvec, answerer, k)
            _print_answer(name, hits, ans, not args.no_sources)

    from openai import APIError  # lazy, as in llm.py

    # Failures worth a one-line message instead of a traceback.
    expected = (MissingAPIKeyError, IncompleteResponseError, LocalServerError, APIError)

    if args.question:
        try:
            ask_once(args.question)
        except expected as e:
            raise SystemExit(_explain(e)) from None
        return

    print("Type a question and press Enter (empty line or Ctrl-D quits).")
    while True:
        try:
            question = input("\n> ").strip()
            if not question:
                break
            ask_once(question)
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        except expected as e:  # keep the session alive; the next question may work
            print(f"\n{_explain(e)}")


def _explain(e: Exception) -> str:
    """Short, actionable message for the errors `ask` expects."""
    if isinstance(e, MissingAPIKeyError):
        return f"{e}\nAdd OPENAI_API_KEY to .env, or use --answerer extractive."
    if isinstance(e, (IncompleteResponseError, LocalServerError)):
        return str(e)
    # openai.APIError: .code/.type come from the API's error body
    if {getattr(e, "code", None), getattr(e, "type", None)} & {"insufficient_quota", "credit_balance_exhausted"}:
        return (
            "OpenAI: no API credits left. Add credits at https://platform.openai.com/settings/organization/billing/ "
            "or use --answerer extractive (free, offline)."
        )
    return f"OpenAI API error ({type(e).__name__}): {getattr(e, 'message', e)}"


if __name__ == "__main__":
    main()
