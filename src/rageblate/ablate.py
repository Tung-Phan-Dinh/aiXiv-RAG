"""The ablation runner: same questions x many corpus snapshots -> one results table.

    aixiv-rag ablate --answerer openai                  # all snapshots in configs/snapshots
    aixiv-rag ablate --answerer extractive              # offline, no API key
    aixiv-rag ablate --answerer none --snapshots full upto_2010   # retrieval only

    aixiv-rag ablate --repeats 5                        # 5 answer draws per prompt on the reference snapshot
    aixiv-rag ablate --plan-only                        # print the exact call count and cost estimate, spend nothing

Output: runs/<run_id>/{results.parquet, run.json, snapshot_rows/<name>.npy}
`sample` in results.parquet is the draw index: 0 is the primary answer every metric uses; samples >= 1 exist
only to measure the model's own run-to-run noise.
`run()` is pure (no global paths) so tests can call it on a tiny in-memory corpus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    list_snapshot_names,
    load_questions,
    load_rag_config,
    load_topics,
    stable_hash,
)
from .data import Context, load_context
from .embed import vectors
from .index import build_index
from .llm import IncompleteResponseError, LocalServerError, MissingAPIKeyError, OllamaLLM, OpenAILLM
from .paths import RUNS
from .plan import build_plan, execute
from .rag import ABSTAIN_RULE, PROMPT_VERSION, PROMPTS, Answerer, ExtractiveAnswerer, OpenAIAnswerer
from .retrieve import to_hits
from .snapshot import Snapshot, resolve_by_name

CONFIRM_ABOVE = 60  # require --yes when a run would make more than this many *uncached* paid calls


def retrieve_all(ctx: Context, snapshots: list[Snapshot], qvecs: np.ndarray, k: int, backend: str) -> dict:
    """{snapshot name: (scores, rows)} - one fresh exact index per snapshot."""
    return {s.name: build_index(ctx, s, backend).search(qvecs, k) for s in snapshots}


def sample_count(snap: Snapshot, repeats: int, repeat_snapshots: set[str] | None) -> int:
    """Draws per (question) in this snapshot: `repeats` if it is a replicate snapshot, else 1."""
    return repeats if (repeat_snapshots is None or snap.name in repeat_snapshots) else 1


def iter_cells(ctx, snapshots, questions, retrieved, repeats=1, repeat_snapshots=None):
    """Yield (snapshot, question index, sample, hits) for every cell of the run, in run order."""
    for snap in snapshots:
        scores, rows = retrieved[snap.name]
        for qi in range(len(questions)):
            hits = to_hits(ctx.df, scores[qi], rows[qi])
            for sample in range(sample_count(snap, repeats, repeat_snapshots)):
                yield snap, qi, sample, hits


def run(
    ctx: Context,
    snapshots: list[Snapshot],
    questions: list[dict],
    qvecs: np.ndarray,
    answerer: Answerer | None,
    k: int = 10,
    backend: str = "faiss",
    out_dir: Path | None = None,
    meta_extra: dict | None = None,
    repeats: int = 1,
    repeat_snapshots: set[str] | None = None,
    retrieved: dict | None = None,
    paid_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Retrieve (and optionally answer) every question in every snapshot.

    `repeats` > 1 draws that many answers per prompt in the `repeat_snapshots` (default: all) to
    measure the model's own noise; retrieval is deterministic so only answers differ across samples.
    """
    records = []
    retrieved = retrieved or retrieve_all(ctx, snapshots, qvecs, k, backend)
    for snap, qi, sample, hits in iter_cells(ctx, snapshots, questions, retrieved, repeats, repeat_snapshots):
        q = questions[qi]
        ans = answerer.answer(q["question"], hits, sample) if answerer else None
        cached = ans.cached if ans else None
        if ans and paid_keys is not None and hasattr(answerer, "prompt"):
            # calls were pre-paid concurrently, so every read here is a cache hit; "cached" must mean
            # "was already cached before this run", i.e. not one of the calls this run paid for
            cached = answerer.llm.key(*answerer.prompt(q["question"], hits), sample) not in paid_keys
        records.append(
            {
                "snapshot": snap.name,
                "question_id": q["id"],
                "sample": sample,
                "question_topic": q.get("topic"),
                "k": len(hits),
                "n_docs": snap.n_docs,
                "retrieved_ids": [h.arxiv_id for h in hits],
                "retrieved_years": [h.year for h in hits],
                "retrieved_scores": [round(h.score, 6) for h in hits],
                "top_score": hits[0].score if hits else None,
                "answer": ans.text if ans else None,
                "abstained": ans.abstained if ans else None,
                "cited_ids": ans.cited_ids if ans else None,
                "invalid_cited_ids": ans.invalid_cited_ids if ans else None,
                "answerer": ans.answerer if ans else "none",
                "answer_cached": cached,
            }
        )
    results = pd.DataFrame(records)

    if out_dir is not None:
        out_dir = Path(out_dir)
        (out_dir / "snapshot_rows").mkdir(parents=True, exist_ok=True)
        for snap in snapshots:
            np.save(out_dir / "snapshot_rows" / f"{snap.name}.npy", snap.rows)
        results.to_parquet(out_dir / "results.parquet", index=False)
        meta = {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "answerer": answerer.name if answerer else "none",
            "k": k,
            "backend": backend,
            "embedder": {"tag": ctx.tag, **ctx.meta},
            "questions_hash": stable_hash(questions),
            "prompt_version": getattr(answerer, "prompt_version", PROMPT_VERSION),
            "abstain_rule": ABSTAIN_RULE,
            "repeats": repeats,
            "repeat_snapshots": sorted(repeat_snapshots) if repeat_snapshots is not None else None,
            "snapshots": {
                s.name: {
                    "n_docs": s.n_docs,
                    "cfg": s.cfg,
                    "cfg_hash": s.cfg_hash,
                    "ids_hash": s.ids_hash,
                    "removed_topics": s.removed_topics,
                }
                for s in snapshots
            },
            "git_commit": _git_commit(),
            "versions": _versions(),
            **(meta_extra or {}),
        }
        (out_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return results


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - no git / no commits yet is fine
        return None


def _versions() -> dict:
    found = {}
    for pkg in ("numpy", "pandas", "pyarrow", "faiss-cpu", "openai", "sentence-transformers", "torch"):
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return found


def build_answerer(
    kind: str,
    rag_cfg: dict,
    tag: str,
    prompt_version: str = PROMPT_VERSION,
    model: str | None = None,
    effort: str | None = None,
) -> Answerer | None:
    if kind == "none":
        return None
    if kind == "extractive":
        ex = rag_cfg["extractive"]
        return ExtractiveAnswerer(ex["min_score"].get(tag, 0.0), ex["n_sentences"])
    if kind == "ollama":  # local model; --effort does not apply (thinking is `ollama.think`)
        o = rag_cfg["ollama"]
        llm = OllamaLLM(
            model or o["model"],
            host=o.get("host"),
            num_ctx=o["num_ctx"],
            max_output_tokens=o["max_output_tokens"],
            temperature=o.get("temperature", 0.0),
            think=o.get("think"),
            seed=o.get("seed", 0),
        )
        return OpenAIAnswerer(llm, prompt_version)
    c = rag_cfg["openai"]
    llm = OpenAILLM(
        model or c["model"],
        c["max_output_tokens"],
        c.get("temperature"),
        effort if effort is not None else c.get("reasoning_effort"),
    )
    return OpenAIAnswerer(llm, prompt_version)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the corpus-ablation experiment.")
    parser.add_argument("--snapshots", nargs="*", help="names (default: every config in configs/snapshots)")
    parser.add_argument("--answerer", choices=["openai", "ollama", "extractive", "none"], default="openai")
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS), default=PROMPT_VERSION, help="v2 = reworded prompt, different abstention sentinel")
    parser.add_argument("--model", help="override openai.model from configs/rag.yaml (robustness check)")
    parser.add_argument("--effort", help="override openai.reasoning_effort")
    parser.add_argument("--k", type=int)
    parser.add_argument("--backend", choices=["faiss", "numpy"])
    parser.add_argument("--embedder")
    parser.add_argument("--run-id")
    parser.add_argument("--limit-questions", type=int, help="use only the first N questions (smoke test)")
    parser.add_argument("--repeats", type=int, default=1, help="answer draws per prompt (>1 measures the noise floor)")
    parser.add_argument(
        "--repeat-snapshots", nargs="*", help="snapshots that get the repeats (default: only the reference snapshot)"
    )
    parser.add_argument("--workers", type=int, help="concurrent model calls (default: 4 for openai, 1 for a local ollama GPU)")
    parser.add_argument("--plan-only", action="store_true", help="print the call count and cost estimate, then stop")
    parser.add_argument("--yes", action="store_true", help=f"confirm a run with more than {CONFIRM_ABOVE} uncached paid calls")
    args = parser.parse_args(argv)

    rag = load_rag_config()
    k, backend = args.k or rag["k"], args.backend or rag["backend"]
    ctx = load_context(args.embedder)
    topics = load_topics()
    names = args.snapshots or list_snapshot_names()
    if rag["reference_snapshot"] not in names:
        names = [rag["reference_snapshot"], *names]  # metrics are relative to the reference
    snapshots = [resolve_by_name(ctx.df, n, topics) for n in names]
    questions = load_questions()[: args.limit_questions]

    repeats = max(1, args.repeats)
    repeat_snapshots = set(args.repeat_snapshots) if args.repeat_snapshots is not None else {rag["reference_snapshot"]}
    unknown = repeat_snapshots - {s.name for s in snapshots}
    if unknown:
        raise SystemExit(f"--repeat-snapshots not in this run: {sorted(unknown)}")

    qvecs = vectors([q["question"] for q in questions], kind="query", tag=ctx.tag, subprocess_ok=True)
    answerer = build_answerer(args.answerer, rag, ctx.tag, args.prompt_version, args.model, args.effort)
    retrieved = retrieve_all(ctx, snapshots, qvecs, k, backend)

    cells = list(iter_cells(ctx, snapshots, questions, retrieved, repeats, repeat_snapshots))
    print(
        f"{len(snapshots)} snapshots x {len(questions)} questions, {repeats} draw(s) on {sorted(repeat_snapshots)} "
        f"= {len(cells)} cells | answerer={args.answerer} embedder={ctx.tag} k={k}"
    )
    plan = build_plan(answerer, ((questions[qi]["question"], hits, s) for _, qi, s, hits in cells), rag.get("pricing_usd_per_mtok"))
    if answerer is not None:
        print(plan.describe(getattr(getattr(answerer, "llm", None), "model", "")))
    if args.plan_only:
        return
    paid = getattr(getattr(answerer, "llm", None), "paid", False)
    if paid and plan.n_new > CONFIRM_ABOVE and not args.yes:
        raise SystemExit(f"{plan.n_new} paid calls. Re-run with --yes to proceed (or --plan-only to just look).")

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; choose another --run-id (runs are never overwritten)")
    try:
        if plan.jobs:
            execute(answerer.llm, plan.jobs, args.workers or (4 if paid else 1))
        results = run(
            ctx, snapshots, questions, qvecs, answerer, k, backend, out_dir,
            {"run_id": run_id, "reference_snapshot": rag["reference_snapshot"]},
            repeats=repeats, repeat_snapshots=repeat_snapshots, retrieved=retrieved,
            paid_keys={answerer.llm.key(*j) for j in plan.jobs} if plan.jobs else None,
        )
    except (MissingAPIKeyError, IncompleteResponseError, LocalServerError) as e:
        raise SystemExit(f"error: {e}") from None
    print(f"wrote {out_dir}  ({len(results)} rows)")
    print(f"next: aixiv-rag report --run {run_id}")


if __name__ == "__main__":
    main()
