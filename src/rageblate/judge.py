"""An independent LLM judge for the two things the change metrics cannot see.

    aixiv-rag judge --run <id> --plan-only       # exact call count + cost estimate, spends nothing
    aixiv-rag judge --run <id> --yes             # writes runs/<id>/judge.parquet
    aixiv-rag judge --run <id> --export-labels 40 # a sample for YOU to hand-label (judge validation)
    aixiv-rag judge --run <id> --score-labels runs/<id>/judge_labels.csv

  support   for every answered cell: are the answer's claims stated in the retrieved context?
            supported | partial | unsupported
  abstain   for every abstention: was declining justified (the context really lacks the evidence),
            or did the context contain enough to answer?   justified | unjustified

This is NOT ground truth. It is a second model reading the same abstracts, blind to which snapshot
produced them. It answers "is this answer grounded in what was retrieved", which is the property RAG
promises; it cannot say whether the papers are right. Its own reliability is unmeasured until you
label a sample by hand (--export-labels / --score-labels) and look at the agreement. Use a judge model
different from (ideally stronger than) the answerer in configs/rag.yaml `judge:` to avoid a model
grading its own style.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_rag_config
from .data import load_corpus
from .llm import IncompleteResponseError, MissingAPIKeyError, OpenAILLM
from .paths import RUNS
from .plan import build_plan, execute
from .rag import format_context
from .retrieve import to_hits

SUPPORT_VERDICTS = ("supported", "partial", "unsupported")
ABSTAIN_VERDICTS = ("justified", "unjustified")
SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "justified": 1.0, "unjustified": 0.0}

SUPPORT_SYSTEM = """You are a strict grader of retrieval-augmented answers. You are given a question, context documents (arXiv abstracts) and an answer written from them.
Decide whether the answer's claims are supported by the context documents. Use ONLY the context, not your own knowledge.
- supported: every substantive claim is stated in the context, and any citations point to documents that say it.
- partial: some claims are supported, others are not stated in the context or are cited to the wrong document.
- unsupported: the main claims are not in the context, or the answer contradicts or ignores what the context says.
Reply with ONLY a JSON object: {"verdict": "supported" | "partial" | "unsupported", "reason": "<one sentence>"}"""

ABSTAIN_SYSTEM = """You are a strict grader of abstentions by retrieval-augmented systems. The system was given a question and context documents (arXiv abstracts) and declined to answer because it judged the context insufficient.
Decide whether declining was justified. Use ONLY the context, not your own knowledge.
- justified: the context does not contain enough information to answer the question substantively.
- unjustified: the context contains enough information to give a substantive, supported answer.
Reply with ONLY a JSON object: {"verdict": "justified" | "unjustified", "reason": "<one sentence>"}"""


def judge_prompt(kind: str, question: str, answer: str, hits) -> tuple[str, str]:
    system = ABSTAIN_SYSTEM if kind == "abstain" else SUPPORT_SYSTEM
    return system, f"Question: {question}\n\nContext documents:\n\n{format_context(hits)}\n\nAnswer:\n{answer}"


def parse_verdict(text: str, kind: str) -> tuple[str, str]:
    """(verdict, reason). Anything unparseable becomes 'unparsed' rather than an exception: a judge hiccup
    must not lose a paid run, and the unparsed share is reported."""
    allowed = ABSTAIN_VERDICTS if kind == "abstain" else SUPPORT_VERDICTS
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).strip().lower()
            if v in allowed:
                return v, str(obj.get("reason", "")).strip()
        except ValueError:
            pass
    return "unparsed", text.strip()[:200]


class _Prompter:
    """Adapter so plan.build_plan can price judge calls: payload = (kind, question, answer)."""

    def __init__(self, llm: OpenAILLM):
        self.llm = llm

    def prompt(self, payload, hits):
        kind, question, answer = payload
        return judge_prompt(kind, question, answer, hits)


def build_judge_llm(rag_cfg: dict, model: str | None, effort: str | None) -> OpenAILLM:
    c = {**rag_cfg["openai"], **(rag_cfg.get("judge") or {})}
    return OpenAILLM(
        model or c["model"],
        c.get("max_output_tokens", 3000),
        c.get("temperature"),
        effort if effort is not None else c.get("reasoning_effort"),
    )


def judge_items(res: pd.DataFrame, corpus: pd.DataFrame, questions: dict[str, str], snapshots: list[str] | None):
    """[(snapshot, question_id, kind, question, answer, hits)] for primary-draw rows with an answer."""
    from .metrics import primary

    res = primary(res)
    if snapshots:
        res = res[res["snapshot"].isin(snapshots)]
    pos = {a: i for i, a in enumerate(corpus["arxiv_id"])}
    out = []
    for r in res.itertuples(index=False):
        if r.answer is None:
            continue
        rows = np.array([pos[i] for i in r.retrieved_ids])
        hits = to_hits(corpus, np.asarray(r.retrieved_scores), rows)
        out.append((r.snapshot, r.question_id, "abstain" if r.abstained else "support", questions[r.question_id], r.answer, hits))
    return out


def run_judge(llm: OpenAILLM, items, plan_only=False, yes=False, workers=4, pricing=None, confirm_above=60) -> pd.DataFrame | None:
    prompter = _Prompter(llm)
    plan = build_plan(prompter, (((k, q, a), hits, 0) for _, _, k, q, a, hits in items), pricing)
    print(plan.describe(llm.model))
    if plan_only:
        return None
    if plan.n_new > confirm_above and not yes:
        raise SystemExit(f"{plan.n_new} paid judge calls. Re-run with --yes to proceed (or --plan-only to just look).")
    execute(llm, plan.jobs, workers)
    rows = []
    for snap, qid, kind, q, a, hits in items:
        system, user = judge_prompt(kind, q, a, hits)
        res = llm.complete(system, user)
        verdict, reason = parse_verdict(res.text, kind)
        rows.append({
            "snapshot": snap, "question_id": qid, "kind": kind, "verdict": verdict,
            "score": SCORE.get(verdict, np.nan), "reason": reason, "judge": f"{llm.model}:{llm.reasoning_effort or 'default'}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- reporting


def report_section(judgments: pd.DataFrame, df: pd.DataFrame, meta: dict, out_dir: Path) -> list[str]:
    from .metrics import snapshot_kind
    from .report import md_table

    kinds = {n: snapshot_kind(s["cfg"]) for n, s in meta["snapshots"].items()}
    j = judgments.copy()
    j["snapshot_kind"] = j["snapshot"].map(kinds)
    ref = meta["reference_snapshot"]
    removed = {n: set(s["removed_topics"]) for n, s in meta["snapshots"].items()}
    topic = df.drop_duplicates("question_id").set_index("question_id")["question_topic"]
    j["targeted"] = [t in removed.get(s, set()) for s, t in zip(j["snapshot"], j["question_id"].map(topic))]

    def summarize(g):
        sup, ab = g[g["kind"] == "support"], g[g["kind"] == "abstain"]
        return pd.Series({
            "n_answered": len(sup),
            "supported": (sup["verdict"] == "supported").mean() if len(sup) else np.nan,
            "partial": (sup["verdict"] == "partial").mean() if len(sup) else np.nan,
            "unsupported": (sup["verdict"] == "unsupported").mean() if len(sup) else np.nan,
            "n_abstained": len(ab),
            "abstain_justified": (ab["verdict"] == "justified").mean() if len(ab) else np.nan,
            "unparsed": (g["verdict"] == "unparsed").mean(),
        })

    by_snap = j.groupby("snapshot", sort=False).apply(summarize, include_groups=False).reset_index()
    by_kind = j[j["snapshot"] != ref].groupby("snapshot_kind").apply(summarize, include_groups=False).reset_index()
    targeted = j[j["targeted"]].pipe(lambda g: summarize(g).to_frame().T.assign(group="targeted (topic removed)"))
    other = j[(~j["targeted"]) & (j["snapshot"] != ref)].pipe(lambda g: summarize(g).to_frame().T.assign(group="not targeted"))
    tgt = pd.concat([targeted, other]).set_index("group").reset_index()
    by_snap.to_csv(out_dir / "judge_by_snapshot.csv", index=False)
    by_kind.to_csv(out_dir / "judge_by_kind.csv", index=False)
    judge_name = ", ".join(sorted(j["judge"].unique()))
    lines = [
        "## 5. Answer quality (LLM judge)",
        (f"Judge: `{judge_name}`, blind to which snapshot produced each cell. **Grounding, not truth**: it checks whether the answer "
         "is stated in the retrieved abstracts (and whether each abstention was warranted), not whether the papers are right. "
         "Its own accuracy is only known if you label a sample by hand (`--export-labels`, `--score-labels`)."),
        "",
        ("Targeted questions (their topic was removed) vs the rest. The first row is the fabrication question: of the "
         "answers still given, how many are unsupported?"),
        "",
        md_table(tgt),
        "",
        "By kind of snapshot:",
        "",
        md_table(by_kind),
        "",
        "By snapshot:",
        "",
        md_table(by_snap),
        "",
    ]
    labels = Path(out_dir).parent / "judge_agreement.json"
    if labels.exists():
        a = json.loads(labels.read_text())
        lines += [f"Judge vs your hand labels: {a['agreement']:.0%} agreement, Cohen's kappa {a['kappa']:.2f} on n = {a['n']}.", ""]
    else:
        lines += ["Judge accuracy is **unvalidated**: no hand-labelled sample scored yet.", ""]
    return lines


# --------------------------------------------------------------------------------------- validation


def export_labels(judgments: pd.DataFrame, items, n: int, path: Path, seed: int = 0) -> int:
    """Random sample of judged cells for hand labelling, half support / half abstention where both exist.
    The `your_verdict` column is left empty for you."""
    lookup = {(s, q): (k, question, a, hits) for s, q, k, question, a, hits in items}
    keyed = judgments[judgments["verdict"] != "unparsed"].drop_duplicates(["snapshot", "question_id"])
    parts = [g.sample(min(len(g), max(1, n // 2)), random_state=seed) for _, g in keyed.groupby("kind")]
    rows = []
    for r in pd.concat(parts).itertuples(index=False) if parts else []:
        k, q, a, hits = lookup[(r.snapshot, r.question_id)]
        rows.append({
            "snapshot": r.snapshot, "question_id": r.question_id, "kind": k, "question": q, "answer": a,
            "context": format_context(hits), "judge_verdict": r.verdict,
            "your_verdict": "", "allowed": "/".join(ABSTAIN_VERDICTS if k == "abstain" else SUPPORT_VERDICTS),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def score_labels(path: Path) -> dict:
    lab = pd.read_csv(path).dropna(subset=["your_verdict"])
    lab = lab[lab["your_verdict"].astype(str).str.strip() != ""]
    mine, theirs = lab["your_verdict"].str.strip().str.lower().tolist(), lab["judge_verdict"].tolist()
    return {
        "n": len(lab), "agreement": float(np.mean([x == y for x, y in zip(mine, theirs)])) if len(lab) else float("nan"),
        "kappa": cohens_kappa(mine, theirs),
    }


# --------------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM-judge the answers of one run (support + abstention justification).")
    parser.add_argument("--run", required=True)
    parser.add_argument("--snapshots", nargs="*", help="only these snapshots (default: all in the run)")
    parser.add_argument("--model", help="judge model (default: configs/rag.yaml judge.model, else the answerer's)")
    parser.add_argument("--effort", help="reasoning effort override")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm more than 60 uncached paid calls")
    parser.add_argument("--export-labels", type=int, metavar="N", help="write ~N judged cells to judge_labels.csv for hand-labelling")
    parser.add_argument("--score-labels", metavar="CSV", help="score your filled-in labels against the judge")
    args = parser.parse_args(argv)

    run_dir = Path(args.run) if Path(args.run).is_dir() else RUNS / args.run
    if args.score_labels:
        out = score_labels(Path(args.score_labels))
        (run_dir / "judge_agreement.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"judge vs you: {out['agreement']:.0%} agreement, kappa {out['kappa']:.2f}, n={out['n']}  (saved; rebuild the report)")
        return

    from .config import load_questions

    rag = load_rag_config()
    res = pd.read_parquet(run_dir / "results.parquet")
    qtext = {q["id"]: q["question"] for q in load_questions()}
    missing = set(res["question_id"]) - set(qtext)
    if missing:
        raise SystemExit(f"questions {sorted(missing)[:5]} are in the run but not in configs/questions.yaml")
    items = judge_items(res, load_corpus(), qtext, args.snapshots)
    if not items:
        raise SystemExit("this run has no answers to judge (retrieval-only run?)")
    llm = build_judge_llm(rag, args.model, args.effort)
    path = run_dir / "judge.parquet"
    try:
        new = run_judge(llm, items, args.plan_only, args.yes, args.workers, rag.get("pricing_usd_per_mtok"))
    except (MissingAPIKeyError, IncompleteResponseError) as e:
        raise SystemExit(f"error: {e}") from None
    if new is None:
        return
    if path.exists():  # merge: a later partial run adds to, and refreshes, earlier verdicts
        old = pd.read_parquet(path)
        new = pd.concat([old[~old.set_index(["snapshot", "question_id"]).index.isin(new.set_index(["snapshot", "question_id"]).index)], new])
    new.to_parquet(path, index=False)
    print(f"wrote {path}  ({len(new)} judgments; unparsed: {(new['verdict'] == 'unparsed').sum()})")
    if args.export_labels:
        n = export_labels(new, items, args.export_labels, run_dir / "judge_labels.csv")
        print(f"wrote {run_dir / 'judge_labels.csv'} ({n} rows): fill `your_verdict`, then --score-labels")
    print(f"next: aixiv-rag report --run {run_dir.name}")


if __name__ == "__main__":
    main()
