"""Know what a run will cost *before* it spends anything, and execute the paid calls in parallel.

A run is (snapshot x question x sample) cells. Many cells share one prompt (a snapshot that does
not change a question's top-k gives the same prompt as `full`), and finished prompts are in the
on-disk cache, so the number of calls that actually cost money is usually far smaller than the
number of cells. `build_plan` counts exactly that; `execute` makes those calls concurrently so the
sequential pass in `ablate.run` is then served entirely from cache.

The dollar figure is an estimate: token counts come from the mean of responses already in the
cache for the same model and effort (so it improves as you run), prices come from
configs/rag.yaml `pricing_usd_per_mtok` and are yours to keep current.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 4.0
GUESS_OUTPUT_TOKENS = 600  # only used when the cache has nothing to learn from


@dataclass
class Plan:
    n_cells: int
    n_unique: int
    n_cached: int
    jobs: list[tuple[str, str, int]] = field(default_factory=list)  # (system, user, sample) still to pay for
    est_input_tokens: float = 0.0
    est_output_tokens: float = 0.0
    est_cost_usd: float | None = None
    basis: str = ""

    @property
    def n_new(self) -> int:
        return len(self.jobs)

    def describe(self, model: str = "") -> str:
        head = f"plan: {self.n_cells} cells -> {self.n_unique} distinct prompts"
        lines = [f"{head} ({self.n_unique - self.n_new} cached, {self.n_new} to pay for)"]
        if self.n_new:
            cost = f"~${self.est_cost_usd:.2f}" if self.est_cost_usd is not None else "cost unknown (no price for model)"
            lines.append(
                f"      {model} estimate: {self.est_input_tokens / 1e3:,.0f}k input + "
                f"{self.est_output_tokens / 1e3:,.0f}k output tokens = {cost}   [{self.basis}]"
            )
        return "\n".join(lines)


def usage_stats(cache_dir: Path, model: str, reasoning_effort: str | None) -> tuple[float, float, int]:
    """Mean (input_tokens, output_tokens) over cached responses of this model+effort, and how many."""
    tot_in = tot_out = n = 0
    for p in Path(cache_dir).glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rec.get("model") != model or rec.get("reasoning_effort") != reasoning_effort:
            continue
        u = rec.get("usage") or {}
        if "input_tokens" in u and "output_tokens" in u:
            tot_in += u["input_tokens"]
            tot_out += u["output_tokens"]
            n += 1
    return (tot_in / n, tot_out / n, n) if n else (0.0, 0.0, 0)


def build_plan(answerer, calls, pricing: dict | None = None) -> Plan:
    """`calls` is an iterable of (question, hits, sample). Answerers without a paid backend cost 0."""
    keys: dict[str, tuple[str, str, int]] = {}
    n_cells = 0
    llm = getattr(answerer, "llm", None)
    for question, hits, sample in calls:
        n_cells += 1
        if llm is None or not hasattr(answerer, "prompt"):
            continue
        system, user = answerer.prompt(question, hits)
        keys.setdefault(llm.key(system, user, sample), (system, user, sample))
    if llm is None:
        return Plan(n_cells, 0, 0, basis="no paid calls")

    jobs = [job for job in keys.values() if not llm.is_cached(*job)]
    plan = Plan(n_cells, len(keys), len(keys) - len(jobs), jobs)
    if not jobs:
        plan.est_cost_usd, plan.basis = 0.0, "everything cached"
        return plan

    mean_in, mean_out, n_seen = usage_stats(llm.cache_dir, llm.model, llm.reasoning_effort)
    if n_seen:
        plan.est_input_tokens, plan.est_output_tokens = mean_in * len(jobs), mean_out * len(jobs)
        plan.basis = f"mean tokens of {n_seen} cached calls"
    else:
        plan.est_input_tokens = sum(len(s) + len(u) for s, u, _ in jobs) / CHARS_PER_TOKEN
        plan.est_output_tokens = GUESS_OUTPUT_TOKENS * len(jobs)
        plan.basis = "rough guess: nothing cached to learn from"
    price = (pricing or {}).get(llm.model)
    if not getattr(llm, "paid", True):  # local model: tokens still estimated (they set the run time), no money
        plan.est_cost_usd, plan.basis = 0.0, f"{plan.basis}; local model, no API cost"
    elif price:
        plan.est_cost_usd = (plan.est_input_tokens * price["input"] + plan.est_output_tokens * price["output"]) / 1e6
    return plan


def execute(llm, jobs: list[tuple[str, str, int]], workers: int = 4, progress_every: int = 25) -> None:
    """Run the paid calls concurrently; stop at the first failure (finished calls stay cached)."""
    if not jobs:
        return
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(llm.complete, *job) for job in jobs]
        try:
            for f in as_completed(futures):
                f.result()
                done += 1
                if done % progress_every == 0 or done == len(jobs):
                    print(f"  paid calls: {done}/{len(jobs)}", flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            print(f"  stopped after {done}/{len(jobs)} calls; completed ones are cached, re-run to continue", flush=True)
            raise
