# Run log

The lab notebook of this project: every change that can move a result, and every run, in order.
`runs/` and `data/` are not in git, so **this file is the only versioned record of what was run and what it showed**.

**How to add an entry.** After a run: add a row to *Runs* (id, date, commit, answerer, prompt, rule, status) and a short
note under *Findings*. If you change anything that can change a number (metric, prompt, topic rules, questions, model
config), add a row to *Changes that affect results* first, with its commit. Each run's `run.json` records
`git_commit`, `git_dirty` (uncommitted files; since 2026-11-15), `prompt_version`, `abstain_rule`, `questions_hash` and the
embedder, so a row here can always be checked against the run itself.

Times are UTC (as in `run.json`). AEDT = UTC+11.

---

## Changes that affect results

| Date | Commit | Change | Effect on results | Runs before / after |
|---|---|---|---|---|
| 2026-10-15 → 11-14 | (pre-git) | Topic rules v1 → v2 (`configs/archive/topics.v1.yaml` → `configs/topics.yaml`); questions 20 → 72; `control_all_*` controls added | Different documents removed, different questions: **topic-removal numbers are not comparable across this line** | `20261015T034036Z` (v1) / all `local-*` (v2) |
| 2026-11-14 | `905bc72` | Local answerer (`--answerer ollama`), first commit of the repo | New model family; answers cached under provider `ollama` and never mixed with OpenAI's | — |
| 2026-11-14 | `beae559` | Ollama connection retries (10/30/60 s); tests isolated from `.env` | None on answers: only connection failures are retried, before the request reaches the model | `local-q35-v2`, `local-q3-14b-v1` ran with this code **uncommitted** (see note A) |
| 2026-11-15 | `b8b482c` | **Abstention rule `first-line` → `anywhere`** (`rag.ABSTAIN_RULE`): sentinel counted anywhere in the answer as a whole word | Abstention can only go **up** (0 abstentions lost, 84 gained over 3 runs). GPT runs unchanged (GPT always put the sentinel first). Abstention is stored at run time, so earlier runs were re-derived from cache as `*-r2` | `local-*` (first-line) / `local-*-r2` (anywhere) |
| 2026-11-15 | `01d60df` | `run.json` records `git_dirty`; report header prints the abstention rule and any uncommitted files | None on numbers; traceability only | from the next run on |

**Note A.** `run.json` before 2026-11-15 records only `HEAD`, not uncommitted changes. `local-q35-v1` ran at `905bc72` plus the
uncommitted test-isolation fix; `local-q35-v2` and `local-q3-14b-v1` also had the uncommitted retry code. Neither change
can alter an answer (one touches tests only, the other retries a request that never reached the model), and the `*-r2`
re-derivations confirm it: answers and retrieval are byte-identical to the originals.

---

## Runs

`status`: **current** = use these numbers · superseded = replaced by a re-derivation · pipeline = a pipeline check, not a finding.

| Run id | Created (UTC) | Commit | Answerer | Prompt | Abstain rule | Snapshots × questions | Status |
|---|---|---|---|---|---|---|---|
| `offline-check` | 2026-10-15 03:22 | none | extractive (hash-1024 embedder) | v1 | first-line | 21 × 20 | pipeline |
| `smoke-openai` | 2026-10-15 03:40 | none | `openai:gpt-6.1-sol:low` | v1 | first-line | 2 × 3 | pipeline |
| `20261015T034036Z` | 2026-10-15 04:02 | none | `openai:gpt-6.1-sol:low` | v1 | first-line | 21 × 20, topics **v1** | pilot (current for GPT; see below) |
| `offline-v2` | 2026-10-15 04:56 | none | extractive (hash-1024), 3 repeats | v1 | first-line | 37 × 72 | pipeline |
| `local-q35-smoke` | 2026-11-14 12:00 | `905bc72` | `ollama:qwen3.5:9b:no-think` | v1 | first-line | 4 × 9 | pipeline |
| `local-q35-v1` | 2026-11-14 13:22 | `905bc72`* | `ollama:qwen3.5:9b:no-think` | v1 | first-line | 37 × 72 | superseded by `-r2` |
| `local-q35-v2` | 2026-11-14 16:42 | `905bc72`* | `ollama:qwen3.5:9b:no-think` | v2 | first-line | 37 × 72 | superseded by `-r2` |
| `local-q3-14b-v1` | 2026-11-14 17:59 | `905bc72`* | `ollama:qwen3:14b:no-think` | v1 | first-line | 37 × 72 | superseded by `-r2` |
| `local-q35-v1-r2` | 2026-11-15 03:00 | `b8b482c` | `ollama:qwen3.5:9b:no-think` | v1 | anywhere | 37 × 72 | **current** |
| `local-q35-v2-r2` | 2026-11-15 03:00 | `b8b482c` | `ollama:qwen3.5:9b:no-think` | v2 | anywhere | 37 × 72 | **current** |
| `local-q3-14b-v1-r2` | 2026-11-15 03:01 | `b8b482c` | `ollama:qwen3:14b:no-think` | v1 | anywhere | 37 × 72 | **current** |

\* plus uncommitted changes, see note A. All `local-*` runs: embedder `bge-small`, k = 10, topics v2, `questions_hash`
`3e92c3bc`, temperature 0 (greedy, so no noise floor: repeated draws would be identical), cost $0.
`20261015T034036Z` used `questions_hash` `3b1a405e` (20 questions); `offline-v2` used `2f4e4c58` (72 questions, an earlier
version of the current file).

The three `*-r2` runs were made from cache with zero model calls (952 of 952 prompts cached each) and contain
byte-identical answers and retrieval to their originals; only the abstention rule differs.

---

## Findings (current runs)

All sanity checks PASS in every current run. Topic effects are against the fair control (`control_all_*`), pooled over 8
topics × 8 questions, 95% bootstrap interval. Positive gap = removing the topic hurt more than random removal.

| Run | Abstain gap | Answer-change gap | Abstain on full | Abstain `upto_2010` | Answered despite topic removed |
|---|---|---|---|---|---|
| `local-q35-v1-r2` (9B, v1) | 0.586 [0.47, 0.70] | 0.586 [0.47, 0.69] | 4.2% | 38.9% | 23 of 64 |
| `local-q35-v2-r2` (9B, v2) | 0.453 [0.33, 0.58] | 0.488 [0.38, 0.60] | 1.4% | 15.3% | 35 of 64 |
| `local-q3-14b-v1-r2` (14B, v1) | 0.359 [0.25, 0.48] | 0.396 [0.29, 0.51] | 1.4% | 18.1% | 41 of 64 |

- **H1 (topic effect) survives a reworded prompt and a second model.** `compare-runs`: same sign in every topic, pooled
  intervals overlap for 9B v1 vs v2 and 9B vs 14B. The answer-change gap is above 0 for all 8 topics in all 3 runs.
  Exception: sea ice on the 14B (0.25 vs 0.81 on the 9B; intervals do not overlap). Machine learning is weak everywhere
  (abstain gap ≤ 0.125).
- **The size of the effect depends on model and prompt** (9B v1 > 9B v2 > 14B). Report direction, not one number.
- **H3 (older corpus):** abstention rises as the corpus is cut back to 2010, in all 3 runs.
- **Topic removal vs year cut-off at equal evidence loss (9B v1):** with all top-10 papers gone, 70% abstain under topic
  removal vs 41% under a year cut-off; older papers on the same subject act as a fallback.
- **Candidate hallucinations (unread / unjudged):** after `remove_machine_learning` the 9B answered 7 of 8 ML questions from
  non-ML papers, sometimes attributing "deep learning" to papers whose abstracts never mention it.
- **Citation fidelity (9B only):** ~1% of answers cite an id not in the context, all copying errors: dropped digits
  (`2609.05557` → `2609.0557`), dropped archive prefixes (`[0510081]`, silently ignored by the parser, not counted), and the
  prompt's example id `0704.0001`. GPT: none in 363 answers.
- **Still undercounted:** refusals written in prose with no sentinel are not counted as abstention, by design. A rough
  pattern match puts them at 0.5% (9B v1), 2.6% (9B v2) and 0.2% (14B) of cells. Only the judge can settle them.

---

## Open decisions

1. Malformed citations: count as invalid (strict) or map back to the unique matching context id (lenient)?
2. q22 (Antarctic sea ice retrieves Arctic papers; the audit cannot see it) and the 12 questions `audit-questions` flags.
3. The judge needs an API model (OpenAI credits are exhausted): needed for grounding of the "answered despite removal"
   cells and for the prose-only refusals.
4. `compare-runs` cannot compute `change_gap` (it rebuilds the table without answer embeddings); read it from each report.
