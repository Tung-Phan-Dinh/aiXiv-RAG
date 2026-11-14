# AGENTS.md - runbook for coding agents

Read this before changing anything. `README.md` explains the project; this file says how to operate it.

## What this repo is

A from-scratch RAG corpus-ablation harness over arXiv `physics.geo-ph` / `physics.ao-ph` abstracts.
Same questions, many corpus snapshots, four metrics. The owner (Andre) is preparing for a CSIRO research
internship on how RAG responses change with the underlying corpus, so **experimental validity beats features**.

## Invariants - never violate these

1. `data/corpus.parquet` is immutable. Snapshots are filters over it (`snapshot.py`), never edited copies.
2. Embeddings must match the corpus row-for-row. `data.load_embeddings` enforces it via an id-order hash;
   never bypass or loosen that check. Re-embed instead.
3. Do not change the embedding model, `doc_template` or query prefix mid-project. That invalidates every cached result.
4. Always build a **fresh** index per snapshot (`index.build_index`). Never search a global index and filter.
5. torch and FAISS must not be imported in the same process (macOS OpenMP crash). `embed.py` imports torch lazily;
   `ablate.py` / `retrieve.py` get query vectors through `embed.vectors(..., subprocess_ok=True)`. `report.py`/`metrics.py`
   must not import FAISS. Do not "fix" this with `KMP_DUPLICATE_LIB_OK`.
6. Topic rules live only in `configs/topics.yaml` and are hashed into snapshots. No ad-hoc topic filtering in code.
7. Runs are never overwritten (`runs/<run_id>`); pick a new `--run-id`.
8. Never commit `data/`, `runs/` or `.env`. Never print or log `OPENAI_API_KEY`.
9. Do not add LangChain / LlamaIndex or other RAG frameworks.

## Environment

- Python 3.14, managed with `uv`. Run everything as `uv run ...`.
- A `.venv` is OS-specific. If you work from a different OS than the owner's Mac, use your own environment
  (`UV_PROJECT_ENVIRONMENT=/somewhere/else uv sync`) and do not touch the shared `.venv`.
- Secrets: `.env` with `OPENAI_API_KEY=...` (loaded by `config.load_dotenv`). The Kaggle CLI auth is separate: `uv run kaggle auth login`.
- Local answerer: `--answerer ollama` (on `ablate` and `ask`) calls Ollama on the owner's home PC over Tailscale; host in `.env` as
  `RAGEBLATE_OLLAMA_HOST`, model/context in `configs/rag.yaml` `ollama:`. Free, so no `--yes` needed. It is a *different model*:
  never mix its runs with OpenAI runs except through `compare-runs`. At `temperature: 0` replicates are identical, so use `--repeats 1`.
- The default embedder is `bge-small` (needs Hugging Face access). `hash-1024` is an offline stand-in for tests/dry runs.
  Select with `--embedder <tag>` or `RAGEBLATE_EMBEDDER`. Vectors are stored per tag in `data/embeddings/<tag>/`.

## End-to-end runbook

Run from the project root. After each step, do the check before moving on.

| # | Command | Check |
|---|---|---|
| 1 | `uv sync` | `uv run pytest -q` passes |
| 2 | `uv run kaggle datasets download -d Cornell-University/arxiv -p data/raw --unzip` | `data/raw/arxiv-metadata-oai-snapshot.json` exists (~5 GB) |
| 3 | `uv run aixiv-rag ingest` | 16,755 rows (1995-2026); `duplicate ids dropped: 1`. ~73 ID-year mismatches are expected (Dec/Jan boundary, or papers held in moderation); ingest asserts none is the impossible direction |
| 4 | `uv run aixiv-rag topics` | each topic has >= ~150 docs (all do: smallest is enso, 283). If one is thin, widen its regexes in `topics.yaml` |
| 5 | `uv run aixiv-rag snapshots` | every snapshot non-empty; `control_X` has the same `n_docs` as `remove_X` |
| 6 | `uv run aixiv-rag embed` | shape `(n_rows, 384)`; then `uv run aixiv-rag check` prints OK |
| 7 | `uv run aixiv-rag retrieve --snapshot upto_2010 --query "ocean heat uptake"` | 10 hits, all with year <= 2010 |
| 8 | `uv run aixiv-rag ablate --answerer extractive --limit-questions 3 --run-id smoke` | finishes, writes `runs/smoke/` (free, offline) |
| 8b | `uv run aixiv-rag audit-questions` | questions retrieve what they claim to (needs the real embedder; read the `flags` column). Free, no API |
| 9a | `uv run aixiv-rag ablate --answerer openai --repeats 5 --plan-only` | prints distinct prompts, how many are cached, and a cost estimate. Spends nothing |
| 9b | `uv run aixiv-rag ablate --answerer openai --repeats 5 --yes` | the owner's call (see below). Paid calls run concurrently (`--workers`), finished ones are cached, so a crash is resumable |
| 10 | `uv run aixiv-rag report --run <run_id>` | `runs/<run_id>/report.md`: every sanity check is PASS |
| 11a | `uv run aixiv-rag judge --run <run_id> --plan-only` | cost of the LLM judge (support + abstention justification) |
| 11b | `uv run aixiv-rag judge --run <run_id> --yes --export-labels 40`, hand-label `judge_labels.csv`, then `judge --score-labels` | judge agreement with a human; re-run `report` |
| 12 | `uv run aixiv-rag ablate ... --prompt-version v2 --run-id <b>` then `uv run aixiv-rag compare-runs <a> <b>` | does the topic effect survive a reworded prompt / other model (`--model`, `--effort`)? |

Paid runs are the owner's to launch. `ablate` and `judge` refuse more than 60 uncached paid calls without `--yes`; do not pass
`--yes` unless the owner asked for a real run. `--plan-only` is always safe and shows the exact number of calls.
`--repeats N` draws N independent answers per prompt on the reference snapshot only (`--repeat-snapshots` to widen it): that is
what the noise floor is measured from.

## Reading a report

- Sanity checks must all PASS. A FAIL means a bug (misalignment, wrong index), not a finding: stop and debug.
- `jaccard_unaffected` must be 1.0 (monotonicity). `jaccard_affected` is the real signal.
- Compare `remove_X` against the controls, per topic, in the headline table. `outside` (`control_X`) cannot touch X's evidence, so
  beating it is nearly guaranteed; `all` (`control_all_X_s1/s2`) can, and is the one to believe. A `*_gap` whose interval includes 0 is
  not evidence of a topic effect. Gaps are signed: positive = removing the topic hurt more.
- Read drift against the noise floor section, not on its own, and read the `_affected` columns: an unaffected question has a
  byte-identical prompt, hence a cached identical answer, hence drift exactly 0 (the cache, not stability).
- The dose-response table compares removal types at equal evidence loss; if they match there, the effect is lost evidence, not topic.
- The judge section (if present) measures grounding, not truth, and is unvalidated until `--score-labels` has been run.
- For targeted questions with `answered_targeted > 0`, open `results.parquet` and read those answers; they are
  candidate hallucinations. Do not call them hallucinations without reading them.

## Honesty rules

- Runs with `hash` embeddings or the `extractive` answerer are pipeline checks. The report banners say so; never
  present their numbers as research findings.
- Report only what you ran. Say so when something was not verified (e.g. no API key, no Hugging Face access).
- Reasoning models reject `temperature`, so the wrapper drops it. Fresh calls are then not deterministic; the on-disk cache
  (`data/cache/llm`) is what makes a finished run reproducible. Never delete it mid-study, and never mix cached answers from
  different models or efforts in one comparison.
- Abstention is a sentinel from one prompt and one model. Do not state a finding about abstention as general until `compare-runs`
  shows it survives `--prompt-version v2` (and ideally another `--model`).
- `answer_change` counts an abstention flip as 1.0 and two abstentions as 0; it is a convention, not a physical quantity.
- Results from before replicates existed have no `sample` column; every reader treats that as "one draw", and the report says no noise
  floor was measured.
- Topic rules are v2 (`configs/topics.yaml`); `configs/archive/topics.v1.yaml` is what run `20261015T034036Z` used. Different rules
  remove different documents, so do not compare topic-removal numbers across the two without saying so.
- The `license` column is per-paper provenance. Do not publish derived corpora/embeddings before the data licence is confirmed.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError ... arxiv-metadata...json` | Download step skipped, or `data/raw` is empty. Paths are anchored to the repo root, so cwd does not matter. |
| `No rows matched` (ingest) | `MAX_LINES` is set in `ingest.py`, or `TARGET_CATEGORIES` is wrong. |
| `AlignmentError: ... changed since these embeddings were made` | Corpus was re-ingested. Re-run `aixiv-rag embed`. |
| `AlignmentError: shape` / `not L2-normalised` | Stale or hand-edited embeddings. Re-run `embed`. |
| `OMP: Error #15` | torch and FAISS ended up in one process. Restore the separation; do not set `KMP_DUPLICATE_LIB_OK`. |
| `OPENAI_API_KEY is not set` | Add it to `.env`, or use `--answerer extractive`. |
| `IncompleteResponseError` | The model used up `max_output_tokens` (hidden reasoning counts). Raise `max_output_tokens` or lower `reasoning_effort` in `configs/rag.yaml` and re-run; finished calls are cached. |
| `400 ... reasoning.effort` | The model does not accept that effort value (e.g. `gpt-6.1-sol` rejects `none` and `minimal`). Pick another in `configs/rag.yaml`. |
| `ValueError: unknown key(s)` | Typo in a snapshot YAML. Allowed keys are listed in the message. |
| Hugging Face download fails | No network access to huggingface.co. Run `embed` on a machine that has it, or use `--embedder hash-1024` for a dry run. |
| `cannot reach Ollama at ...` | Tailscale is down on one machine, the PC is asleep, or Ollama listens only on localhost (set `OLLAMA_HOST=0.0.0.0` on the PC and restart Ollama). |
| `... does not fit num_ctx` / `filled the whole context window` | Prompt longer than the local context window; Ollama would truncate it silently. Raise `ollama.num_ctx`. |
| `uv sync` picks the wrong Python | `requires-python >= 3.14`; check `.python-version`. |

## Extending

- **New snapshot**: add `configs/snapshots/<name>.yaml` (schema in `snapshot.py`). `tests/test_topics_snapshot.py::test_shipped_configs_are_valid` validates it.
- **New topic**: add regexes to `topics.yaml`; check size and overlap with `aixiv-rag topics`; add `remove_<t>`, `control_<t>` and
  `control_all_<t>_s1/_s2` snapshots; tag at least 6-8 questions with `topic: <t>` and run `audit-questions`.
- **New question**: append to `questions.yaml` (unique `id`, `topic` or `null`). Changing questions changes `questions_hash`; cached answers for old questions stay valid.
- **New metric**: pure function in `metrics.py` with a test in `tests/test_metrics_ablate.py`, then surface it in `report.py`.
- **Change the prompt**: bump `PROMPT_VERSION` in `rag.py`; the prompt text is part of the LLM cache key.
- Run `uv run pytest -q` and `uv run ruff check src tests` before finishing.

## Working with the owner

Andre is learning Python properly. When he asks to *understand* something, give pseudocode or pointers on what to try.
When he asks for code, give clean, commented code. When reviewing his code, flag bugs directly. Keep answers concise;
ask when a decision is his to make.
