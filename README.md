# aiXiv-RAG: a corpus-ablation harness for RAG

**Question:** how do a RAG system's answers change when the data underneath it changes?

The harness builds a retrieval-augmented question answerer over arXiv geoscience abstracts
(`physics.geo-ph`, `physics.ao-ph`), then asks the same fixed question set against many
**corpus snapshots**: year cut-offs, whole topics removed, and size-matched random-removal
controls. It measures what moves.

Everything is written from scratch (no LangChain / LlamaIndex) so every stage is inspectable.

## Pipeline

```
Kaggle arXiv dump ──ingest──▶ data/corpus.parquet  (immutable, one row per paper)
                                   │
                                   ├──embed──▶ data/embeddings/<tag>/embeddings.npy  (+ meta.json with an id-order hash)
                                   │
configs/snapshots/*.yaml ──────────┤   a snapshot = a FILTER over the corpus (row indices), never a copy
                                   ▼
questions.yaml ──▶ ablate: fresh exact index per snapshot ─▶ top-k ─▶ answer (OpenAI, cached) ─▶ runs/<id>/results.parquet
                                   ▼
                              report: stability · drift · temporal profile · abstention  ─▶ runs/<id>/report.md
```

## Quick start

```bash
uv sync                                                      # Python 3.14, see pyproject.toml
echo 'OPENAI_API_KEY=sk-...' > .env                          # git-ignored

# data (needs a Kaggle login: `uv run kaggle auth login`)
uv run kaggle datasets download -d Cornell-University/arxiv -p data/raw --unzip

uv run aixiv-rag ingest        # -> data/corpus.parquet, prints per-year counts
uv run aixiv-rag topics        # how many docs each topic rule matches
uv run aixiv-rag embed         # bge-small-en-v1.5 (MPS on Apple Silicon); a few minutes
uv run aixiv-rag check         # corpus <-> embeddings alignment

uv run aixiv-rag retrieve --snapshot upto_2010 --query "ocean heat uptake"
uv run aixiv-rag ask "What drives ENSO?" --snapshot full --snapshot remove_enso   # your own question; omit it for an interactive prompt
uv run aixiv-rag audit-questions                    # free: do the questions retrieve what they claim to?
uv run aixiv-rag ablate --answerer openai --repeats 5 --plan-only   # exact call count + cost estimate, spends nothing
uv run aixiv-rag ablate --answerer openai --repeats 5 --yes         # 37 snapshots x 72 questions, 5 draws on the reference
uv run aixiv-rag judge --run <run_id> --plan-only   # optional LLM judge of support / abstention justification
uv run aixiv-rag report --run <run_id>              # prints the path of runs/<run_id>/report.md
uv run pytest
```

No API key yet? `--answerer extractive` runs the whole pipeline offline with a non-LLM baseline,
and `--answerer none` does retrieval only. The offline `hash-1024` embedder
(`--embedder hash-1024` / `RAGEBLATE_EMBEDDER`) lets you exercise everything without downloading a
model. Both are flagged in the report and are pipeline checks, not research results.

## Design decisions that keep the experiment valid

| Decision | Why |
|---|---|
| One **immutable** `corpus.parquet`; snapshots are row-index filters | Comparable across runs; nothing can drift silently. |
| `year` = year of the **first** version, not `update_date` | Otherwise old papers leak into later snapshots. |
| One abstract = one chunk | Corpus composition is then the only thing that varies. |
| **Fresh** exact index per snapshot | Searching a global index and filtering afterwards returns fewer than k hits and corrupts overlap metrics. |
| Embeddings carry an **id-order hash**; loading refuses a mismatch | Row misalignment is the classic silent bug. |
| Topic rules are **frozen regexes** in `configs/topics.yaml`, hashed into the snapshot | "Remove topic X" means the same documents every run. |
| Two random **controls** remove the same *number* of docs: `outside` (never from the topic) and `all` (anywhere, two seeds, averaged) | `outside` separates "this topic mattered" from "any 5% of the corpus"; but it cannot touch the topic's evidence, so `all` is the fair one. |
| Headline gaps carry **bootstrap intervals over questions** and are paired by question | 8 questions per topic cannot support a bare point estimate. |
| Extra answer draws (`--repeats`) measure the model's own **noise floor** | A change in answers means nothing until it exceeds what two draws of the same prompt do. |
| Answers cached on disk (`data/cache/llm`), keyed by model, prompt and settings | Re-runs are free and byte-identical. `temperature=0` is sent when the model accepts it, but reasoning models reject it and it is dropped, so fresh calls are **not** deterministic: the cache is what makes a finished run reproducible. |
| torch and FAISS never share a process (query vectors come from a cache filled by a subprocess) | macOS ships two OpenMP runtimes -> `OMP: Error #15`. |

## Metrics (`src/rageblate/metrics.py`)

All are relative to the reference snapshot `full`.

1. **Retrieval stability**: Jaccard of the top-k ids. Reported separately for questions whose
   reference top-k lost documents (`affected`) and those that did not.
2. **Answer drift**: `1 - cosine` between embedded answers (non-abstained pairs), plus `answer_change`
   (defined for every pair: an abstention flip = 1.0, two abstentions = 0), a noise floor from repeated draws, and
   a scaled version (0 = same-prompt noise, 1 = as far apart as answers to unrelated questions).
3. **Temporal profile**: distribution of publication years among retrieved docs.
4. **Abstention and support**: the model must answer `INSUFFICIENT_EVIDENCE` when the context cannot
   support an answer. For *targeted* questions (their evidence topic was removed), an answer instead of
   an abstention is a candidate fabrication: read those. Also the rate of citing ids that were not in
   the context.

5. **Answer quality** (optional, `aixiv-rag judge`): an independent LLM judge, blind to the snapshot, rates whether each answer is
   supported by what was retrieved and whether each abstention was justified. Grounding, not truth; validate it on a hand-labelled sample.

Also in the report: a **dose-response** table (outcome by share of the reference top-k removed, per kind of snapshot), citation overlap
with the reference answer, and `compare-runs` to check a conclusion against another prompt, model or embedder.

Built-in sanity checks in every report: retrieved docs never leave their snapshot, and
**monotonicity** (if no reference top-k doc was removed, the top-k must be identical).

## Layout

```
configs/        embed.yaml  rag.yaml  topics.yaml  questions.yaml  snapshots/*.yaml
src/rageblate/  ingest embed data topics snapshot index retrieve llm rag plan ablate audit metrics judge compare report cli
tests/          pytest suite for the validity properties above
data/           (git-ignored) raw dump, corpus, embeddings, caches
runs/           (git-ignored) one folder per ablation run
AGENTS.md       runbook for coding agents
```

## Data and licensing

The corpus is derived from the Kaggle `Cornell-University/arxiv` metadata snapshot. `data/` is
git-ignored; the repo ships code, configs and download instructions only. Before publishing any derived
data (corpus, embeddings, retrieved text), check the dataset licence on Kaggle and arXiv's bulk-data terms:
the per-record `license` field describes the papers, not the metadata file.

## Limitations

- Topic labels are regex rules, not a classifier; they are transparent and frozen, not perfect.
- Top-k of 10 over ~17k abstracts: very small topics give noisy removal effects (`aixiv-rag topics` warns). Early years are thin
  (`upto_2010` has ~2.2k docs), so year-cutoff snapshots there are coarse.
- ~75 abstracts are under 30 words (withdrawal notices, one-liners). They are kept, not silently filtered.
- `year` is the first-submission year. The arXiv id month can differ (Dec/Jan boundary, or papers held in moderation); ingest asserts the id never *predates* the submission.
- Answer drift uses an embedding of the answer text, which is a proxy for meaning change.
- Abstention is detected by a sentinel the model was asked to emit, from one prompt and one model by default (`--prompt-version v2`
  and `compare-runs` test that). Nothing here knows whether an answer is *correct*: the judge checks grounding in the retrieved
  abstracts, and is itself only as good as its agreement with your hand labels.
- Topic membership is still regexes; leakage (a topic-relevant paper missing from the rule) is reduced in v2, not eliminated.
  `aixiv-rag topics` prints an overlap matrix; removing one topic also removes parts of overlapping topics.
- The questions are hand-written, 8 per topic. Intervals reflect that; more questions narrow them, repeats do not.
