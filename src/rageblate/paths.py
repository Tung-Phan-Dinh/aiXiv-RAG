"""Single source of truth for where things live on disk."""

from pathlib import Path

# src/rageblate/paths.py -> parents[2] is the project root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"  # git-ignored: raw dump, corpus, embeddings, caches
RAW = DATA / "raw" / "arxiv-metadata-oai-snapshot.json"
CORPUS = DATA / "corpus.parquet"  # immutable, written by ingest.py
EMBEDDINGS_DIR = DATA / "embeddings"  # one sub-folder per embedder tag
CACHE = DATA / "cache"  # LLM answers, query vectors

RUNS = ROOT / "runs"  # one sub-folder per ablation run
CONFIGS = ROOT / "configs"
SNAPSHOT_CONFIGS = CONFIGS / "snapshots"
