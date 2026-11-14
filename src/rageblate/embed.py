"""Turn text into vectors.

Two backends (chosen per embedder tag in configs/embed.yaml):
  * sentence-transformers - the real thing (BAAI/bge-small-en-v1.5 by default).
  * hash                  - deterministic, offline, dependency-free bag-of-words vectors,
                            for unit tests and dry runs. NOT for research results.

torch is imported lazily so that processes that use FAISS never load it: on macOS the two
ship different OpenMP runtimes and loading both can crash with "OMP: Error #15".
Retrieval/ablation processes therefore get query vectors from an on-disk cache that a
short-lived subprocess fills (see `vectors`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from .config import load_embed_config, stable_hash
from .data import doc_texts, ids_sha256, load_corpus
from .paths import CACHE, EMBEDDINGS_DIR

_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with", "we", "our", "their", "these", "those", "can", "has", "have", "been", "using", "used", "use", "show", "shown", "paper", "results", "result", "also"]
)
_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Signed feature hashing of unigrams+bigrams, sublinear tf, L2-normalised."""

    backend = "hash"

    def __init__(self, dim: int = 1024, query_prefix: str = ""):
        self.dim = dim
        self.query_prefix = query_prefix  # unused; kept for a uniform interface

    def _encode_one(self, text: str) -> np.ndarray:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, count in Counter(feats).items():
            h = zlib.crc32(feat.encode("utf-8"))  # stable across runs (unlike hash())
            sign = 1.0 if (h >> 20) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v[0], norm = 1.0, 1.0
        return v / norm

    def encode(self, texts, *, kind: str = "doc", show_progress: bool = False) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(t) for t in texts]).astype(np.float32)

    def info(self) -> dict:
        return {"device": "cpu"}


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(self, model: str, batch_size: int = 64, query_prefix: str = ""):
        import torch  # lazy: see module docstring
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model_name = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model, device=self.device)

    def encode(self, texts, *, kind: str = "doc", show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if kind == "query":  # bge: prefix on queries only, never on documents
            texts = [self.query_prefix + t for t in texts]
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return out.astype(np.float32)

    def info(self) -> dict:
        import sentence_transformers
        import torch

        return {
            "device": self.device,
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        }


def resolve(tag: str | None = None) -> tuple[str, dict]:
    """Pick the embedder: explicit tag > $RAGEBLATE_EMBEDDER > configs/embed.yaml default."""
    all_cfg = load_embed_config()
    tag = tag or os.environ.get("RAGEBLATE_EMBEDDER") or all_cfg["default"]
    if tag not in all_cfg["embedders"]:
        raise KeyError(f"unknown embedder {tag!r}; known: {sorted(all_cfg['embedders'])}")
    cfg = dict(all_cfg["embedders"][tag])
    cfg["doc_template"] = all_cfg.get("doc_template", "{title}. {abstract}")
    return tag, cfg


def build_embedder(cfg: dict):
    if cfg["backend"] == "hash":
        return HashEmbedder(cfg.get("dim", 1024), cfg.get("query_prefix", ""))
    if cfg["backend"] == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg["model"], cfg.get("batch_size", 64), cfg.get("query_prefix", ""))
    raise ValueError(f"unknown embedder backend {cfg['backend']!r}")


# ---------------------------------------------------------------- corpus embeddings


def embed_corpus(df, tag: str, cfg: dict, embedder, base_dir: Path = EMBEDDINGS_DIR) -> dict:
    """Embed every document once, in corpus row order, and save vectors + provenance."""
    texts = doc_texts(df, cfg["doc_template"])
    emb = embedder.encode(texts, kind="doc", show_progress=True).astype(np.float32)
    if emb.shape[0] != len(df):
        raise RuntimeError("embedder returned a different number of rows than documents")

    folder = base_dir / tag
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", emb)
    meta = {
        "tag": tag,
        "backend": cfg["backend"],
        "model": cfg.get("model"),
        "dim": int(emb.shape[1]),
        "n": int(emb.shape[0]),
        "normalised": True,
        "doc_template": cfg["doc_template"],
        "config_hash": stable_hash(cfg),
        "ids_sha256": ids_sha256(df["arxiv_id"]),  # guard against row misalignment
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **embedder.info(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")  # written last
    return meta


# ---------------------------------------------------------------- query / answer vectors


def _cache_file(tag: str, cfg: dict, kind: str, text: str) -> Path:
    key = hashlib.sha256(f"{stable_hash(cfg)}|{kind}|{text}".encode()).hexdigest()[:32]
    return CACHE / "emb" / tag / f"{kind}-{key}.npy"


def _warm(texts: list[str], kind: str, tag: str) -> None:
    tag, cfg = resolve(tag)
    vecs = build_embedder(cfg).encode(texts, kind=kind)
    for text, vec in zip(texts, vecs):
        path = _cache_file(tag, cfg, kind, text)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec)


def vectors(texts: list[str], *, kind: str = "query", tag: str | None = None, subprocess_ok: bool = False) -> np.ndarray:
    """Cached vectors for arbitrary texts. kind is "query" (gets the bge prefix) or "doc".

    subprocess_ok=True: fill missing entries in a child process, so the calling process
    (which may hold FAISS) never imports torch.
    """
    tag, cfg = resolve(tag)
    paths = [_cache_file(tag, cfg, kind, t) for t in texts]
    missing = [t for t, p in zip(texts, paths) if not p.exists()]
    if missing:
        if subprocess_ok and cfg["backend"] != "hash":
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
                json.dump({"texts": missing, "kind": kind}, f)
            try:
                subprocess.run(
                    [sys.executable, "-m", "rageblate.embed", "--embedder", tag, "--warm-json", f.name],
                    check=True,
                )
            finally:
                os.unlink(f.name)
        else:
            _warm(missing, kind, tag)
    return np.stack([np.load(p) for p in paths]).astype(np.float32)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Embed the corpus (or warm the vector cache).")
    parser.add_argument("--embedder", help="tag from configs/embed.yaml (default: configs default)")
    parser.add_argument("--warm-json", help="internal: JSON file with {texts, kind} to cache")
    args = parser.parse_args(argv)

    tag, cfg = resolve(args.embedder)
    if args.warm_json:
        payload = json.loads(Path(args.warm_json).read_text(encoding="utf-8"))
        _warm(payload["texts"], payload["kind"], tag)
        return

    df = load_corpus()
    print(f"embedding {len(df):,} documents with {tag!r} ({cfg['backend']})")
    meta = embed_corpus(df, tag, cfg, build_embedder(cfg))
    print(f"wrote {EMBEDDINGS_DIR / tag}  shape=({meta['n']}, {meta['dim']})  device={meta.get('device')}")


if __name__ == "__main__":
    main()
