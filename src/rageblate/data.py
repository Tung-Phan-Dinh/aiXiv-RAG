"""Loading the corpus and its embeddings, with a hard guard against row misalignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CORPUS, EMBEDDINGS_DIR


class AlignmentError(RuntimeError):
    """Embeddings do not correspond row-for-row to the corpus parquet."""


def ids_sha256(ids) -> str:
    """Hash of the ordered id list. Newline-joined so ("a","bc") != ("ab","c")."""
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def load_corpus(path: Path = CORPUS, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `aixiv-rag ingest` first")
    df = pd.read_parquet(path, columns=columns)
    if "arxiv_id" in df.columns and not df["arxiv_id"].is_unique:
        raise AlignmentError("corpus has duplicate arxiv_id values")
    return df


def doc_texts(df: pd.DataFrame, template: str = "{title}. {abstract}") -> list[str]:
    """The exact text that gets embedded for each document (row order preserved)."""
    return [template.format(title=t, abstract=a) for t, a in zip(df["title"], df["abstract"])]


@dataclass
class Context:
    """Everything a retrieval process needs: corpus rows + the matching vectors."""

    df: pd.DataFrame
    emb: np.ndarray  # (n_docs, dim) float32, L2-normalised, same row order as df
    meta: dict
    tag: str


def load_embeddings(df: pd.DataFrame, tag: str, base_dir: Path = EMBEDDINGS_DIR):
    folder = base_dir / tag
    meta_path, emb_path = folder / "meta.json", folder / "embeddings.npy"
    if not (meta_path.exists() and emb_path.exists()):
        raise FileNotFoundError(f"no embeddings for {tag!r} in {folder} - run `aixiv-rag embed`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    emb = np.load(emb_path)

    if emb.shape != (len(df), meta["dim"]):
        raise AlignmentError(f"embeddings shape {emb.shape} != ({len(df)}, {meta['dim']})")
    if emb.dtype != np.float32:
        raise AlignmentError(f"embeddings dtype {emb.dtype}, expected float32")
    if meta["ids_sha256"] != ids_sha256(df["arxiv_id"]):
        raise AlignmentError(
            "corpus.parquet changed since these embeddings were made (id-order hash differs). "
            "Re-run `aixiv-rag embed`."
        )
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise AlignmentError("embeddings are not L2-normalised (inner product != cosine)")
    return emb, meta


def load_context(tag: str | None = None) -> Context:
    from .embed import resolve  # local import keeps module import light

    tag, _ = resolve(tag)
    df = load_corpus()
    emb, meta = load_embeddings(df, tag)
    return Context(df=df, emb=emb, meta=meta, tag=tag)
