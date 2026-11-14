"""Exact (flat) inner-product search over one snapshot.

Always build a *fresh* index per snapshot from the cached embedding matrix. Never search a
global index and filter afterwards: that silently returns fewer than k results and corrupts
the overlap metrics. At ~20k x 384 a flat index builds in milliseconds.

Vectors are L2-normalised, so inner product == cosine similarity.
"""

from __future__ import annotations

import numpy as np

from .data import Context
from .snapshot import Snapshot


class SearchIndex:
    def __init__(self, vectors: np.ndarray, rows: np.ndarray, backend: str = "faiss"):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(rows) or len(vectors) == 0:
            raise ValueError("need a non-empty (n, d) matrix with one corpus row id per vector")
        self.vectors = vectors
        self.rows = np.asarray(rows)  # position in this index -> row in the corpus
        self.n, self.dim = vectors.shape
        self.backend = backend
        self._faiss = None
        if backend == "faiss":
            import faiss  # lazy: keep torch and faiss out of the same process on macOS

            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.vectors)
        elif backend != "numpy":
            raise ValueError(f"unknown backend {backend!r} (use 'faiss' or 'numpy')")

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, corpus_rows), each shape (n_queries, min(k, n)), best first."""
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(k, self.n)
        if self._faiss is not None:
            scores, pos = self._faiss.search(q, k)
        else:
            sims = q @ self.vectors.T
            pos = np.argsort(-sims, axis=1, kind="stable")[:, :k]  # ties -> lower position first
            scores = np.take_along_axis(sims, pos, axis=1)
        return scores, self.rows[pos]


def build_index(ctx: Context, snapshot: Snapshot, backend: str = "faiss") -> SearchIndex:
    return SearchIndex(ctx.emb[snapshot.rows], snapshot.rows, backend)
