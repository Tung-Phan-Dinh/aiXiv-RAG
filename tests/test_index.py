import numpy as np
import pytest
from conftest import QUERIES

from rageblate.index import SearchIndex, build_index
from rageblate.snapshot import resolve_snapshot
from rageblate.topics import topic_mask


@pytest.fixture(scope="module")
def qvecs(embedder):
    return embedder.encode(list(QUERIES.values()), kind="query")


def test_faiss_and_numpy_agree(ctx, topics, make_cfg, qvecs):
    pytest.importorskip("faiss")
    snap = resolve_snapshot(ctx.df, make_cfg(), topics)
    s_f, r_f = build_index(ctx, snap, "faiss").search(qvecs, 8)
    s_n, r_n = build_index(ctx, snap, "numpy").search(qvecs, 8)
    assert np.allclose(s_f, s_n, atol=1e-5)
    for qi in range(len(qvecs)):
        # Exact ties at the k-th score may be broken differently by the two backends, so a
        # document in only one result set must score exactly the boundary (k-th) score.
        for doc in set(r_f[qi]) ^ set(r_n[qi]):
            assert float(qvecs[qi] @ ctx.emb[doc]) == pytest.approx(float(s_n[qi].min()), abs=1e-5)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_search_stays_inside_snapshot(ctx, topics, make_cfg, qvecs, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    snap = resolve_snapshot(ctx.df, make_cfg(include={"year_max": 2010}), topics)
    _, rows = build_index(ctx, snap, backend).search(qvecs, 10)
    assert set(rows.ravel()) <= set(snap.rows)
    assert ctx.df["year"].to_numpy()[rows].max() <= 2010


def test_monotonicity_untouched_topk_is_identical(ctx, topics, make_cfg, qvecs):
    """If no full-corpus top-k doc is removed, the subset's top-k must be identical."""
    full = resolve_snapshot(ctx.df, make_cfg(), topics)
    sub = resolve_snapshot(ctx.df, make_cfg(exclude={"topic": "earthquakes"}), topics)
    removed = topic_mask(ctx.df, topics["earthquakes"])
    _, r_full = build_index(ctx, full, "numpy").search(qvecs, 5)
    _, r_sub = build_index(ctx, sub, "numpy").search(qvecs, 5)
    checked = 0
    for a, b in zip(r_full, r_sub):
        if not removed[a].any():
            assert list(a) == list(b)
            checked += 1
    assert checked >= 2  # the property was actually exercised
    # ...and a query about the removed topic must change
    assert set(r_full[1]) != set(r_sub[1])


def test_search_is_deterministic(ctx, topics, make_cfg, qvecs):
    snap = resolve_snapshot(ctx.df, make_cfg(), topics)
    index = build_index(ctx, snap, "numpy")
    assert np.array_equal(index.search(qvecs, 6)[1], index.search(qvecs, 6)[1])


def test_k_larger_than_index_is_clamped():
    vecs = np.eye(3, dtype=np.float32)
    _, rows = SearchIndex(vecs, np.array([10, 20, 30]), "numpy").search(vecs[:1], 50)
    assert rows.shape == (1, 3) and rows[0, 0] == 10


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        SearchIndex(np.zeros((0, 4), dtype=np.float32), np.array([]), "numpy")
