import numpy as np
import pytest

from rageblate import embed
from rageblate.data import AlignmentError, ids_sha256, load_embeddings


def test_hash_embedder_is_normalised_and_deterministic(embedder):
    a = embedder.encode(["sea ice melt in the arctic", "earthquake fault rupture"])
    b = embedder.encode(["sea ice melt in the arctic", "earthquake fault rupture"])
    assert a.dtype == np.float32 and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)


def test_hash_embedder_similar_text_is_closer(embedder):
    q, near, far = embedder.encode(["arctic sea ice melt", "sea ice melt thickness in the arctic", "earthquake aftershock magnitude"])
    assert q @ near > q @ far


def test_row_alignment_batch_equals_individual(embedder):
    texts = ["alpha beta gamma", "ocean overturning", "seismic fault", "turbulence spectrum", "glaciers retreat"]
    batch = embedder.encode(texts)
    for i, t in enumerate(texts):
        assert np.allclose(batch[i], embedder.encode([t])[0], atol=1e-6)


def test_embed_corpus_roundtrip_and_alignment_guard(ctx, embedder, tmp_path):
    cfg = {"backend": "hash", "dim": 256, "doc_template": "{title}. {abstract}"}
    meta = embed.embed_corpus(ctx.df, "t", cfg, embedder, base_dir=tmp_path)
    assert meta["ids_sha256"] == ids_sha256(ctx.df["arxiv_id"])

    emb, meta2 = load_embeddings(ctx.df, "t", base_dir=tmp_path)
    assert emb.shape == (len(ctx.df), 256) and meta2["n"] == len(ctx.df)

    shuffled = ctx.df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(AlignmentError, match="changed"):
        load_embeddings(shuffled, "t", base_dir=tmp_path)
    with pytest.raises(AlignmentError, match="shape"):
        load_embeddings(ctx.df.iloc[:-1], "t", base_dir=tmp_path)


def test_ids_hash_depends_on_order_and_boundaries():
    assert ids_sha256(["a", "b"]) != ids_sha256(["b", "a"])
    assert ids_sha256(["a", "bc"]) != ids_sha256(["ab", "c"])


def test_query_vector_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "CACHE", tmp_path)
    first = embed.vectors(["arctic sea ice", "fault rupture"], kind="query", tag="hash-1024")
    assert len(list(tmp_path.rglob("*.npy"))) == 2
    second = embed.vectors(["arctic sea ice", "fault rupture"], kind="query", tag="hash-1024")
    assert np.array_equal(first, second) and first.shape == (2, 1024)
