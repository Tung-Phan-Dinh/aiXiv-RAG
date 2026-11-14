"""Shared fixtures: a tiny synthetic corpus with four clear themes and offline hash embeddings."""

from __future__ import annotations

import pandas as pd
import pytest

from rageblate.data import Context, ids_sha256
from rageblate.embed import HashEmbedder

THEMES = {
    "sea_ice": "arctic sea ice melt thickness extent polar glaciers",
    "earthquakes": "earthquake seismic fault rupture aftershock magnitude",
    "ocean": "ocean circulation overturning gulf stream eddies heat transport",
    "turbulence": "turbulence boundary layer eddy viscosity kolmogorov spectrum",
}
FILLER = ["observations", "models", "simulation", "data", "analysis", "regional", "global", "decadal", "seasonal", "satellite"]
QUERIES = {
    "sea_ice": "arctic sea ice melt",
    "earthquakes": "earthquake fault rupture",
    "ocean": "ocean overturning circulation",
    "turbulence": "turbulence boundary layer",
}


def make_corpus(n: int = 96) -> pd.DataFrame:
    keys = list(THEMES)
    rows = []
    for i in range(n):
        theme = keys[i % len(keys)]
        year = 2001 + (i * 7) % 24  # 2001..2024
        yy = year % 100
        arxiv_id = f"{yy:02d}01.{i:04d}" if i % 8 else f"physics/{yy:02d}01{i:03d}"  # new- and old-style ids
        f1, f2 = FILLER[i % len(FILLER)], FILLER[(i * 3 + 1) % len(FILLER)]
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": f"{theme.replace('_', ' ')} study {i}",
                "abstract": f"{THEMES[theme]} {f1} {f2} case {i}.",
                "year": year,
                "categories": "physics.ao-ph" if i % 2 else "physics.geo-ph physics.ao-ph",
                "license": None,
                "update_date": "2020-01-01",
            }
        )
    df = pd.DataFrame(rows).sort_values("arxiv_id").reset_index(drop=True)
    df["year"] = df["year"].astype("int16")
    return df


@pytest.fixture(scope="session")
def topics():
    return {
        "sea_ice": {"any_of": ["sea[- ]ice", "glaciers?"]},
        "earthquakes": {"any_of": ["earthquakes?", "seismic"]},
    }


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder(dim=256)


@pytest.fixture(scope="session")
def ctx(embedder):
    df = make_corpus()
    texts = [f"{t}. {a}" for t, a in zip(df["title"], df["abstract"])]
    emb = embedder.encode(texts, kind="doc")
    meta = {"dim": 256, "ids_sha256": ids_sha256(df["arxiv_id"]), "backend": "hash", "model": None}
    return Context(df=df, emb=emb, meta=meta, tag="hash-1024")


@pytest.fixture(scope="session")
def questions():
    return [{"id": f"q{i}", "topic": t, "question": q} for i, (t, q) in enumerate(QUERIES.items())] + [
        {"id": "q_ctrl", "topic": None, "question": "ocean heat transport eddies"}
    ]


@pytest.fixture
def make_cfg():
    def _make(name="t", include=None, exclude=None):
        return {"name": name, "include": include or {}, "exclude": exclude or {}}

    return _make
