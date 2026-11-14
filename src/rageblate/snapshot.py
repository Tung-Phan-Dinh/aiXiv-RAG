"""A snapshot is a *filter over the immutable corpus*, never a modified copy of it.

Config (configs/snapshots/<name>.yaml):

    include:  {year_min, year_max, categories_any}     # what is allowed in
    exclude:  {topic: name|[names],                    # remove every doc matching the topic(s)
               random: {matched_to_topic|n|frac, seed}} # control: remove random docs instead

The result is a sorted array of corpus row indices, so a snapshot is cheap, reproducible, and
comparable across runs. For `random.matched_to_topic: T` we remove as many documents as topic T
has (within the include set). Two pools to draw them from (`random.pool`):

  outside_topic (default)  only from documents outside T. The control differs from the topic removal
                           only in *which* documents disappear, not how many - but it can never touch
                           the evidence for questions about T, so it is a gentle control.
  all                      from any remaining document, topic documents included. Same number removed,
                           blind to topic: the harder, fairer baseline for "is it the topic, or just
                           that many documents".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import list_snapshot_names, load_snapshot_config, load_topics, stable_hash
from .data import ids_sha256, load_corpus
from .topics import topic_mask


@dataclass(frozen=True)
class Snapshot:
    name: str
    cfg: dict
    cfg_hash: str  # covers the config AND the topic rules it references
    rows: np.ndarray  # sorted corpus row indices that are inside the snapshot
    ids_hash: str  # hash of the selected arxiv_ids, in order

    @property
    def n_docs(self) -> int:
        return len(self.rows)

    @property
    def removed_topics(self) -> list[str]:
        return removed_topics(self.cfg)


def removed_topics(cfg: dict) -> list[str]:
    """Topics a snapshot removes outright (not the ones only used to size a random control)."""
    t = cfg["exclude"].get("topic")
    return [] if t is None else ([t] if isinstance(t, str) else list(t))


def _referenced_topics(cfg: dict) -> list[str]:
    names = removed_topics(cfg)
    rnd = cfg["exclude"].get("random") or {}
    if "matched_to_topic" in rnd:
        names.append(rnd["matched_to_topic"])
    return sorted(set(names))


def resolve_snapshot(df: pd.DataFrame, cfg: dict, topics: dict[str, dict]) -> Snapshot:
    inc, exc = cfg["include"], cfg["exclude"]
    year = df["year"].to_numpy()

    keep = np.ones(len(df), dtype=bool)
    if "year_min" in inc:
        keep &= year >= inc["year_min"]
    if "year_max" in inc:
        keep &= year <= inc["year_max"]
    if "categories_any" in inc:
        wanted = set(inc["categories_any"])
        keep &= df["categories"].str.split().map(lambda cats: not wanted.isdisjoint(cats)).to_numpy(dtype=bool)
    in_include_set = keep.copy()

    for name in removed_topics(cfg):
        keep &= ~topic_mask(df, topics[name])

    rnd = exc.get("random")
    if rnd:
        pool = keep.copy()
        if "matched_to_topic" in rnd:
            in_topic = topic_mask(df, topics[rnd["matched_to_topic"]])
            n = int((in_include_set & in_topic).sum())
            if rnd.get("pool", "outside_topic") == "outside_topic":
                pool &= ~in_topic  # this control never touches the topic's own documents
        elif "n" in rnd:
            n = int(rnd["n"])
        else:
            n = round(rnd["frac"] * in_include_set.sum())
        candidates = np.flatnonzero(pool)
        n = min(n, len(candidates))
        drop = np.random.default_rng(rnd.get("seed", 0)).choice(candidates, size=n, replace=False)
        keep[drop] = False

    rows = np.flatnonzero(keep)
    used = {t: topics[t] for t in _referenced_topics(cfg)}
    return Snapshot(
        name=cfg["name"],
        cfg=cfg,
        cfg_hash=stable_hash({"cfg": cfg, "topics": used}),
        rows=rows,
        ids_hash=ids_sha256(df["arxiv_id"].to_numpy()[rows]),
    )


def resolve_by_name(df: pd.DataFrame, name: str, topics: dict[str, dict] | None = None) -> Snapshot:
    return resolve_snapshot(df, load_snapshot_config(name), topics or load_topics())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List snapshots and how many documents each contains.")
    parser.add_argument("names", nargs="*", help="snapshot names (default: all in configs/snapshots)")
    args = parser.parse_args(argv)

    df, topics = load_corpus(), load_topics()
    rows = []
    for name in args.names or list_snapshot_names():
        snap = resolve_by_name(df, name, topics)
        years = df["year"].to_numpy()[snap.rows]
        rows.append(
            {
                "snapshot": snap.name,
                "n_docs": snap.n_docs,
                "years": f"{years.min()}-{years.max()}" if len(years) else "-",
                "cfg_hash": snap.cfg_hash[:8],
                "ids_hash": snap.ids_hash[:8],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
