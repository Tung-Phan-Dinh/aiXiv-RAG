"""Frozen, rule-based topic labels.

A topic is a list of regexes in configs/topics.yaml; a document belongs to the topic if any
regex matches its title or abstract (case-insensitive). The rules are deterministic, so
"remove topic X" means the same documents in every run. The rules (not the labels) are what
get hashed into a snapshot's provenance - change a rule and the snapshot hash changes.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_topics
from .data import load_corpus


def topic_regex(spec: dict) -> str:
    return "|".join(f"(?:{p})" for p in spec["any_of"])


def topic_mask(df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Boolean array (len(df),): True where the document matches the topic."""
    text = df["title"] + " " + df["abstract"]
    return text.str.contains(topic_regex(spec), case=False, regex=True).to_numpy(dtype=bool)


def topic_table(df: pd.DataFrame, topics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, spec in topics.items():
        mask = topic_mask(df, spec)
        years = df.loc[mask, "year"]
        rows.append(
            {
                "topic": name,
                "n_docs": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "first_year": int(years.min()) if len(years) else None,
                "last_year": int(years.max()) if len(years) else None,
                "median_year": float(years.median()) if len(years) else None,
            }
        )
    return pd.DataFrame(rows)


def topic_overlap(df: pd.DataFrame, topics: dict[str, dict]) -> pd.DataFrame:
    """Square table: cell (row A, column B) = share of topic A's documents that are also in topic B.

    Topics are not disjoint, so removing topic B also removes (some of) topic A's evidence. A large
    off-diagonal cell is a reason to expect spill-over between the two topics' questions."""
    names = list(topics)
    masks = {n: topic_mask(df, topics[n]) for n in names}
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        size = max(int(masks[a].sum()), 1)
        for b in names:
            out.loc[a, b] = float((masks[a] & masks[b]).sum()) / size
    out.index.name = "share of row also in column"
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show how many documents each topic rule matches.")
    parser.add_argument("--min-docs", type=int, default=150, help="warn if a topic has fewer docs")
    args = parser.parse_args(argv)

    df = load_corpus()
    topics = load_topics()
    table = topic_table(df, topics)
    print(table.to_string(index=False))
    print("\nOverlap (share of the ROW topic's documents that are also in the COLUMN topic):")
    print(topic_overlap(df, topics).round(2).to_string())
    small = table[table["n_docs"] < args.min_docs]
    for name in small["topic"]:
        print(f"WARNING: topic {name!r} matches < {args.min_docs} docs; removal effects will be noisy")


if __name__ == "__main__":
    main()
