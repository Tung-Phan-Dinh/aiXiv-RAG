"""Stream the arXiv metadata snapshot and write an immutable corpus parquet."""
import json
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

# src/rageblate/ingest.py -> parents[2] is the project root
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "arxiv-metadata-oai-snapshot.json"
OUT = ROOT / "data" / "corpus.parquet"

TARGET_CATEGORIES = {"physics.geo-ph", "physics.ao-ph"}
MAX_LINES = None  # set to e.g. 50_000 while developing


def has_target(categories: str) -> bool:
    """True if any whitespace-separated category is a target (exact match)."""
    return not TARGET_CATEGORIES.isdisjoint(categories.split())


def first_year(versions: list[dict]) -> int:
    """Year of the earliest version. Don't trust list order, so take min()."""
    return min(parsedate_to_datetime(v["created"]) for v in versions).year


def clean_text(s: str) -> str:
    """Collapse all whitespace runs (incl. newlines) to single spaces."""
    return " ".join(s.split())


def parse_record(line: str) -> tuple[dict | None, str | None]:
    """Return (row, None) if kept, or (None, reason) if skipped."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None, "bad_json"

    categories = rec.get("categories") or ""
    if not has_target(categories):
        return None, "not_target"

    try:
        year = first_year(rec["versions"])
    except (KeyError, TypeError, ValueError):  # missing, None, empty, unparsable
        return None, "bad_version"

    abstract = clean_text(rec.get("abstract") or "")
    if not abstract:
        return None, "empty_abstract"

    return {
        "arxiv_id": rec["id"],
        "title": clean_text(rec.get("title") or ""),
        "abstract": abstract,
        "year": year,
        "categories": categories,
        "license": rec.get("license"),
        "update_date": rec.get("update_date") or "",
    }, None


def dedup_latest(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """For a repeated arxiv_id keep the row with the latest update_date.

    Ties go to the later line in the file (_order records the file position).
    Returns (deduplicated frame, number of rows dropped).
    """
    df = df.copy()
    df["_order"] = range(len(df))
    df = df.sort_values(["arxiv_id", "update_date", "_order"])
    n_before = len(df)
    df = df.drop_duplicates(subset="arxiv_id", keep="last").drop(columns="_order")
    return df, n_before - len(df)


def main() -> None:
    rows: list[dict] = []
    skipped: Counter = Counter()
    n_lines = 0

    with RAW.open("r", encoding="utf-8") as f:
        for n_lines, line in enumerate(f, start=1):
            if MAX_LINES is not None and n_lines > MAX_LINES:
                n_lines -= 1
                break
            # Cheap pre-filter: a line without any target string cannot match, so skip the
            # (slow) json.loads. Lines that pass are still checked exactly by has_target().
            if not any(c in line for c in TARGET_CATEGORIES):
                skipped["not_target"] += 1
                continue
            row, reason = parse_record(line)
            if row is None:
                skipped[reason] += 1
            else:
                rows.append(row)

    if not rows:
        print(f"lines read: {n_lines:,}  skipped: {dict(skipped)}")
        raise SystemExit("No rows matched. Is MAX_LINES set, or TARGET_CATEGORIES wrong?")

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int16")

    # The raw file can hold the same arxiv_id twice (e.g. math-ph/0702063): keep the latest.
    df, n_dup_dropped = dedup_latest(df)
    assert df["arxiv_id"].is_unique, "duplicate arxiv_id survived dedup"
    df = df.sort_values("arxiv_id").reset_index(drop=True)  # deterministic output

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # ---- report ----
    print(f"lines read:      {n_lines:,}")
    print(f"rows kept:       {len(df):,}")
    print(f"skipped:         {dict(skipped)}")
    print(f"duplicate ids dropped: {n_dup_dropped}")

    both = df["categories"].str.split().apply(lambda c: TARGET_CATEGORIES <= set(c)).sum()
    print(f"in both target categories: {both:,}")

    print("\nrows per year:")
    print(df["year"].value_counts().sort_index().to_string())

    # sanity check: new-style IDs start with YYMM, so compare with first-version year
    new_style = df["arxiv_id"].str.match(r"^\d{4}\.\d+$")
    id_year = 2000 + df.loc[new_style, "arxiv_id"].str[:2].astype(int)
    # An id is assigned at or after submission, so it can never be EARLIER than the first
    # submission year. The other direction is expected: Dec/Jan boundary, or papers held in
    # moderation and announced months after they were submitted.
    assert (id_year >= df.loc[new_style, "year"]).all(), "an arxiv id predates its first submission"
    mismatch = df.loc[new_style][id_year != df.loc[new_style, "year"]]
    print(f"\nID-year vs first-version-year mismatches: {len(mismatch)}")
    if len(mismatch):
        print(mismatch[["arxiv_id", "year"]].head(5).to_string(index=False))

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
