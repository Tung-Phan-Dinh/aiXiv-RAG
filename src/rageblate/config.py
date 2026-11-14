"""Loading YAML configs, a tiny .env reader, and stable hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS, ROOT, SNAPSHOT_CONFIGS

SNAPSHOT_KEYS = {"name", "description", "include", "exclude"}
INCLUDE_KEYS = {"year_min", "year_max", "categories_any"}
EXCLUDE_KEYS = {"topic", "random"}
RANDOM_KEYS = {"n", "frac", "matched_to_topic", "seed", "pool"}
RANDOM_POOLS = {"outside_topic", "all"}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_yaml(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(obj: Any) -> str:
    """sha256 of a JSON-serialisable object, independent of key order."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_embed_config() -> dict:
    return load_yaml(CONFIGS / "embed.yaml")


def load_rag_config() -> dict:
    return load_yaml(CONFIGS / "rag.yaml")


def load_topics() -> dict[str, dict]:
    topics = load_yaml(CONFIGS / "topics.yaml")["topics"]
    for name, spec in topics.items():
        if not spec.get("any_of"):
            raise ValueError(f"topic {name!r} needs a non-empty any_of list")
    return topics


def load_questions() -> list[dict]:
    questions = load_yaml(CONFIGS / "questions.yaml")["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in questions.yaml")
    for q in questions:
        if not q.get("question"):
            raise ValueError(f"question {q['id']} has no text")
        q.setdefault("topic", None)
    return questions


def list_snapshot_names() -> list[str]:
    return sorted(p.stem for p in SNAPSHOT_CONFIGS.glob("*.yaml"))


def _check_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:  # catch typos like `year_mx` instead of silently ignoring them
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")


def load_snapshot_config(name_or_path: str) -> dict:
    """Load and validate one snapshot config (by name in configs/snapshots/, or by path)."""
    path = Path(name_or_path)
    if not (path.suffix == ".yaml" and path.exists()):
        path = SNAPSHOT_CONFIGS / f"{name_or_path}.yaml"
    cfg = load_yaml(path) or {}
    cfg.setdefault("name", path.stem)
    cfg["include"] = cfg.get("include") or {}
    cfg["exclude"] = cfg.get("exclude") or {}
    _check_keys(cfg, SNAPSHOT_KEYS, str(path))
    _check_keys(cfg["include"], INCLUDE_KEYS, f"{path}: include")
    _check_keys(cfg["exclude"], EXCLUDE_KEYS, f"{path}: exclude")
    rnd = cfg["exclude"].get("random")
    if rnd:
        _check_keys(rnd, RANDOM_KEYS, f"{path}: exclude.random")
        if sum(k in rnd for k in ("n", "frac", "matched_to_topic")) != 1:
            raise ValueError(f"{path}: exclude.random needs exactly one of n / frac / matched_to_topic")
        if rnd.get("pool", "outside_topic") not in RANDOM_POOLS:
            raise ValueError(f"{path}: exclude.random.pool must be one of {sorted(RANDOM_POOLS)}")
        if "pool" in rnd and "matched_to_topic" not in rnd:
            raise ValueError(f"{path}: exclude.random.pool only applies with matched_to_topic")
    return cfg
