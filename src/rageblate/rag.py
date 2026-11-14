"""Prompting, citation parsing and abstention detection (the "G" in RAG).

Conventions the metrics rely on:
  * Every claim is cited with the arXiv id in square brackets: [0704.0001] or [physics/0501001].
  * If the context cannot support an answer, the model must start with the prompt's sentinel
    (INSUFFICIENT_EVIDENCE in v1, NO_SUPPORTING_EVIDENCE in v2). We detect abstention by that
    sentinel, never by guessing from free text.
  * Two prompt versions exist so a conclusion can be checked against a reworded prompt and a
    different sentinel (`ablate --prompt-version v2`, then `compare-runs`). A prompt is part of the
    LLM cache key, so versions never share answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import OllamaLLM, OpenAILLM
from .retrieve import Hit

PROMPT_VERSION = "v1"  # default
ABSTAIN = "INSUFFICIENT_EVIDENCE"
ABSTAIN_V2 = "NO_SUPPORTING_EVIDENCE"

SYSTEM_PROMPT = f"""You answer scientific questions using ONLY the context documents provided (arXiv abstracts).

Rules:
1. Use only information stated in the context. Do not use outside knowledge.
2. Cite every claim with the arXiv id of the supporting document in square brackets, e.g. [0704.0001] or [physics/0501001]. Cite only ids that appear in the context.
3. If the context does not contain enough evidence to answer the question, reply with exactly {ABSTAIN} on the first line, then one sentence saying what is missing. Do not guess.
4. If documents disagree, say so and cite both sides.
5. Be concise: at most 150 words."""

SYSTEM_PROMPT_V2 = f"""You are a careful research assistant. Answer the question strictly from the arXiv abstracts supplied below; no other source may be used.

Format:
- Give a short answer of 120 words or fewer. After each statement put the id of the abstract that supports it in square brackets, for example [0704.0001] or [physics/0501001]. Never cite an id that is not listed.
- Where abstracts conflict, report both views with their ids.
- If the abstracts do not allow an answer, write {ABSTAIN_V2} alone on the first line, then one sentence naming what is absent. Do not speculate."""

# version -> (system prompt, abstention sentinel)
PROMPTS = {"v1": (SYSTEM_PROMPT, ABSTAIN), "v2": (SYSTEM_PROMPT_V2, ABSTAIN_V2)}

_ID = r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Za-z\-]+)?/\d{7})"
_ID_RE = re.compile(rf"^{_ID}(?:v\d+)?$")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.arxiv_id} | {h.year}] {h.title}\n{h.abstract}" for h in hits)


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    return f"Question: {question}\n\nContext documents:\n\n{format_context(hits)}"


def extract_citations(text: str) -> list[str]:
    """arXiv ids cited in [brackets], de-duplicated in order, version suffix stripped."""
    seen: list[str] = []
    for group in _BRACKET_RE.findall(text):
        for token in re.split(r"[;,\s]+", group.strip()):
            if _ID_RE.match(token):
                token = re.sub(r"v\d+$", "", token)
                if token not in seen:
                    seen.append(token)
    return seen


def is_abstain(text: str, sentinel: str = ABSTAIN) -> bool:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return bool(lines) and lines[0].startswith(sentinel)


@dataclass
class Answer:
    text: str
    abstained: bool
    cited_ids: list[str] = field(default_factory=list)
    invalid_cited_ids: list[str] = field(default_factory=list)  # cited but not in the context
    answerer: str = ""
    cached: bool = False


def _finish(text: str, hits: list[Hit], answerer: str, cached: bool, sentinel: str = ABSTAIN) -> Answer:
    in_context = {h.arxiv_id for h in hits}
    cited = extract_citations(text)
    return Answer(
        text=text,
        abstained=is_abstain(text, sentinel),
        cited_ids=cited,
        invalid_cited_ids=[c for c in cited if c not in in_context],
        answerer=answerer,
        cached=cached,
    )


class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: list[Hit], sample: int = 0) -> Answer: ...


class OpenAIAnswerer:
    """LLM answerer. Works with any client of the OpenAILLM interface (OpenAI or a local Ollama model);
    the provider is the first field of `name`, e.g. `openai:gpt-6.1-sol:low:v1` or `ollama:qwen3:14b:no-think:v1`."""

    def __init__(self, llm: OpenAILLM | OllamaLLM, prompt_version: str = PROMPT_VERSION):
        if prompt_version not in PROMPTS:
            raise ValueError(f"unknown prompt version {prompt_version!r}; choose from {sorted(PROMPTS)}")
        self.llm = llm
        self.prompt_version = prompt_version
        self.system, self.sentinel = PROMPTS[prompt_version]
        self.name = f"{llm.provider}:{llm.model}:{llm.reasoning_effort or 'default'}:{prompt_version}"

    def prompt(self, question: str, hits: list[Hit]) -> tuple[str, str]:
        """(system, user) exactly as sent; equal prompts share one cached response."""
        return self.system, build_user_prompt(question, hits)

    def answer(self, question: str, hits: list[Hit], sample: int = 0) -> Answer:
        res = self.llm.complete(self.system, build_user_prompt(question, hits), sample)
        return _finish(res.text.strip(), hits, self.name, res.cached, self.sentinel)


class ExtractiveAnswerer:
    """Offline baseline, no LLM: first sentence of the top abstracts, cited; abstains when the
    best retrieval score is below a threshold. Lets the whole pipeline run without an API key,
    but its answers are NOT model outputs - do not report them as such."""

    def __init__(self, min_score: float = 0.0, n_sentences: int = 3):
        self.min_score, self.n_sentences = min_score, n_sentences
        self.name = f"extractive:min_score={min_score}"

    def answer(self, question: str, hits: list[Hit], sample: int = 0) -> Answer:  # deterministic: sample is ignored
        if not hits or hits[0].score < self.min_score:
            text = f"{ABSTAIN}\nNo retrieved document is similar enough to the question."
        else:
            parts = []
            for h in hits[: self.n_sentences]:
                first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", h.abstract, maxsplit=1)[0]
                parts.append(f"{first} [{h.arxiv_id}]")
            text = " ".join(parts)
        return _finish(text, hits, self.name, False)
