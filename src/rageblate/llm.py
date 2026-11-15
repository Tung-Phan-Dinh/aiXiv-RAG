"""Thin LLM wrappers with an on-disk response cache: OpenAI (Responses API) and a local Ollama server.

Every request is keyed by (provider, model, system, user, max_output_tokens, temperature,
reasoning_effort) plus, for replicate draws, a `sample` index. Sample 0 is keyed exactly as
before the replicate feature existed, so earlier caches stay valid; samples >= 1 are extra,
independently-billed draws of the same prompt, used to measure the model's own run-to-run noise
(the "noise floor"). Re-running an ablation therefore costs nothing and returns byte-identical
answers - which is what makes runs reproducible and diffs between snapshots meaningful.

Two failure modes are made loud instead of silent:
  * a response that is not `completed` (typically: hidden reasoning tokens used up
    max_output_tokens) raises IncompleteResponseError and is NOT cached - a truncated answer
    would otherwise look like a corpus effect in the metrics;
  * a model that rejects `temperature` gets it dropped (once, remembered), because reasoning
    models do not accept sampling parameters. Fresh calls are then not deterministic; the
    cache is what keeps a finished run reproducible.

`OllamaLLM` has the same interface (key / is_cached / complete) for a model served by Ollama, e.g.
on a home PC reached over Tailscale. Its cache keys carry provider "ollama", so local and OpenAI
answers can never be mixed up, even under the same model name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import load_dotenv
from .paths import CACHE

PROVIDER = "openai"
CHARS_PER_TOKEN = 3.5  # conservative for English + arXiv ids: overestimates tokens, so the context check errs safe


class MissingAPIKeyError(RuntimeError):
    """OPENAI_API_KEY is not configured."""


class IncompleteResponseError(RuntimeError):
    """The API returned no usable answer (status != completed, or empty text)."""


class LocalServerError(RuntimeError):
    """The local model server is unreachable or rejected the request."""


def _write_cache(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)  # atomic: no half-written cache files


@dataclass
class LLMResult:
    text: str
    cached: bool
    model: str
    usage: dict


class OpenAILLM:
    provider = PROVIDER
    paid = True

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 3000,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        cache_dir: Path = CACHE / "llm",
        client=None,
    ):
        self.model, self.max_output_tokens, self.temperature = model, max_output_tokens, temperature
        self.reasoning_effort = reasoning_effort
        self.cache_dir = Path(cache_dir)
        self._client = client
        self._send_temperature = temperature is not None

    def _get_client(self):
        if self._client is None:
            load_dotenv()
            if not os.environ.get("OPENAI_API_KEY"):
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set. Put OPENAI_API_KEY=... in .env (git-ignored) "
                    "or export it. For an offline run use --answerer extractive."
                )
            from openai import OpenAI

            self._client = OpenAI(max_retries=5)
        return self._client

    def key(self, system: str, user: str, sample: int = 0) -> str:
        fields = {
            "provider": PROVIDER,
            "model": self.model,
            "system": system,
            "user": user,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        if sample:  # only replicates get the field, so sample 0 keeps its pre-replicate key
            fields["sample"] = int(sample)
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()

    _key = key  # backwards-compatible name

    def is_cached(self, system: str, user: str, sample: int = 0) -> bool:
        return (self.cache_dir / f"{self.key(system, user, sample)}.json").exists()

    def complete(self, system: str, user: str, sample: int = 0) -> LLMResult:
        path = self.cache_dir / f"{self.key(system, user, sample)}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(rec["text"], True, rec["model"], rec.get("usage", {}))

        client = self._get_client()
        kwargs = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.max_output_tokens,
            "store": False,  # do not keep the request on OpenAI's side
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        try:
            resp = client.responses.create(**kwargs, **({"temperature": self.temperature} if self._send_temperature else {}))
        except Exception as e:  # reasoning models reject sampling parameters
            if type(e).__name__ == "BadRequestError" and "temperature" in str(e).lower() and self._send_temperature:
                self._send_temperature = False
                resp = client.responses.create(**kwargs)
            else:
                raise

        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        reasoning_tokens = getattr(getattr(resp.usage, "output_tokens_details", None), "reasoning_tokens", None)
        if reasoning_tokens is not None:
            usage["reasoning_tokens"] = reasoning_tokens

        status = getattr(resp, "status", "completed")
        text = resp.output_text or ""
        if status != "completed" or not text.strip():
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            raise IncompleteResponseError(
                f"{self.model} returned no usable answer (status={status}, reason={reason}, usage={usage}). "
                "Raise max_output_tokens or lower reasoning_effort in configs/rag.yaml, then re-run: "
                "calls that already completed are cached and cost nothing."
            )

        _write_cache(
            path,
            {
                "text": text,
                "model": self.model,
                "response_model": getattr(resp, "model", None),  # the dated snapshot the alias resolved to
                "usage": usage,
                "status": status,
                "temperature_sent": self._send_temperature,
                "reasoning_effort": self.reasoning_effort,
                "sample": int(sample),
            },
        )
        return LLMResult(text, False, self.model, usage)


def _normalise_host(host: str) -> str:
    """`homepc`, `homepc:11434` or `http://homepc:11434/` -> `http://homepc:11434`."""
    host = host.strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    if host.count(":") == 1:  # only the scheme's colon: no port given
        host += ":11434"
    return host


RETRY_WAITS = (10, 30, 60)  # seconds before each retry of an unreachable server (~100 s in all)


def _http_post_json(url: str, payload: dict | None, timeout: float, retries: int = len(RETRY_WAITS)) -> dict:
    """POST (or GET when payload is None) JSON with the standard library; raises LocalServerError.

    Only *connection* failures are retried (the PC sleeping, Tailscale reconnecting): the request never
    reached the model, so a retry cannot change the answer. An HTTP error from the server is not retried."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LocalServerError(f"Ollama returned HTTP {e.code} for {url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
                print(f"  Ollama unreachable ({e}); retry {attempt + 1}/{retries} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise LocalServerError(
                f"cannot reach Ollama at {url} ({e}). Is Tailscale up on both machines, and is Ollama listening on "
                "the Tailscale interface (OLLAMA_HOST=0.0.0.0 on the PC)?"
            ) from None


class OllamaLLM:
    """A model served by Ollama's native /api/chat (not its OpenAI-compatible endpoint: only the native
    API lets us set `num_ctx` per request).

    Why num_ctx matters: Ollama's default context window is small and an over-long prompt is
    truncated *silently*. Our prompts (question + k abstracts) are ~2.5k tokens, so a truncated
    prompt would drop abstracts and look like a corpus effect. Two guards make that loud:
    a pre-flight size check, and a post-hoc check that the server did not fill the whole window.

    Replicate draws (`sample` >= 1) use seed = seed + sample. With temperature 0 decoding is greedy,
    so replicates are identical by construction: the noise floor is then 0, not measured.
    """

    provider = "ollama"
    paid = False

    def __init__(
        self,
        model: str,
        host: str | None = None,
        num_ctx: int = 8192,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        think: bool | None = False,
        seed: int = 0,
        timeout: float = 600.0,
        retries: int = len(RETRY_WAITS),
        cache_dir: Path = CACHE / "llm",
        post=None,
    ):
        load_dotenv()
        self.model, self.num_ctx, self.max_output_tokens = model, num_ctx, max_output_tokens
        self.temperature, self.think, self.seed, self.timeout = temperature, think, seed, timeout
        # .env wins over the config, so the PC's Tailscale name need not be committed
        self.host = _normalise_host(os.environ.get("RAGEBLATE_OLLAMA_HOST") or host or "localhost")
        # plan.usage_stats and the answerer name read this; for Ollama it records thinking on/off
        self.reasoning_effort = {True: "think", False: "no-think", None: None}[think]
        self.cache_dir = Path(cache_dir)
        self._post = post or (lambda url, payload: _http_post_json(url, payload, self.timeout, retries))
        self._digest: str | None = None

    def key(self, system: str, user: str, sample: int = 0) -> str:
        fields = {
            "provider": self.provider,
            "model": self.model,
            "system": system,
            "user": user,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
        }
        if sample:
            fields["sample"] = int(sample)
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()

    def is_cached(self, system: str, user: str, sample: int = 0) -> bool:
        return (self.cache_dir / f"{self.key(system, user, sample)}.json").exists()

    def model_digest(self) -> str | None:
        """The exact weights behind the tag (provenance: a re-pulled tag can point at new weights)."""
        if self._digest is None:
            try:
                tags = self._post(f"{self.host}/api/tags", None)
                self._digest = next((m.get("digest") for m in tags.get("models", []) if m.get("name") == self.model), "")
            except LocalServerError:
                self._digest = ""
        return self._digest or None

    def complete(self, system: str, user: str, sample: int = 0) -> LLMResult:
        path = self.cache_dir / f"{self.key(system, user, sample)}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(rec["text"], True, rec["model"], rec.get("usage", {}))

        est_prompt = (len(system) + len(user)) / CHARS_PER_TOKEN
        if est_prompt + self.max_output_tokens > self.num_ctx:
            raise IncompleteResponseError(
                f"prompt (~{est_prompt:,.0f} tokens) + max_output_tokens ({self.max_output_tokens}) does not fit "
                f"num_ctx={self.num_ctx}; Ollama would silently truncate it. Raise ollama.num_ctx in configs/rag.yaml."
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_output_tokens,
                "temperature": self.temperature,
                "seed": self.seed + int(sample),
            },
            "keep_alive": "30m",  # keep the model in VRAM between calls
        }
        if self.think is not None:  # omitted for models that have no thinking mode
            payload["think"] = self.think
        resp = self._post(f"{self.host}/api/chat", payload)

        usage = {"input_tokens": resp.get("prompt_eval_count", 0), "output_tokens": resp.get("eval_count", 0)}
        text = ((resp.get("message") or {}).get("content") or "").strip()
        reason = resp.get("done_reason")
        if reason == "length" or not text:
            raise IncompleteResponseError(
                f"{self.model} returned no usable answer (done_reason={reason}, usage={usage}). "
                "Raise ollama.max_output_tokens (and num_ctx) in configs/rag.yaml, or set think: false."
            )
        if usage["input_tokens"] >= self.num_ctx - 16:
            raise IncompleteResponseError(
                f"the prompt filled the whole context window ({usage['input_tokens']} >= num_ctx={self.num_ctx}): "
                "Ollama has probably truncated it. Raise ollama.num_ctx in configs/rag.yaml."
            )

        _write_cache(
            path,
            {
                "text": text,
                "model": self.model,
                "provider": self.provider,
                "host": self.host,
                "model_digest": self.model_digest(),
                "usage": usage,
                "status": reason,
                "num_ctx": self.num_ctx,
                "think": self.think,
                "reasoning_effort": self.reasoning_effort,
                "sample": int(sample),
            },
        )
        return LLMResult(text, False, self.model, usage)
