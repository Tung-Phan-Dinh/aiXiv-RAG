import json
from types import SimpleNamespace

import pytest

from rageblate import llm as llm_mod
from rageblate import rag
from rageblate.llm import IncompleteResponseError, MissingAPIKeyError, OpenAILLM
from rageblate.retrieve import Hit


def hit(i, score=0.9, aid=None, year=2010):
    return Hit(i, i, aid or f"0704.000{i}", year, score, f"title {i}", f"First sentence {i}. Second sentence {i}.")


def test_extract_citations_formats():
    text = "A [0704.0001]. B [physics/0501001, 0704.0002v2]. C [hep-th/9901001; math.AG/0001001] D [1] [see note]."
    assert rag.extract_citations(text) == ["0704.0001", "physics/0501001", "0704.0002", "hep-th/9901001", "math.AG/0001001"]
    assert rag.extract_citations("no brackets 0704.0001") == []
    assert rag.extract_citations("[0704.0001] again [0704.0001]") == ["0704.0001"]  # de-duplicated


def test_is_abstain_uses_sentinel_on_first_line():
    assert rag.is_abstain("INSUFFICIENT_EVIDENCE\nNothing about this topic.")
    assert rag.is_abstain("\n  INSUFFICIENT_EVIDENCE")
    assert not rag.is_abstain("The context says INSUFFICIENT_EVIDENCE is rare.")
    assert not rag.is_abstain("")


def test_prompt_contains_ids_years_and_question():
    hits = [hit(1), hit(2, year=1999)]
    prompt = rag.build_user_prompt("What is X?", hits)
    assert "Question: What is X?" in prompt and "[0704.0001 | 2010]" in prompt and "[0704.0002 | 1999]" in prompt
    assert rag.ABSTAIN in rag.SYSTEM_PROMPT


def test_extractive_answerer_cites_retrieved_docs_only():
    hits = [hit(1), hit(2), hit(3), hit(4)]
    ans = rag.ExtractiveAnswerer(min_score=0.5, n_sentences=3).answer("q", hits)
    assert not ans.abstained and ans.cited_ids == ["0704.0001", "0704.0002", "0704.0003"]
    assert ans.invalid_cited_ids == [] and "Second sentence" not in ans.text


def test_extractive_answerer_abstains_below_threshold_and_on_empty():
    assert rag.ExtractiveAnswerer(min_score=0.5).answer("q", [hit(1, score=0.2)]).abstained
    assert rag.ExtractiveAnswerer().answer("q", []).abstained


def test_invalid_citations_are_detected():
    ans = rag._finish("Claim [0704.0001] and [9999.9999].", [hit(1)], "x", False)
    assert ans.cited_ids == ["0704.0001", "9999.9999"] and ans.invalid_cited_ids == ["9999.9999"]


# ---------------------------------------------------------------- LLM wrapper (stub client, no network)


class _Resp:
    def __init__(self, text="answer text", status="completed", reason=None, reasoning_tokens=7):
        self.output_text = text
        self.status = status
        self.incomplete_details = SimpleNamespace(reason=reason) if reason else None
        self.model = "stub-model-2026-01-01"
        self.usage = SimpleNamespace(
            input_tokens=1, output_tokens=2, output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens)
        )


class BadRequestError(Exception):  # same class name the SDK uses
    pass


class _Client:
    """Stands in for openai.OpenAI(): the code under test calls `client.responses.create(...)`."""

    def __init__(self, reject_temperature=False, make_resp=_Resp):
        self.calls = []
        self.reject = reject_temperature
        self.make_resp = make_resp
        self.responses = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject and "temperature" in kw:
            raise BadRequestError("Unsupported parameter: 'temperature' is not supported with this model.")
        return self.make_resp()


def test_llm_caches_responses_on_disk(tmp_path):
    client = _Client()
    llm = OpenAILLM("m", max_output_tokens=123, cache_dir=tmp_path, client=client)
    a = llm.complete("sys", "user")
    b = llm.complete("sys", "user")
    c = llm.complete("sys", "other user")
    assert (a.cached, b.cached, c.cached) == (False, True, False)
    assert a.text == b.text == "answer text" and len(client.calls) == 2
    call = client.calls[0]
    assert call["model"] == "m" and call["instructions"] == "sys" and call["input"] == "user"
    assert call["max_output_tokens"] == 123 and call["temperature"] == 0.0 and call["store"] is False
    assert "reasoning" not in call  # no effort configured -> API default
    assert a.usage == {"input_tokens": 1, "output_tokens": 2, "reasoning_tokens": 7}


def test_llm_cache_key_depends_on_model_and_effort(tmp_path):
    client = _Client()
    OpenAILLM("m1", cache_dir=tmp_path, client=client).complete("s", "u")
    other_model = OpenAILLM("m2", cache_dir=tmp_path, client=client).complete("s", "u")
    other_effort = OpenAILLM("m1", reasoning_effort="low", cache_dir=tmp_path, client=client).complete("s", "u")
    assert not other_model.cached and not other_effort.cached and len(client.calls) == 3


def test_llm_sends_reasoning_effort(tmp_path):
    client = _Client()
    OpenAILLM("m", reasoning_effort="low", cache_dir=tmp_path, client=client).complete("s", "u")
    assert client.calls[0]["reasoning"] == {"effort": "low"}


def test_llm_drops_temperature_if_model_rejects_it(tmp_path):
    client = _Client(reject_temperature=True)
    llm = OpenAILLM("m", cache_dir=tmp_path, client=client)
    assert llm.complete("s", "u1").text == "answer text"
    llm.complete("s", "u2")
    assert "temperature" in client.calls[0] and all("temperature" not in c for c in client.calls[1:])


def test_llm_other_errors_propagate(tmp_path):
    class Boom(_Client):
        def create(self, **kw):
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        OpenAILLM("m", cache_dir=tmp_path, client=Boom()).complete("s", "u")


def test_incomplete_response_raises_and_is_not_cached(tmp_path):
    client = _Client(make_resp=lambda: _Resp("half an ans", status="incomplete", reason="max_output_tokens"))
    llm = OpenAILLM("m", cache_dir=tmp_path, client=client)
    for _ in range(2):  # the second attempt must hit the API again: nothing was cached
        with pytest.raises(IncompleteResponseError, match="max_output_tokens"):
            llm.complete("s", "u")
    assert len(client.calls) == 2 and not list(tmp_path.glob("*.json"))


def test_empty_text_raises(tmp_path):
    client = _Client(make_resp=lambda: _Resp("  \n"))  # e.g. all tokens went to reasoning, or a refusal item
    with pytest.raises(IncompleteResponseError):
        OpenAILLM("m", cache_dir=tmp_path, client=client).complete("s", "u")


def test_missing_api_key_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod, "load_dotenv", lambda *a, **k: None)  # never read a real .env in tests
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        OpenAILLM("m", cache_dir=tmp_path).complete("s", "u")


def test_openai_answerer_parses_citations_and_abstention(tmp_path):
    reply = "Heat uptake rises [0704.0001] and [9999.9999]."
    llm = OpenAILLM("m", reasoning_effort="low", cache_dir=tmp_path, client=_Client(make_resp=lambda: _Resp(reply)))
    answerer = rag.OpenAIAnswerer(llm)
    ans = answerer.answer("q", [hit(1)])
    assert ans.cited_ids == ["0704.0001", "9999.9999"] and ans.invalid_cited_ids == ["9999.9999"]
    assert not ans.abstained and not ans.cached and ans.answerer == "openai:m:low:v1"
    assert answerer.answer("q", [hit(1)]).cached  # second identical call is served from disk

    abstaining = _Client(make_resp=lambda: _Resp("INSUFFICIENT_EVIDENCE\nNothing about this."))
    ans = rag.OpenAIAnswerer(OpenAILLM("m2", cache_dir=tmp_path, client=abstaining)).answer("q", [hit(1)])
    assert ans.abstained and ans.cited_ids == []


def test_ask_explains_out_of_credits_without_a_traceback():
    from rageblate.ask import _explain

    class FakeAPIError(Exception):
        code, type, message = "credit_balance_exhausted", "insufficient_quota", "no credits"

    assert "no API credits left" in _explain(FakeAPIError())
    assert "--answerer extractive" in _explain(MissingAPIKeyError("OPENAI_API_KEY is not set"))


# --- local model (Ollama) -------------------------------------------------------------------------


class _FakeOllama:
    """Stands in for Ollama's HTTP API: records the payloads it was sent."""

    def __init__(self, content="An answer [0704.0001].", done_reason="stop", prompt_tokens=2500):
        self.content, self.done_reason, self.prompt_tokens = content, done_reason, prompt_tokens
        self.chats = []

    def __call__(self, url, payload):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:14b", "digest": "sha256:abc"}]}
        self.chats.append(payload)
        return {
            "message": {"role": "assistant", "content": self.content},
            "done_reason": self.done_reason,
            "prompt_eval_count": self.prompt_tokens,
            "eval_count": 40,
        }


def _ollama(tmp_path, server, **kw):
    from rageblate.llm import OllamaLLM

    return OllamaLLM("qwen3:14b", host="homepc", cache_dir=tmp_path, post=server, **kw)


def test_ollama_sends_context_window_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("RAGEBLATE_OLLAMA_HOST", raising=False)
    server = _FakeOllama()
    llm = _ollama(tmp_path, server, num_ctx=8192, think=False)
    assert llm.host == "http://homepc:11434" and not llm.paid

    a, b = llm.complete("sys", "user"), llm.complete("sys", "user")
    assert not a.cached and b.cached and a.text == b.text and len(server.chats) == 1
    sent = server.chats[0]
    assert sent["options"]["num_ctx"] == 8192 and sent["think"] is False and sent["stream"] is False
    rec = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert rec["provider"] == "ollama" and rec["model_digest"] == "sha256:abc"

    llm.complete("sys", "user", sample=2)
    assert server.chats[-1]["options"]["seed"] == 2  # replicate draws get their own seed


def test_ollama_and_openai_never_share_a_cache_entry(tmp_path):
    same_name = OpenAILLM("qwen3:14b", cache_dir=tmp_path, client=_Client())
    assert _ollama(tmp_path, _FakeOllama()).key("s", "u") != same_name.key("s", "u")


def test_ollama_refuses_prompts_that_would_be_silently_truncated(tmp_path):
    server = _FakeOllama()
    with pytest.raises(IncompleteResponseError, match="num_ctx"):
        _ollama(tmp_path, server, num_ctx=1000).complete("s", "x" * 10_000)
    assert server.chats == []  # refused before sending

    with pytest.raises(IncompleteResponseError, match="truncated"):  # server filled the whole window
        _ollama(tmp_path, _FakeOllama(prompt_tokens=8192), num_ctx=8192).complete("s", "u")
    with pytest.raises(IncompleteResponseError):  # output cut off
        _ollama(tmp_path, _FakeOllama(done_reason="length")).complete("s", "u")
    assert not list(tmp_path.glob("*.json"))  # failures are never cached


def test_ollama_answerer_name_and_unreachable_server(tmp_path):
    from rageblate.llm import LocalServerError, OllamaLLM

    ans = rag.OpenAIAnswerer(_ollama(tmp_path, _FakeOllama())).answer("q", [hit(1)])
    assert ans.answerer == "ollama:qwen3:14b:no-think:v1" and ans.cited_ids == ["0704.0001"]

    dead = OllamaLLM("m", host="127.0.0.1:9", timeout=2, cache_dir=tmp_path)  # port 9: nothing listens
    with pytest.raises(LocalServerError, match="Tailscale"):
        dead.complete("s", "u")
