import os

import responses

from benchkit.providers.openai_responses import OpenAIResponsesProvider
from benchkit.types import LLMRequest


def _make_response_payload(text: str):
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


@responses.activate
def test_openai_provider_parses_output():
    os.environ["OPENAI_API_KEY"] = "test"
    provider = OpenAIResponsesProvider()

    responses.add(
        responses.POST,
        "https://api.openai.com/v1/responses",
        json=_make_response_payload("{\"label\":\"ok\"}"),
        status=200,
    )

    req = LLMRequest(provider="openai", payload={"model": "gpt-test", "input": []})
    resp = provider.run(req)
    assert resp.text == "{\"label\":\"ok\"}"
    assert resp.usage.total_tokens == 15


@responses.activate
def test_openai_provider_handles_error():
    os.environ["OPENAI_API_KEY"] = "test"
    provider = OpenAIResponsesProvider()

    responses.add(
        responses.POST,
        "https://api.openai.com/v1/responses",
        json={"error": "bad"},
        status=400,
    )

    req = LLMRequest(provider="openai", payload={"model": "gpt-test", "input": []})
    resp = provider.run(req)
    assert resp.error is not None
