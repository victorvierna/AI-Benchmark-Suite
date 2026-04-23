import os

import responses

from benchkit.providers.lmstudio_responses import LMStudioResponsesProvider
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
def test_lmstudio_provider_parses_output():
    os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:1234/v1"
    os.environ.pop("LMSTUDIO_API_KEY", None)

    responses.add(
        responses.POST,
        "http://localhost:1234/v1/responses",
        json=_make_response_payload("{\"label\":\"ok\"}"),
        status=200,
    )

    provider = LMStudioResponsesProvider()
    req = LLMRequest(provider="lmstudio", payload={"model": "local-model", "input": []})
    resp = provider.run(req)
    assert resp.text == "{\"label\":\"ok\"}"
    assert resp.usage.total_tokens == 15
