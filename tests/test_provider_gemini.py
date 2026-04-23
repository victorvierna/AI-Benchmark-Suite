import json
import os

import responses

from benchkit.providers.gemini_generate_content import GeminiGenerateContentProvider
from benchkit.types import LLMRequest


@responses.activate
def test_gemini_provider_builds_payload_and_parses_output():
    os.environ["GEMINI_API_KEY"] = "test-gemini-key"
    os.environ.pop("GEMINI_BASE_URL", None)

    def callback(request):
        assert request.headers["x-goog-api-key"] == "test-gemini-key"
        body = json.loads(request.body.decode("utf-8"))
        assert body["systemInstruction"]["parts"][0]["text"] == "System"
        assert body["contents"][0]["role"] == "user"
        assert body["generationConfig"]["maxOutputTokens"] == 200
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseJsonSchema"]["required"] == ["label"]
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "{\"label\":\"ok\"}"}],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        return (200, {"Content-Type": "application/json"}, json.dumps(payload))

    responses.add_callback(
        responses.POST,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent",
        callback=callback,
    )

    provider = GeminiGenerateContentProvider()
    req = LLMRequest(
        provider="gemini",
        payload={
            "model": "gemini-test",
            "input": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                    },
                }
            },
            "max_output_tokens": 200,
            "temperature": 0,
        },
    )
    resp = provider.run(req)
    assert resp.text == "{\"label\":\"ok\"}"
    assert resp.usage.total_tokens == 15


def test_gemini_provider_requires_api_key():
    os.environ.pop("GEMINI_API_KEY", None)
    provider = GeminiGenerateContentProvider()
    resp = provider.run(LLMRequest(provider="gemini", payload={"model": "gemini-test", "input": []}))
    assert resp.error is not None
    assert "GEMINI_API_KEY" in resp.error.message
