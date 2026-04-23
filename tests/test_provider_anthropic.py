import json
import os

import responses

from benchkit.providers.anthropic_messages import AnthropicMessagesProvider
from benchkit.types import LLMRequest


@responses.activate
def test_anthropic_provider_builds_payload_and_parses_output():
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    os.environ.pop("ANTHROPIC_BASE_URL", None)

    def callback(request):
        assert request.headers["x-api-key"] == "test-anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.body.decode("utf-8"))
        assert body["model"] == "claude-test"
        assert body["system"] == "System"
        assert body["messages"] == [{"role": "user", "content": "User"}]
        assert body["max_tokens"] == 200
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["format"]["schema"]["required"] == ["label"]
        payload = {
            "content": [{"type": "text", "text": "{\"label\":\"ok\"}"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return (200, {"Content-Type": "application/json"}, json.dumps(payload))

    responses.add_callback(
        responses.POST,
        "https://api.anthropic.com/v1/messages",
        callback=callback,
    )

    provider = AnthropicMessagesProvider()
    req = LLMRequest(
        provider="anthropic",
        payload={
            "model": "claude-test",
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
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 15


def test_anthropic_provider_requires_api_key():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    provider = AnthropicMessagesProvider()
    resp = provider.run(LLMRequest(provider="anthropic", payload={"model": "claude-test", "input": []}))
    assert resp.error is not None
    assert "ANTHROPIC_API_KEY" in resp.error.message
