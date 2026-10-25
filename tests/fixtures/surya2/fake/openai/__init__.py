"""Fake OpenAI client: chat completions over HTTP with only the fields the adapter reads."""

import json
import urllib.error
import urllib.request
from types import SimpleNamespace


class APIStatusError(Exception):
    pass


class _Completions:
    def __init__(self, base_url):
        self.base_url = base_url

    def create(self, **request):
        data = json.dumps(request).encode()
        http = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(http) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise APIStatusError(f"Error code: {error.code} - {error.read().decode()}") from None
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=c["message"]["content"]),
                    finish_reason=c["finish_reason"],
                )
                for c in body["choices"]
            ],
            usage=SimpleNamespace(**body["usage"]),
        )


class OpenAI:
    def __init__(self, api_key, base_url, max_retries=2, timeout=None):
        assert max_retries == 0  # a failed request is not silently repeated
        self.chat = SimpleNamespace(completions=_Completions(base_url))
