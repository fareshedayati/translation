from __future__ import annotations


class Provider:
    def translate(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def __init__(self, api_key: str, model: str, max_tokens: int, effort: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key, max_retries=5), model
        self.max_tokens, self.effort = max_tokens, effort

    def translate(self, system: str, user: str) -> str:
        kwargs = {"model": self.model, "instructions": system, "input": user, "max_output_tokens": self.max_tokens}
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        response = self.client.responses.create(**kwargs)
        return (response.output_text or "").strip()


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, model: str, max_tokens: int, effort: str):
        from anthropic import Anthropic
        self.client, self.model = Anthropic(api_key=api_key, max_retries=5), model
        self.max_tokens, self.effort = max_tokens, effort

    def translate(self, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        response = self.client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


def make_provider(name: str, api_key: str, model: str, max_tokens: int, effort: str) -> Provider:
    if not api_key:
        raise ValueError("API key is empty; set the configured environment variable or use --dry-run")
    if name == "openai":
        return OpenAIProvider(api_key, model, max_tokens, effort)
    if name == "anthropic":
        return AnthropicProvider(api_key, model, max_tokens, effort)
    raise ValueError("provider.name must be 'openai' or 'anthropic'")
