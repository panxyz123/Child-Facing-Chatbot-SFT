"""Chat client for the generator: an OpenAI-compatible endpoint (vLLM / llama.cpp / Ollama serving the fine-tuned Qwen).

The server applies the model's chat template (ChatML for Qwen) on the server, so the messages below become the same prompt format the model
was trained on (system prompt with the set age, then the conversation)."""
from typing import Callable, Dict, List


class ChatLLM:
    def __init__(self, base_url, model, temperature, top_p, top_k, repetition_penalty, max_tokens, timeout=60):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        self.params = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.extra = dict(top_k=top_k, repetition_penalty=repetition_penalty)   # vLLM sampling extensions

    def chat(self, messages: List[Dict[str, str]]):
        """Returns (text, usage) with usage = {prompt_tokens, completion_tokens, finish_reason} from the server."""
        out = self.client.chat.completions.create(model=self.model, messages=messages, extra_body=self.extra, **self.params)
        usage = None
        if getattr(out, "usage", None) is not None:
            usage = dict(prompt_tokens=out.usage.prompt_tokens, completion_tokens=out.usage.completion_tokens)
        if usage is not None:
            usage["finish_reason"] = out.choices[0].finish_reason      # "length" = cut at max_tokens
        return (out.choices[0].message.content or "").strip(), usage

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        return self.chat(messages)[0]


class StubLLM:
    """For tests: a function of the messages, or a fixed list of replies returned in order."""

    def __init__(self, replies=None, fn: Callable = None):
        self.replies, self.fn, self.calls = list(replies or []), fn, []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.fn:
            return self.fn(messages)
        return self.replies.pop(0) if self.replies else "Okay!"

    def chat(self, messages):
        text = self(messages)
        return text, dict(prompt_tokens=sum(len(m["content"].split()) for m in messages), completion_tokens=len(text.split()))
