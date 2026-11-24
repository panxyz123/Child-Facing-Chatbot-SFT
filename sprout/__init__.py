"""Sprout: an age-aware RAG agent for children (fine-tuned Qwen2.5-1.5B served by vLLM).

SproutAgent is imported lazily, so the guardrails, router and config can be used (e.g. by eval/guard_eval.py)
without LangGraph installed."""
from .config import AgentConfig, system_prompt

__all__ = ["SproutAgent", "AgentConfig", "system_prompt"]


def __getattr__(name):
    if name == "SproutAgent":
        from .agent import SproutAgent
        return SproutAgent
    raise AttributeError(name)
