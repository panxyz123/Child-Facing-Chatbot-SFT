"""All agent settings in one place. Values marked TBD are placeholders until the retrieval benchmark
(rag/retrieval_benchmark.py) picks the method and the relevance threshold."""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

AGES = ["0-3", "4-6", "7-9", "10-13", "14-18"]

# FKGL of the training references per age group (25th, 75th percentile; from the GRPO notebook, v2 train split)
FKGL_BAND: Dict[str, Tuple[float, float]] = {
    "0-3": (1.09, 4.06), "4-6": (1.85, 4.64), "7-9": (5.93, 8.12), "10-13": (7.59, 10.0), "14-18": (9.55, 13.26),
}


@dataclass
class AgentConfig:
    # generation: an OpenAI-compatible server (vLLM) serving the fine-tuned model
    llm_base_url: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_model: str = os.environ.get("VLLM_MODEL", "sprout")       # vLLM --served-model-name or --lora-modules name
    temperature: float = 0.8                                     # same sampling as training and evaluation
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.05
    max_tokens: int = 160                                        # GRPO max_completion_length
    history_turns: int = 3                                       # at most this many previous user/assistant pairs
    context_tokens: int = 2048                                   # TinyLlama's context window
    tokenizer: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"        # for prompt token budgeting

    # tracing: one JSON line per request (no raw text unless log_text)
    trace_path: Optional[str] = os.environ.get("SPROUT_TRACE")  # e.g. logs/traces.jsonl; None = no file
    log_text: bool = False

    # routing
    router_path: str = os.environ.get("SPROUT_ROUTER", "models/router.joblib")
    knowledge_threshold: float = 0.4     # P(needs knowledge) above this goes to retrieval; low because a missed
                                         # knowledge question risks a hallucination, an extra retrieval only costs latency

    # retrieval
    db_path: str = os.environ.get("LANCE_STORAGE_PATH", "rag/lancedb_data")
    table: str = os.environ.get("LANCE_TABLE_NAME", "wiki_science_v2_fixed")
    retrieval_method: str = "hybrid_rerank"   # TBD: vector | fts | hybrid | hybrid_rerank
    hybrid_alpha: float = 0.5                 # TBD: weight on the full-text score
    top_k_docs: int = 2                       # passages in the prompt (TinyLlama has a 2048-token context)
    passage_words: int = 120                  # each passage cut to this many words
    gate_threshold: Optional[float] = None    # TBD: min top-1 reranker score to use the passages; None = always use

    # output checks
    fkgl_too_hard_margin: float = 3.0         # answer FKGL above the age group's 75th percentile + margin is too hard
    min_words_for_fkgl: int = 20              # FKGL is too noisy below this
    max_regenerations: int = 1
    crisis_line: str = "988"                  # US Suicide & Crisis Lifeline (call or text); locale-specific

    fkgl_band: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(FKGL_BAND))


def system_prompt(age: str) -> str:
    """Exactly the system prompt used in SFT and GRPO training."""
    return (f"You are talking to a child aged {age}. "
            f"Generate a friendly response with age-appropriate knowledge.")
