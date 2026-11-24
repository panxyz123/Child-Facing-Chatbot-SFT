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
    # generation: an OpenAI-compatible server (vLLM, llama.cpp, Ollama) serving Qwen2.5-1.5B-Instruct + the Sprout LoRA
    llm_base_url: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_model: str = os.environ.get("VLLM_MODEL", "sprout")       # vLLM --served-model-name or --lora-modules name
    temperature: float = 0.3                                     # 0.8 -> 0.3 cut false claims 25% -> 15% (eval/README.md)
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.05
    max_tokens: int = 160                                        # GRPO max_completion_length
    history_turns: int = 3                                       # at most this many previous user/assistant pairs
    context_tokens: int = 2048                                   # serving window (--max-model-len); Qwen allows more
    tokenizer: str = "Qwen/Qwen2.5-1.5B-Instruct"                # for prompt token budgeting

    # tracing: one JSON line per request (no raw text unless log_text)
    trace_path: Optional[str] = os.environ.get("SPROUT_TRACE")  # e.g. logs/traces.jsonl; None = no file
    log_text: bool = False

    # routing
    router_path: str = os.environ.get("SPROUT_ROUTER", "models/router.joblib")
    knowledge_threshold: float = 0.4     # P(needs knowledge) above this goes to retrieval; low because a missed
                                         # knowledge question risks a hallucination, an extra retrieval only costs latency

    # retrieval
    db_path: str = os.environ.get("LANCE_STORAGE_PATH", "rag/lancedb_data")
    table: str = os.environ.get("LANCE_TABLE_NAME", "wiki_v4_para")   # paragraph chunks
    retrieval_method: str = "hybrid_rerank"   # vector | fts | hybrid | hybrid_rerank (best: results/retrieval_v4)
    rerank_pool: int = 20                     # hybrid candidates the cross-encoder reorders
    ann_nprobes: int = 200                    # IVF-PQ partitions searched: recall@20 0.975 vs exact, p50 30 ms vs 667 ms
    ann_refine: int = 20                      # re-score 20x candidates with full vectors (results/retrieval_v4/ann_index_high.json)
    hybrid_alpha: float = 0.5                 # weight on the full-text score
    top_k_docs: int = 2                       # passages in the prompt (2048-token serving window)
    passage_words: int = 150                  # each passage cut to this many words (= the chunk size)
    gate_threshold: Optional[float] = 2.0     # min top-1 reranker logit to use the passages (keeps ~78%, rag/README.md)

    # output checks
    fkgl_too_hard_margin: float = 3.0         # answer FKGL above the age group's 75th percentile + margin is too hard
    min_words_for_fkgl: int = 20              # FKGL is too noisy below this
    max_regenerations: int = 1
    crisis_line: str = "988"                  # US Suicide & Crisis Lifeline (call or text); locale-specific

    # output guard model (second output layer after the rules): Qwen3Guard-Gen judges message + answer.
    # None = rules only. Chosen on the dev split of the safety suite (results/guardrails/guard_models/).
    guard_url: Optional[str] = os.environ.get("SPROUT_GUARD_URL")      # e.g. http://localhost:8090/v1
    guard_model: str = os.environ.get("SPROUT_GUARD_MODEL", "guard")
    guard_tokenizer: str = "Qwen/Qwen3Guard-Gen-0.6B"   # its chat template builds the moderation prompt
    guard_controversial_unsafe: bool = True   # "Controversial" is blocked too: stricter policy for children
    guard_fail_closed: bool = True            # guard unreachable or unparseable -> safe reply, never the raw answer
    guard_timeout: float = 10.0

    fkgl_band: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(FKGL_BAND))


def system_prompt(age: str) -> str:
    """Exactly the system prompt used in SFT and GRPO training."""
    return (f"You are talking to a child aged {age}. "
            f"Generate a friendly response with age-appropriate knowledge.")
