# Child-Facing Agentic RAG System: from LoRA SFT & RL to vLLM Inference Serving

**Winner of the RapidFire AI SFT Competition** (the original SFT stage, `sft/`)

A child-facing assistant that answers at the reading level of the age a parent sets, grounds factual answers in
retrieved documents, and is served with vLLM on a single T4 GPU. The model is TinyLlama-1.1B-Chat, adapted with
LoRA SFT and GRPO. Every stage is built here end to end: data pipeline, fine-tuning, RL, retrieval, agent,
guardrails, and serving.

## System overview

```
child message + age set by the parent
        |
  input guardrail        unsafe topics, personal information, prompt injection
        |
  router                 knowledge question?  --yes-->  hybrid retrieval (LanceDB) + rerank
        |                                                       |
        +-------------------------------------------------------+
        |
  age-aware generation   TinyLlama + LoRA (SFT + GRPO), served by vLLM
        |
  output guardrail       safety filter, reading-level check against the set age
        |
     response
```

## Components: why each part exists

### 1. Data: age as a condition, not a guess

The original dataset labeled every row's age by asking Gemini to guess it from the child's message. An audit
found those guesses matched the source age only 26.3% of the time (chance: 20%). In the product the age is set by
a parent, so v2 is built that way: for each age group, typical inputs for that age, and a response written for that
age. Every row also carries a `needs_knowledge` label (questions where a wrong fact is a hallucination), which the
router and the hallucination evaluation use.

- 5 balanced age groups (0-3, 4-6, 7-9, 10-13, 14-18), real child speech for the youngest groups
- Gemini structured output with caching, retry with backoff, and quota-aware stopping
- readability outlier filter (FKGL) per age group, duplicate removal, leak-free splits stratified by age and
  knowledge (700 / 100 / 393)

Datasets: [`yxpan/children_sft_dataset`](https://huggingface.co/datasets/yxpan/children_sft_dataset) (v1),
[`yxpan/children_sft_dataset_v2`](https://huggingface.co/datasets/yxpan/children_sft_dataset_v2) (v2).
Pipeline details: [`data/README.md`](data/README.md).

### 2. LoRA SFT: teach the age-conditioned style

TRL `SFTTrainer` with PEFT LoRA and completion-only loss. A config sweep over rank, target modules, learning rate,
LoRA alpha and training length, selected on validation BERTScore, with readability (FKGL distance to the reference)
reported per age group and per question type. A failure analysis on the test split breaks the misses down by
direction (too simple / too hard), truncation and style, which defines what RL should fix.

### 3. GRPO: optimize the reading level directly

SFT imitates references but does not directly optimize "does this answer fit the set age". GRPO (TRL) adds that
as a reward, on top of the SFT model:

- rewards: readability band per age group (from the training references), relevance to the reference
  (embedding similarity), and a format reward against overlong or repetitive answers
- KL penalty to the SFT model, LoRA on the merged SFT weights
- vLLM colocated rollouts, benchmarked against HF `generate`
- evaluation beyond the hold-out set: the same input at all 5 set ages, checking that reading level rises with age
  (Spearman correlation), plus a reward-hacking spot check

### 4. RAG: ground knowledge answers

Small models hallucinate facts, and for children a confident wrong fact is the main risk. Knowledge questions are
answered from a cleaned Wikipedia science corpus:

- corpus cleaning and chunking, LanceDB index
- vector, full-text and hybrid search, with reranking
- hallucination evaluation on the knowledge questions of the test split, with and without retrieval

### 5. Agent and guardrails

A LangGraph agent ties the parts together: route by question type, retrieve only when needed, build an age-aware
prompt with conversation memory, and generate. Input guardrails block unsafe topics and personal information;
output guardrails filter unsafe content and check that the reading level fits the set age.

### 6. Serving: vLLM under realistic load

The fine-tuned model is served with vLLM and measured under the workloads the agent actually produces:

- HF `generate` vs vLLM: per-token latency and throughput under concurrency (continuous batching, PagedAttention)
- prefix caching on shared system prompts, and where it stops helping (requests retrieving different documents)
- KV cache capacity vs context length (GQA: about 22 KB per token)
- realistic output lengths, multi-turn chat and RAG prompts, and SLO-based load tests (TTFT, TPOT)

## Results (single T4)

One run, one seed per number. v1 and v2 use different test sets, so they are not directly comparable.

### SFT on v2

Best config by validation BERTScore: LoRA r=32 on all linear layers, lr 1e-4, 700 steps. 393-row test split:

| | base TinyLlama | SFT |
|---|---|---|
| BERTScore-F1 | 0.854 | 0.880 |
| FKGL within 2 grade levels of the reference | 39.2% | 59.8% |
| mean FKGL gap to the reference | 3.12 | 1.93 |

FKGL within 2 grade levels, by age group:

| | 0-3 | 4-6 | 7-9 | 10-13 | 14-18 |
|---|---|---|---|---|---|
| base | 27.3% | 37.7% | 33.8% | 49.4% | 47.5% |
| SFT | 59.7% | 63.6% | 67.5% | 58.2% | 50.0% |

Knowledge and non-knowledge questions improve about equally (58.9% and 60.7% after SFT). 14-18 barely improves
even with clean age labels, so the teen gap is not only a labeling problem: this is the gap GRPO targets.

### v1 dataset

**SFT** (LoRA r=32 on all linear layers, lr 1e-4, alpha 64), 600-row hold-out:

| | base TinyLlama | SFT |
|---|---|---|
| BERTScore-F1 | 0.848 | 0.883 |
| FKGL within 2 grade levels of the reference | 35.0% | 59.3% |
| mean FKGL gap to the reference | 3.46 | 2.16 |

The 14-18 group did not improve; most rows labeled 14-18 turned out to be toddler utterances, which led to v2.

**GRPO** on top of SFT (250 steps):

| | SFT | SFT + GRPO |
|---|---|---|
| FKGL within 2 grade levels | 59.3% | 61.3% |
| mean FKGL gap | 2.16 | 1.94 |
| BERTScore-F1 | 0.883 | 0.883 |
| age sensitivity: same input at 5 set ages, mean Spearman (set age vs FKGL) | 0.435 | 0.455 |
| share of inputs with Spearman > 0.5 | 47.5% | 57.5% |

The hold-out gain is about one standard error. The policy stayed close to SFT (KL about 0.01), and the reward
(FKGL inside the age group's band) is not the same quantity as the hold-out metric (distance to each row's own
reference); the v2 reward design addresses both.

**GRPO rollouts, HF `generate` vs vLLM** (same prompts and sampling):

| | HF | vLLM | speedup |
|---|---|---|---|
| rollout throughput | 301 tokens/s | 2,440 tokens/s | 8.1x |
| GRPO step time | 8.46 s | 3.78 s | 2.2x |

The end-to-end gain is smaller than the rollout gain because training (forward, backward, reference log-probs) is
not accelerated: with HF, generation was about 60% of a step; with vLLM the bottleneck moves to training.

**Serving** (vLLM vs HF `generate`, base TinyLlama, 128-token outputs):

| | HF | vLLM |
|---|---|---|
| decode time per token (batch 1) | 26.6 ms | 10.6 ms |
| throughput, 8 concurrent requests | 214 tokens/s | 637 tokens/s (3.0x) |
| throughput, 32 concurrent requests | 950 tokens/s | 1,541 tokens/s (1.6x) |

Prefix caching cut time-to-first-token from 305 ms to 48 ms (6.3x) with 32 requests sharing a ~1.1k-token prefix,
an upper bound since requests retrieving different documents share much less. With fixed output lengths static
batching wastes nothing, which is why the gap narrows at 32 requests.

## Repository layout

| path | what it is |
|---|---|
| `src/`, `data/` | dataset pipeline (v1 and v2) and its documentation |
| `sft/` | LoRA SFT experiments and failure analysis, plus the original RapidFire AI experiment |
| `rl/` | GRPO with age-conditioned rewards, HF vs vLLM rollout benchmark |
| `rag/` | corpus cleaning, LanceDB index, vector / full-text / hybrid search and reranking |
| `agent.py` | LangGraph agent (vLLM or Ollama backend) |
| `serving/` | vLLM serving benchmarks and the end-to-end agent test |

Notebooks ending in `_v1data.ipynb` / `_v2data.ipynb` are executed runs kept with their outputs; the notebooks
without a suffix are the current versions, set up for v2. All notebooks run on Colab (T4) and save results to Drive.

Not in the repo: LoRA adapters (about 100 MB each, to be published on the Hugging Face Hub) and the LanceDB index
(`rag/lancedb_data/`, 2.2 GB; rebuild it with the notebooks in `rag/`).
