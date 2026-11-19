# Age-Aware Chatbot for Children

A small language model (TinyLlama-1.1B-Chat) that answers children's messages at the reading level of the age a
parent sets, built end to end: data pipeline, LoRA SFT, GRPO, a RAG agent, and vLLM serving on a single T4 GPU.

The original SFT stage (`sft/`) won the RapidFire AI SFT competition. This repo extends it into a full system.
Work in progress: the status of each part is listed below.

## Repository layout

| path | what it is |
|---|---|
| `src/`, `data/` | dataset pipeline (v1 and v2), the datasets, and how they were built: see [`data/README.md`](data/README.md) |
| `sft/` | LoRA SFT with TRL + PEFT (`sft_lora_experiments.ipynb`), plus the original RapidFire experiment |
| `rl/` | GRPO with an age-conditioned readability reward, and HF-vs-vLLM rollout benchmarks |
| `rag/` | Wikipedia science corpus cleaning, LanceDB index, vector / full-text / hybrid search and reranking |
| `agent.py` | LangGraph agent: retrieve, build context, generate (vLLM or Ollama backend) |
| `serving/` | vLLM serving benchmark and an end-to-end agent test |

Notebooks ending in `_v1data.ipynb` / `_v2data.ipynb` are executed runs on the v1 / v2 dataset, kept with their
outputs. The notebooks without a suffix are the current versions, set up for the v2 dataset.

## Data

Two Hugging Face datasets: [`yxpan/children_sft_dataset`](https://huggingface.co/datasets/yxpan/children_sft_dataset) (v1)
and [`yxpan/children_sft_dataset_v2`](https://huggingface.co/datasets/yxpan/children_sft_dataset_v2) (v2).

An audit of v1 found that every age label had been guessed by Gemini from the child's message, and those guesses
matched the source age only 26.3% of the time (chance: 20%). v2 treats the age as a condition that is given, not
inferred: 5 balanced age groups, a `needs_knowledge` label on every row (questions where a wrong fact is a
hallucination), and train / validation / test splits stratified by age and knowledge (700 / 100 / 393).
Details in [`data/README.md`](data/README.md).

## Results so far (single T4)

All numbers are from one run with one seed. v1 and v2 use different test sets, so their numbers are not
directly comparable.

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

Knowledge and non-knowledge questions improve about equally (58.9% and 60.7% within 2 grade levels after SFT).
**14-18 barely improves even with clean age labels**, so the teen gap seen on v1 is not only a labeling problem.
This is what the failure analysis and GRPO on v2 target next.

### v1 dataset

**SFT** (best config: LoRA r=32 on all linear layers, lr 1e-4, alpha 64), 600-row hold-out:

| | base TinyLlama | SFT |
|---|---|---|
| BERTScore-F1 | 0.848 | 0.883 |
| FKGL within 2 grade levels of the reference | 35.0% | 59.3% |
| mean FKGL gap to the reference | 3.46 | 2.16 |

The 14-18 group did not improve. That traced back to the v1 labels: most rows labeled 14-18 were toddler
utterances, which led to the v2 dataset.

**GRPO** on top of SFT (250 steps, rewards: readability band per age group, relevance to the reference, format):

| | SFT | SFT + GRPO |
|---|---|---|
| FKGL within 2 grade levels | 59.3% | 61.3% |
| mean FKGL gap | 2.16 | 1.94 |
| BERTScore-F1 | 0.883 | 0.883 |
| age sensitivity: same input at 5 set ages, mean Spearman (set age vs FKGL) | 0.435 | 0.455 |
| share of inputs with Spearman > 0.5 | 47.5% | 57.5% |

The hold-out gain (+2 points) is about one standard error, so it is not yet a reliable improvement. The model
moved little from SFT (KL to SFT about 0.01), and the reward (FKGL inside the age group's band) is not the same
quantity as the hold-out metric (distance to each row's own reference). Both are being revisited on v2.

**GRPO rollouts, HF `generate` vs vLLM** (same prompts and sampling, T4):

| | HF | vLLM | speedup |
|---|---|---|---|
| rollout throughput | 301 tokens/s | 2,440 tokens/s | 8.1x |
| GRPO step time | 8.46 s | 3.78 s | 2.2x |

End to end is smaller than the rollout speedup because training (forward, backward, reference log-probs) is not
accelerated: with HF, generation was roughly 60% of a step; with vLLM the bottleneck moves to training.

**Serving** (vLLM vs HF `generate`, base TinyLlama, outputs fixed at 128 tokens, T4):

| | HF | vLLM |
|---|---|---|
| decode time per token (batch 1) | 26.6 ms | 10.6 ms |
| throughput, 8 concurrent requests | 214 tokens/s | 637 tokens/s (3.0x) |
| throughput, 32 concurrent requests | 950 tokens/s | 1,541 tokens/s (1.6x) |

Prefix caching cut time-to-first-token from 305 ms to 48 ms (6.3x) when 32 requests share a ~1.1k-token prefix.
That is an upper bound: requests that retrieve different documents share much less. KV cache size is about
22 KB per token (GQA, 8x smaller than multi-head attention). With fixed output lengths, static batching wastes
nothing, which is why the gap narrows at 32 requests; realistic output lengths are added in the notebook.

## Roadmap

- [x] v1 dataset, SFT and GRPO, serving benchmark (results above)
- [x] v2 dataset: set ages, balanced groups, knowledge labels, stratified splits
- [x] SFT on v2 (results above)
- [ ] v1 vs v2: evaluate the v1-trained model on the v2 test set
- [ ] SFT failure analysis by age group and question type, to decide what GRPO should fix
- [ ] GRPO on v2, plus reward ablations (no relevance reward, KL strength)
- [ ] Hallucination evaluation on knowledge questions, with and without RAG
- [ ] Agent: age-aware prompts, route knowledge questions to retrieval, conversation memory, basic input and output guardrails
- [ ] Serving the fine-tuned model: realistic workloads (multi-turn chat, RAG prompts), SLO-based load tests

## Not in this repo

- LoRA adapters (about 100 MB each): to be published on the Hugging Face Hub.
- The LanceDB index (`rag/lancedb_data/`, 2.2 GB): rebuild it with the notebooks in `rag/`.
