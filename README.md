# Sprout: An Age-Aware RAG Agent for Children

Sprout grows with the child: the same question gets a different answer at 4 and at 14.

**Winner of the RapidFire AI SFT Competition** (the original SFT stage, `sft/`)

## Motivation

Children already talk to AI assistants, but general-purpose assistants handle them with one coarse switch: an
under-13 or under-18 experience that mainly filters sensitive content, not an answer adapted to the child's age.
Kids apps that do adapt mostly rely on prompting a large hosted model, without measuring whether the answer actually
fits the age. Yet a 4-year-old and a 14-year-old need very different answers: different vocabulary, sentence length,
depth of explanation and tone. For children, a confident wrong fact or an unsafe reply
also costs more than it does for adults, and parents need to trust the answers without checking each one.

Sprout is a prototype of an assistant built for that setting. A parent sets the child's age once, and every answer is
written for that age: grounded in retrieved documents when the question needs facts, checked by guardrails, and
served by a 1.1B model that fits on a single T4 GPU.

## Highlights

- **Age-aligned answers.** LoRA SFT, then GRPO with a readability reward. SFT raised answers at the right reading
  level from 39% to 60%; GRPO raised the share of questions whose answers get harder as the set age rises from 75%
  to 97.5%.
- **Low hallucination on knowledge questions.** Questions that need facts are routed to hybrid retrieval over a
  cleaned Wikipedia science corpus, and the answer is grounded in what was retrieved; hallucination is evaluated on
  the knowledge questions of the test split, with and without retrieval.
- **Safety.** Input guardrails (unsafe topics, personal information, prompt injection) and output guardrails (unsafe
  content, reading level against the set age).
- **Small model, low deployment cost.** TinyLlama-1.1B-Chat with LoRA: about 2.2 GB of fp16 weights and about 22 KB of
  KV cache per token (GQA), so the whole system runs on one 16 GB T4.
- **Low latency through inference optimization.** vLLM continuous batching and PagedAttention: 10.6 ms per decoded
  token at batch 1 (vs 26.6 ms with HF `generate`), up to 3.0x throughput under concurrency, and prefix caching that
  cut time-to-first-token from 305 ms to 48 ms on shared prompts. vLLM rollouts also made GRPO training steps 2.2x
  faster.

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

- rewards: readability, relevance to the reference (embedding similarity), and a format reward against overlong
  or repetitive answers
- the readability reward targets what the evaluation measures, the distance d between the answer's FKGL and that
  row's reference, as a dense reward 1 - d/5. Two earlier designs fell short: an age-group band (v1) did not match
  the evaluation metric and gave the too-simple teen answers full credit, since they sat inside their band; a flat
  reward within 2 grade levels (first v2 run) gave about 80% of samples the same score, so GRPO groups had little
  readability signal to learn from
- KL penalty to the SFT model, LoRA on the merged SFT weights
- evaluation averages several samples per test prompt: a single sample at temperature 0.8 flips many rows between
  runs and produced a gain that disappeared under the averaged evaluation
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

GRPO starts from config E (LoRA alpha 64, 350 steps) instead: its test results match this config by age group,
while this config's validation loss (1.61 vs about 1.53) shows overfitting from training twice as long.

Knowledge and non-knowledge questions improve about equally (58.9% and 60.7% after SFT). 14-18 barely improves
even with clean age labels, so the teen gap is not only a labeling problem: this is the gap GRPO targets.

### GRPO on v2

Starting from SFT config E: dense readability reward 1 - d/5 (d = FKGL distance to the row's reference),
relevance and format rewards, lr 5e-5, beta 0.01, 250 steps. During training the readability reward rose from 0.57 to
0.63 (mean FKGL distance on training samples from about 2.1 to 1.85 grade levels), with KL to SFT around 0.012;
BERTScore and answer length did not change.

Test split, 4 samples per prompt, 160 new tokens (the training limit), vLLM; paired bootstrap over test rows:

| | SFT | SFT + GRPO |
|---|---|---|
| FKGL within 2 grade levels | 58.1% | 60.1% (+2.0, 95% CI [-1.0, +4.9]) |
| mean FKGL gap to the reference | 1.97 | 1.92 |
| age sensitivity: same input at 5 set ages, mean Spearman (set age vs FKGL) | 0.687 | 0.787 |
| share of inputs with Spearman > 0.5 | 75.0% | 97.5% |

FKGL within 2 grade levels by age group: 0-3 60.1% to 64.3%, 4-6 57.1% to 64.0%, 7-9 60.6% to 57.5%,
10-13 58.2% to 58.2%, 14-18 54.4% to 56.6%.

**Limitations.**

- The test gain is small and its confidence interval includes 0. The clearer effect is age sensitivity: answers
  to the same question separate more by set age, but that sweep uses only 40 inputs (Wilcoxon p = 0.08).
- Evaluation noise is large. With one sample per prompt, GRPO looked like +5.3 points overall and +15 for ages
  10-13 (p = 0.02); averaging 4 samples per prompt removed both. Results here use the 4-sample evaluation.
- The target is noisy. FKGL is unstable on short answers, and each prompt has a single reference, while correct
  answers can differ in reading level. The reward optimizes distance to that one reference.
- One run per configuration, one seed, and only 2 GRPO runs on v2 (flat vs dense reward, with different lr and
  beta), so the reward change and the hyperparameter change are not separated.
- Quality is checked only by automatic proxies (BERTScore, length, distinct-3, spot checks of outputs); there is no
  human rating of the answers.
- Teen answers remain too simple (median FKGL 0.8 below the reference for 14-18), so GRPO did not fix the gap the
  SFT failure analysis found.

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
reference); on v2 the readability reward measures that distance directly.

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

## Setup

```
conda env create -f environment.yml && conda activate sprout
python -m pytest tests -q                 # offline tests of the agent graph, no GPU needed
python scripts/train_router.py            # knowledge router -> models/router.joblib
```

Run the agent on a laptop with [Ollama](https://ollama.com) serving the model
([yxpan/sprout-tinyllama-1.1b](https://huggingface.co/yxpan/sprout-tinyllama-1.1b), exported with
`scripts/export_model.py`), or on a GPU with vLLM:

```
(cd models/sprout && ollama create sprout -f Modelfile)
VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=sprout python agent.py --age 7-9 --trace "why is the sky blue?"
```

The retrieval index (`rag/lancedb_data/`) is not in the repository; rebuild it with the notebooks in `rag/`.

## Repository layout

| path | what it is |
|---|---|
| `src/`, `data/` | dataset pipeline (v1 and v2) and its documentation |
| `sft/` | LoRA SFT experiments and failure analysis, plus the original RapidFire AI experiment |
| `rl/` | GRPO with age-conditioned rewards, HF vs vLLM rollout benchmark |
| `rag/` | corpus cleaning, LanceDB index, vector / full-text / hybrid search and reranking |
| `agent.py` | LangGraph agent (vLLM or Ollama backend) |
| `serving/` | vLLM serving benchmarks and the end-to-end agent test |

Notebooks ending in `_v1data.ipynb` / `_v2data.ipynb` / `_v2data_dense.ipynb` are executed runs kept with their outputs
(`rl/`: the first v2 GRPO run with the flat reward, and the final run with the dense reward); the notebooks
without a suffix are the current versions, set up for v2. All notebooks run on Colab (T4) and save results to Drive.

Not in the repo: LoRA adapters (about 100 MB each, to be published on the Hugging Face Hub) and the LanceDB index
(`rag/lancedb_data/`, 2.2 GB; rebuild it with the notebooks in `rag/`).
