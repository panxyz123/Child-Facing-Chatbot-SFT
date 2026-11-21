---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: transformers
license: apache-2.0
language: [en]
datasets: [yxpan/children_sft_dataset_v2]
pipeline_tag: text-generation
tags: [children, age-aware, readability, lora, sft, grpo, tinyllama]
---

# Sprout: an age-aware assistant model for children (TinyLlama-1.1B)

Sprout answers a child's message at the reading level of the age a parent sets. The same question gets a short,
simple answer for a 4-year-old and a fuller, more advanced one for a 14-year-old.

It is TinyLlama-1.1B-Chat with two fine-tuning stages merged into the weights: LoRA SFT on an age-conditioned
dataset, then GRPO with a readability reward. It is the generation model of the Sprout agent, which adds retrieval
for knowledge questions and input / output guardrails.

| | |
|---|---|
| Base model | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Size | 1.1B parameters, fp16 (about 2.2 GB) |
| Age groups | 0-3, 4-6, 7-9, 10-13, 14-18 |
| Language | English |
| Stage adapters | [SFT LoRA](https://huggingface.co/yxpan/sprout-tinyllama-1.1b-sft-lora), [GRPO LoRA](https://huggingface.co/yxpan/sprout-tinyllama-1.1b-grpo-lora) |

## Intended use

- A research and portfolio prototype of an age-aware assistant for children, used **inside an application that adds
  safety filtering and retrieval** (the Sprout agent).
- Studying age-conditioned generation, readability control, and small-model fine-tuning (SFT and RL).

**Out of scope.** Unsupervised use by children without guardrails; medical, legal, or safety advice; any use where
factual accuracy matters without retrieval and checking. This model has **no safety filtering of its own**.

## How to use

The system prompt must state the age exactly as in training, with the age one of the five groups above:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "yxpan/sprout-tinyllama-1.1b"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

age = "7-9"
messages = [
    {"role": "system", "content": f"You are talking to a child aged {age}. Generate a friendly response with age-appropriate knowledge."},
    {"role": "user", "content": "Why is the sky blue?"},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=160, do_sample=True, temperature=0.8, top_p=0.9, top_k=30,
                     repetition_penalty=1.05)
print(tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True))
```

Serving with vLLM: `vllm serve yxpan/sprout-tinyllama-1.1b --dtype half --max-model-len 2048`, then use the
OpenAI-compatible chat API with the same system prompt. The sampling settings above are the ones used in evaluation.

## Training data

[yxpan/children_sft_dataset_v2](https://huggingface.co/datasets/yxpan/children_sft_dataset_v2): 1,193 (child message,
age, response) rows, balanced across the five age groups, split 700 / 100 / 393 (train / validation / test), stratified
by age and by whether the question needs factual knowledge.

- The age is a condition set before generation, not guessed from the message (an earlier version that let an LLM
  guess the age was right only 26% of the time).
- Messages for ages 0-6 include real child speech; older ages use questions from an age-labeled child-assistant
  dataset. Responses were written for the set age by Gemini, and responses whose reading level (FKGL) was an outlier
  for their age group were removed.

## Training procedure

**Stage 1, SFT** (TRL `SFTTrainer`, PEFT LoRA, completion-only loss): rank 32, alpha 64, all linear layers
(q, k, v, o, gate, up, down), learning rate 1e-4, 350 steps, fp16 on a T4. Chosen over a config trained twice as long,
which scored similarly on the test split but showed overfitting on validation loss.

**Stage 2, GRPO** (TRL `GRPOTrainer`, on the merged SFT model): a new LoRA (rank 16, alpha 32, all linear layers),
learning rate 5e-5, KL coefficient 0.01, 4 sampled answers per prompt, 250 steps, rollouts generated with vLLM.
Rewards, weighted 1.0 / 0.5 / 0.3:

- readability: 1 - d/5, where d is the distance between the answer's Flesch-Kincaid grade level and the reference
  answer's (0 beyond 5 grade levels)
- relevance: embedding similarity to the reference answer (all-MiniLM-L6-v2), so answers stay on topic
- format: penalties for empty answers, answers longer than the age group's length cap, and repetition loops

## Evaluation

Test split of the dataset above (393 rows). "Within 2 grade levels" is the share of answers whose FKGL is within 2
of the reference answer's.

| | TinyLlama-Chat | + SFT | + SFT + GRPO (this model) |
|---|---|---|---|
| FKGL within 2 grade levels (1 sample per prompt) | 39.2% | 58.3% | 63.6% |
| FKGL within 2 grade levels (4 samples per prompt, vLLM) | | 58.1% | 60.1% |
| mean FKGL distance to the reference (1 sample) | 3.12 | 1.96 | 1.91 |
| BERTScore-F1 to the reference | 0.854 | 0.880 | 0.881 |
| same question at 5 set ages: share whose reading level rises with age (Spearman > 0.5, 40 questions) | 32.5% | 75.0% | 97.5% |

SFT gives the large gain in reading-level fit. GRPO's clearest effect is that answers separate more by set age; its
gain on the test split is small, and with 4 samples per prompt it is not statistically significant
(+2.0 points, 95% CI [-1.0, +4.9]).

## Limitations

- **Factual errors.** A 1.1B model often states wrong facts with confidence, especially when explaining science.
  Use it with retrieval for knowledge questions.
- **No safety filtering.** Unsafe or inappropriate requests are not refused reliably; deploy it behind guardrails.
- **Teen answers tend to be too simple** (for ages 14-18, median FKGL about 0.8 grade levels below the reference).
- **Reading level is measured with FKGL**, a noisy proxy on short texts; the references come from one model (Gemini),
  so "age-appropriate" here means close to that model's answers, not a validated developmental standard.
- English only; single-turn training data.

## License

The weights are released under Apache 2.0, the license of the base model. The training responses were generated
with Google Gemini.
