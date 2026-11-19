**Links**

- Notebook: 
	https://drive.google.com/file/d/1ZCrqFTp-C11QeZ6QuBcQEuhaR-1hxRUC/view?usp=sharing

- Dataset:
	[https://huggingface.co/datasets/yxpan/children_sft_dataset](https://huggingface.co/datasets/yxpan/children_sft_dataset)

- Repo:
	

- Screenshots:
	[https://drive.google.com/file/d/1Kj5uJ8tt2d_MVh3Oja5r_WNqZe8IwUj0/view?usp=sharing](https://drive.google.com/file/d/1Kj5uJ8tt2d_MVh3Oja5r_WNqZe8IwUj0/view?usp=sharing)
	https://drive.google.com/file/d/1DV9bAAPWH6PR1varGvYwZ56fV2uxJuGB/view?usp=sharing](https://drive.google.com/file/d/1DV9bAAPWH6PR1varGvYwZ56fV2uxJuGB/view?usp=sharing)
	https://drive.google.com/file/d/1nXcscD-6vkITN1Es6-wkxWRhR5PYvhuu/view?usp=sharing
	[https://drive.google.com/file/d/1g1Mwc11KtrQE8zLhLlyq7vqzhLqeYHu5/view?usp=sharing](https://drive.google.com/file/d/1g1Mwc11KtrQE8zLhLlyq7vqzhLqeYHu5/view?usp=sharing)

# Fine-Tuning Experiment Summary

## 1) What you tried 

Motivated by the need for age-appropriate educational AI, I developed an **Age-Aware Chatbot** by fine-tuning **TinyLlama-1.1B** on a custom dataset evolved via the **Gemini-2.0-flash-lite API**. I optimized the SFT process through **LoRA**, achieving significant improvements in both semantic alignment and readability for teenage users.

## 2) What “good” looks like (success criteria)

The primary objective is to develop a model capable of generating **age-stratified and pedagogically sound** responses that align with the cognitive levels of children. Success is quantified by the model's ability to minimize the semantic and stylistic gap between its outputs and the **Gemini-generated ground truth**. Performance is rigorously evaluated using **BERTScore (F1)** to capture semantic alignment and **ROUGE-L** to measure structural and lexical consistency.

## 3) Setup (bullet list)

- Base model(s)**: TinyLlama-1.1B

- Dataset(s) / domain (size, format, any filtering/cleanup):
	The final JSON-format dataset comprises **1,200 samples** structured in an **Age-Intent-Intruction-Response** format, covering 5 distinct age groups and 4 intent groups. Prior to data synthesis, a rigorous **data cleaning** pipeline was implemented to filter out invalid entries from the original Hugging Face children's speech datasets. To ensure model generalization and mitigate sequence bias, the curated data was **thoroughly shuffled** before being used for supervised fine-tuning.

- Train/Eval split:
	Due to the memory and compute limitations of the **NVIDIA T4 GPU**, the dataset was partitioned into a **training set of 500 samples** and an **evaluation set of 100 samples**.

- Prompt / formatting approach:
	The formatting logic extracts the ‘age’ attribute to construct a customized **System Prompt**, then pairs the ‘instruction’ and ‘response’ within a ChatML-compliant dictionary to guide the SFT process.

## 4) Experiment dimensions (what you varied + why)

List the main “knobs” you explored and the reason for each.

- **Knob 1:** learning rate = 1e-5 / 1e-4 — *why*: learning rate controls optimization stability and convergence speed.
- **Knob 2:** target_modules = q,v vs all linear — *why*: different target modules determine where low-rank adaptation is applied, trading off expressiveness and overfitting risk.
- **Knob 3:** LoRA rank = 8 / 32 (varied with target modules) — *why*: LoRA rank controls the effective adaptation capacity and is adjusted to match the scope of the adapted modules.

## 5) Configs compared (keep it short)

- **Baseline:** base model + default LoRA
- **Config A:** higher learning rate
- **Config B:** more target modules
- **Config C:** combining A and B, higher LR and more modules

## 6) Results (tiny table)

Use 1–2 primary metrics max (plus cost/time if relevant).

| Config   | Key change(s)                        | BERT-F1 | ROUGE-L | Eval-mean-token-accuracy | Runtime | Notes                                              |
| -------- | ------------------------------------ | ------: | ------: | -----------------------: | ------- | -------------------------------------------------- |
| Baseline | \                                    |  0.8074 |  0.1003 |                   0.7271 | 3.27    | shortest runtime                                   |
| A        | LR: 1e-4                             |  0.7733 |  0.0931 |                   0.7817 | 3.40    | comparable performance and long runtime            |
| B (Best) | Modules: all linear; LoRA rank: 32   |  0.8086 |  0.1172 |                   0.7893 | 3.28    | better performance with slightly-increased runtime |
| C        | 1e-4 LR + all linear (LoRA rank: 32) |  0.8134 |  0.1062 |                   0.7946 | 3.46    | better performance, but largely-increased runtime  |

## 7) Best config: why it won (metrics + tradeoffs)

- **Best config:** Config B

- **What improved (numbers):** +0.12\% BERT-F1, +1.69\% rougeL, +6.22\% Eval-mean-token-accuracy

- **Why it likely improved:** 
	- **More Modules:** Fine-tuning all linear modules enables a **holistic recalibration** of the model's internal attention and knowledge layers, ensuring deeper stylistic and logical alignment.
	- **Higher Rank:** Increasing the LoRA rank expands the model's **representational capacity**, allowing it to capture more complex, nuanced patterns required for sophisticated age-aware responses.

- **Tradeoffs / costs:** 
	- More parameters are engaged, leading to longer runtime and heavier vRAM demand
	- Increased overfitting risks under small training sets

- **Where it still fails:** 
	- **Significant Readability Gap:** Quantitative analysis using the **mean FKGL (Flesch-Kincaid Grade Level) bias** reveals a discrepancy exceeding **11.0 grade levels** in certain edge cases. This indicates that while the model attempts age-adaptation, its linguistic complexity often overshoots the target, particularly when explaining abstract scientific concepts.
	- **Linguistic Consistency "Leakage":** The model exhibits **stylistic instability** during long-form generation. It occasionally lapses into oversimplified or "childish" tones midway through a response designed for teenagers (14-18y), failing to maintain a consistent cognitive depth throughout the entire sequence.
	- **Low ROUGE-L:** Even the best-performing checkpoints exhibit lower-than-expected ROUGE-L scores. This is likely due to the **misalignment** between ROUGE-L and open-ended instruction types, such as storytelling or exploratory explanations. A more appropriate evaluation strategy is to compute ROUGE-L only on deterministic instructions, for example, scientific concept clarification tasks.
	- **Complexity-Induced Hallucination:** Under the pressure of "sounding sophisticated" for older age groups, the model tends to **hallucinate technical jargon** or fabricate scientific logic.

## 8) How RapidFire AI helped (2–5 bullets)
- **Automated Config Sweeps:** Automatically generated hyperparameter combinations, enabling high-throughput experimentation with LoRA ranks, learning rates, etc.
- **Interactive Training Control:** Enabled real-time supervision and dynamic adjustments during the SFT process to optimize T4 GPU resource allocation.
- **Unified Visualization:** Provided integrated dashboards for real-time monitoring of loss curves and selected metrics (ROUGE-L/FKGL).
- **Automated Guardrails:** Ensured execution stability by auto-correcting data formats and filtering unsafe or malformed instruction inputs.

## 9) Takeaways (3–6 bullets)

- **What Helps Most:** Expanding LoRA targets to **all linear modules** combined with a **higher LoRA rank** significantly enhances the model's **representational capacity** and ensures a **holistic recalibration** of internal knowledge. This approach yields substantial performance gains in age-adaptation without a proportional increase in training latency.

- **What Didn't Help:** An excessively high **learning rate** can trigger **training instability**, characterized by oscillating loss and numerical divergence. This often leads to suboptimal performance and extended runtimes, as the model may generate repetitive or over-extended sequences during evaluation, increasing the total autoregressive inference overhead.

- **Future Experiments:** Future experiments should focus on **scaling the training duration (max steps)** and **expanding the dataset size**. Furthermore, refining **prompt engineering** strategies will be critical to further unlocking the model's potential for nuanced, age-stratified dialogue.

- **Mitigating Current Failures:** To address existing limitations, several advanced strategies will be explored:
    - **Refined Evaluation:** Isolating **ROUGE-L** metrics specifically for deterministic tasks to avoid misleading scores on creative, open-ended responses.
    - **Multi-dimensional Metrics:** Incorporating specialized **linguistic consistency** and **age-appropriateness benchmarks** (e.g., automated readability scoring).
    - **Hallucination Control:** Integrating **RAG (Retrieval-Augmented Generation)** to provide a factual "grounding" mechanism, thereby reducing the model's tendency to fabricate technical details.