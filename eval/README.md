# Hallucination evaluation

How often do answers to children's knowledge questions contain false facts, and what reduces it?

## Highlights

- **False claims cut from 33.0% to 14.9%** on the same 30 test questions and the same judge, by moving the
  generator from TinyLlama-1.1B SFT (temperature 0.8) to Qwen2.5-1.5B-Instruct with the same LoRA SFT recipe
  (temperature 0.3). Answers with at least one false claim: 56.7% to 30.0%.
- At the same temperature (0.3), Qwen fixes 14 answers that TinyLlama got wrong and breaks 3 (sign test p about
  0.01).
- **The age-aware style carries over**: Qwen SFT on the 393-row hold-out reaches BERTScore-F1 0.888 (TinyLlama SFT
  0.880) and raises answers within 2 grade levels of the reference from 43.8% (base Qwen) to 56.5%, with the
  largest gain for ages 14-18 (20.0% to 56.2%).

## Method

- **Judge** (`src/judge_hallucination.py`, Gemini 3.5 flash-lite, FActScore-style): split each answer into atomic
  claims, grade each as correct / acceptable simplification for the age / false / unverifiable. Reported: share of
  false claims, and share of answers with any false claim. A manual spot check of the "false" verdicts agreed with
  the judge on almost all of them.
- **Paired design**: every condition answers the same questions; comparisons count answers fixed and broken.
- **Conditions**: through the agent (`eval/rag_e2e.py`, `scripts/run_agent_subset.py`) with and without retrieval;
  Qwen answers generated on Colab (`eval/qwen_norag_check.ipynb`) from the agent's own prompts
  (`scripts/export_e2e_prompts.py`), so the prompts match. `scripts/compare_norag.py` prints all pairs.

## Results (30 test questions)

| generator | temperature | retrieval | answers with an error | false claims |
|---|---|---|---|---|
| TinyLlama SFT | 0.8 | no | 56.7% | 33.0% |
| TinyLlama SFT | 0.3 | no | 66.7% | 31.8% |
| TinyLlama SFT | 0.3 | yes | 63.3% | 27.8% |
| Qwen2.5-1.5B SFT | 0.8 | no | 50.0% | 25.3% |
| **Qwen2.5-1.5B SFT** | **0.3** | **no** | **30.0%** | **14.9%** |
| Qwen2.5-1.5B SFT | 0.3 | yes | 33.3% | 14.0% |

What the table says:

- **The generator is the bottleneck.** A lower temperature helps Qwen (7 fixed, 1 broken) but not TinyLlama
  (4 fixed, 7 broken): TinyLlama's errors are missing knowledge, Qwen's are partly sampling noise.
- **Retrieval has no measurable effect here**, for either model (Qwen: 3 fixed, 4 broken). The same holds on a
  larger 100-question TinyLlama run (answers with an error 53.5% without vs 60.4% with retrieval, 95% CI
  [-4.0, +17.8]) and does not change when passages are gated by reranker score.

Qwen SFT training data adds the agent's retrieved passages to the prompts of knowledge questions
(`scripts/build_rag_sft.py`, targets unchanged), so the model is trained on the same prompt format it is served with.

## Error patterns (Qwen, temperature 0.8)

Invented causes for "why" questions (the most common), mixed-up historical names and dates, simple slips that a
lower temperature removes ("blue and yellow make yellow"), and a few strict calls on child-level simplifications.

## Limitations

- 30 questions: the model and temperature effects are clear, small differences (such as retrieval) are not.
- Qwen answers were sampled with HF `generate` on Colab and TinyLlama answers through Ollama, so the cross-model
  comparison mixes two sampling implementations; within-model comparisons do not.
- The test questions are mostly common knowledge, where a model can answer without retrieval. Showing a retrieval
  benefit needs questions the model cannot answer but the corpus can.
- The deployed agent and the serving benchmarks still use TinyLlama; Qwen is the candidate replacement.
