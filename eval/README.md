# Hallucination evaluation

How often do answers to children's knowledge questions contain false facts, and what reduces it?

## Highlights

- **False claims cut from 33.0% to 14.9%** on the same 30 test questions and the same judge, by moving the
  generator from TinyLlama-1.1B SFT (temperature 0.8) to Qwen2.5-1.5B-Instruct with the same LoRA SFT recipe
  (temperature 0.3). Answers with at least one false claim: 56.7% to 30.0%.
- At the same temperature (0.3), Qwen fixes 14 answers that TinyLlama got wrong and breaks 3 (sign test p about
  0.01).
- **The age-aware style carries over**: at the deployed sampling (temperature 0.3, 160 tokens), Qwen SFT on the
  393-row hold-out raises answers within 2 grade levels of the reference from 43.8% (base Qwen) to 62.6%, with the
  largest gain for ages 14-18 (8.8% to 53.8%); BERTScore-F1 0.870 to 0.890.
- **Paragraph retrieval helps where it is used**: on the questions whose top passage passes the relevance gate,
  false claims are lowest with the paragraph index (11.5%, vs 12.3% without retrieval and 14.3% with the old
  index); small differences on 20 questions.

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
| Qwen2.5-1.5B SFT | 0.3 | yes, article leads (`wiki_v3`, no gate) | 33.3% | 14.0% |
| Qwen2.5-1.5B SFT | 0.3 | yes, paragraphs + gate (`wiki_v4_para`, deployed) | 36.7% | 18.1% |
| Qwen2.5-1.5B SFT + GRPO | 0.3 | yes, paragraphs + gate | 50.0% | 24.2% |

What the table says:

- **The generator is the bottleneck.** A lower temperature helps Qwen (7 fixed, 1 broken) but not TinyLlama
  (4 fixed, 7 broken): TinyLlama's errors are missing knowledge, Qwen's are partly sampling noise.
- **Retrieval has no measurable effect here**, for either model (Qwen: 3 fixed, 4 broken). The same holds on a
  larger 100-question TinyLlama run (answers with an error 53.5% without vs 60.4% with retrieval, 95% CI
  [-4.0, +17.8]) and does not change when passages are gated by reranker score.
- **By gate**, for the deployed paragraph index (Qwen SFT, 0.3): the 20 questions whose top passage passes the gate
  have the fewest false claims with it (7 of 61 claims, 11.5%; without retrieval 9 of 73, 12.3%; old index 9 of
  63, 14.3%). The 10 below the gate get no passages, only the "say you are not sure" line, and differ from the
  no-retrieval run by sampling alone (10 of 33 vs 6 of 28). Better retrieval quality (`rag/README.md`) therefore
  shows up as slightly fewer false claims where passages are used, not as a measurable overall change.

## GRPO on Qwen: more age-sensitive, less factual

GRPO with the readability reward (`rl/grpo_qwen_rag.ipynb`, same rewards as on TinyLlama), evaluated at 0.3:

| hold-out (393), 4 samples per row | SFT | SFT + GRPO |
|---|---|---|
| within 2 grade levels | 63.1% | 63.7% (+0.6, 95% CI [-2.5, +3.8]) |
| mean FKGL gap | 1.86 | 1.77 (-0.09, CI [-0.17, -0.00], p = 0.04) |
| age sensitivity, mean Spearman over 40 inputs x 5 ages | 0.767 | 0.797 (+0.031, CI [-0.057, +0.118], Wilcoxon p = 0.48) |
| inputs whose reading level rises with the set age (Spearman > 0.5) | 85% | 90% (5 gained, 3 lost, p = 0.73) |
| false claims (30 questions above) | 18.1% | 24.2% (5 fixed, 9 broken) |

The age trend repeats what GRPO did on TinyLlama, where it is clearer (mean Spearman 0.687 to 0.787, bootstrap CI
[+0.005, +0.202], Wilcoxon p = 0.08; Spearman > 0.5 from 75% to 98% of inputs, 10 gained and 1 lost, exact p = 0.01;
`scripts/age_sweep_stats.py`). On Qwen the SFT model is already age-sensitive (85%), leaving less room, and the new errors are mostly misleading child
analogies ("when you stop and spin the other way, it's nighttime"; "the beat is like the referee's whistle"):
the reward pays for simpler wording and has no term for being right. Neither change is significant on its own;
the deployed model is the SFT model, since a wrong fact costs more for a child than a slightly better reading
level. Next: a factuality term in the reward (NLI against the reference or the passages) or a larger KL penalty.

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
- The judge is an LLM; it is strict on some child-level analogies, which affects the SFT and GRPO rows alike.
