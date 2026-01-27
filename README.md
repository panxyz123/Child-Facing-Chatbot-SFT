# Age-Aware Child-AI Interaction Assistant

This repository contains the implementation of an **Age-Aware Educational Chatbot**, fine-tuned on **TinyLlama-1.1B**. The project focuses on bridging the gap in AI accessibility for children by dynamically adjusting cognitive depth, explanation styles, and linguistic complexity based on the user's age.

## 🚀 Key Highlights

- **Data Evolution**: Synthesized a custom **Age-Intent-Response** dataset (1,200 samples) using Gemini 2.0-flash-lite.
    
- **Efficient Fine-Tuning**: Optimized for **single NVIDIA T4 GPU** using LoRA with an "all-linear-module" target strategy.
    
- **Metric Insight**: Explored the "ROUGE-L Paradox" in open-ended educational tasks, favoring **BERTScore** and **FKGL** for semantic and pedagogical alignment.
    
- **Infrastructure**: Accelerated experiments using the **RapidfireAI** framework.
    

---

## 📂 Dataset: Evolution & Preprocessing

The dataset was derived from foundational Hugging Face children's speech datasets and "evolved" into a sophisticated instruction-tuning set.

- **Format**: `[Age] - [Intent] - [Instruction] - Response]`
    
- **Cleaning**: Rigorous filtering of invalid entries from raw HF datasets followed by a thorough shuffle.
    
- **Structure**: 1,200 samples split into 500 (Train) / 100 (Eval) / 600 (Hold-out), constrained by hardware VRAM. 
    
- **Prompting**: Utilized a **ChatML-style** template with dynamic system prompts to inject age-constraints.
    

---

## 🛠️ Training Strategy & Optimization

### 1. LoRA Configuration

To maximize representational capacity, I moved beyond standard $q, v$ tuning:

- **Target Modules**: All linear modules ($q, k, v, o$, and MLP layers).
    
- **Rank ($r$)**: Increased rank to capture nuanced linguistic shifts.
    

### 2. Hyperparameter Balancing

- **Learning Rate**: Carefully tuned to avoid training instability and "generation explosion" during evaluation.
    

---

## 📊 Evaluation 

- **Baseline Model**: **TinyLlama-1.1B + Default LoRA**

| Config   | Key change(s)                        | BERT-F1 | ROUGE-L | Eval-mean-token-accuracy | Runtime | Notes                                              |
| -------- | ------------------------------------ | ------: | ------: | -----------------------: | ------- | -------------------------------------------------- |
| Baseline | \                                    |  0.8074 |  0.1003 |                   0.7271 | 3.27    | shortest runtime                                   |
| A        | LR: 1e-4                             |  0.7733 |  0.0931 |                   0.7817 | 3.40    | comparable performance and long runtime            |
| B (Best) | Modules: all linear; LoRA rank: 32   |  0.8086 |  0.1172 |                   0.7893 | 3.28    | better performance with slightly-increased runtime |
| C        | 1e-4 LR + all linear (LoRA rank: 32) |  0.8134 |  0.1062 |                   0.7946 | 3.46    | better performance, but largely-increased runtime  |

- **Result**: Holistic recalibration of the model's internal knowledge without significant runtime overhead.

---

## ⚠️ Known Limitations

1. **Linguistic Consistency**: Occasional "stylistic leakage" in long-form responses where the tone shifts mid-text.
    
2. **Sophistication Bias**: A tendency to hallucinate technical jargon when attempting to match high-school level complexity without sufficient RAG grounding.
    
3. **Readability Gap**: In complex scientific domains, further evaluation of mean FKGL bias still exceeds target levels.

4. **Low ROUGE-L Paradox**: The overall ROUGE-L scores remain low across checkpoints, primarily due to the misalignment between ROUGE-L and open-ended instruction types such as storytelling or exploratory explanations. Since ROUGE-L favors lexical overlap, it fails to reliably reflect performance on creative or open-ended responses.
    

---

## 🛠️ Tooling

Built and managed with **RapidfireAI**, which provided:

- **Automated Config Sweeps**: Efficiently searched LoRA ranks and LR combinations.
    
- **Interactive Dashboards**: Real-time monitoring of loss curves and custom metrics.
    
- **Format Guardrails**: Automatic validation of instruction inputs to ensure stable runs.
    

---

## 🔮 Future Work

- **DPO Alignment**: Use Direct Preference Optimization to penalize stylistic leakage.
    
- **RAG Grounding**: Integrate intent-aware external knowledge bases to reduce hallucination in advanced scientific queries.
    
- **Deterministic Evaluation**: Re-evaluating ROUGE-L strictly on closed-ended tasks.