"""Merge the LoRA adapters into TinyLlama, save the Sprout model, optionally push it to the Hugging Face Hub, and write
an Ollama Modelfile so the agent can run on a laptop without a GPU.

The GRPO adapter was trained on top of the merged SFT model, so the order is: base + SFT adapter -> merge ->
+ GRPO adapter -> merge.

    # adapters downloaded from Drive:
    #   llm_sft_runs/sft_lora_v2data/E_alpha64/adapter
    #   llm_rl_runs/grpo_v2data_dense/runs/grpo_main/adapter
    python scripts/export_model.py --sft-adapter adapters/sft_E --grpo-adapter adapters/grpo_dense --out models/sprout
    python scripts/export_model.py ... --push yxpan/sprout-tinyllama-1.1b       # also upload (huggingface-cli login first)

    # run locally with Ollama, then point the agent at it:
    cd models/sprout && ollama create sprout -f Modelfile
    VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=sprout python agent.py --age 7-9 --trace "why is the sky blue?"
"""
import argparse
import os

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TinyLlama-Chat's chat template, written for Ollama (same format SFT and GRPO were trained with)
MODELFILE = '''FROM .
TEMPLATE """{{- range .Messages }}<|{{ .Role }}|>
{{ .Content }}</s>
{{ end }}<|assistant|>
"""
PARAMETER stop "</s>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 30
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 160
'''

CARD = """---
base_model: {base}
license: apache-2.0
tags: [children, age-aware, lora, grpo, tinyllama]
---
# Sprout: an age-aware assistant for children (TinyLlama-1.1B-Chat, fine-tuned)

{base} with two LoRA stages merged in: SFT on an age-conditioned dataset
([yxpan/children_sft_dataset_v2](https://huggingface.co/datasets/yxpan/children_sft_dataset_v2)), then GRPO with a
readability reward. The system prompt must state the child's age, exactly as in training:

    You are talking to a child aged {{age}}. Generate a friendly response with age-appropriate knowledge.

with age one of 0-3, 4-6, 7-9, 10-13, 14-18. Code and evaluation: the Sprout repository.
This model alone has no safety filtering; use it inside the agent, which adds input and output guardrails.
"""


ADAPTER_CARD = {
    "sft-lora": """---
base_model: {base}
library_name: peft
tags: [lora, sft, children, age-aware]
---
# Sprout SFT LoRA adapter

LoRA adapter (SFT stage) for {base}, trained on an age-conditioned dataset. Load it on top of {base}.
The full model with both stages merged is [{merged}](https://huggingface.co/{merged}).
""",
    "grpo-lora": """---
base_model: {base}
library_name: peft
tags: [lora, grpo, children, age-aware]
---
# Sprout GRPO LoRA adapter

LoRA adapter from the GRPO stage. It was trained on top of the SFT model (the SFT adapter merged into {base}), so apply
it after merging the SFT adapter, not to {base} directly. Its adapter_config points to the local path used in training.
The full model with both stages merged is [{merged}](https://huggingface.co/{merged}).
""",
}


def merge(base_id, adapters, dtype_out):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)   # merge in fp32 on CPU
    for path in adapters:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        print("merged", path)
    return model.to(dtype_out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--grpo-adapter", default=None, help="omit to export the SFT model only")
    ap.add_argument("--out", default="models/sprout")
    ap.add_argument("--push", default=None, help="Hub repo id for the merged model, e.g. yxpan/sprout-tinyllama-1.1b")
    ap.add_argument("--push-adapters", action="store_true", help="also upload the adapters to <push>-sft-lora / -grpo-lora")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer
    adapters = [a.sft_adapter] + ([a.grpo_adapter] if a.grpo_adapter else [])
    model = merge(a.base, adapters, torch.float16)
    tok = AutoTokenizer.from_pretrained(a.base)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    open(os.path.join(a.out, "Modelfile"), "w").write(MODELFILE)
    card_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "model_card.md")
    card = open(card_file).read() if os.path.exists(card_file) else CARD.format(base=a.base)
    open(os.path.join(a.out, "README.md"), "w").write(card)
    print("saved", a.out)

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=a.out, repo_id=a.push, ignore_patterns=["Modelfile"])
        print("pushed", a.push)
        if a.push_adapters:
            import shutil
            import tempfile
            for name, path in [("sft-lora", a.sft_adapter), ("grpo-lora", a.grpo_adapter)]:
                if not path:
                    continue
                repo = f"{a.push}-{name}"
                # PEFT's auto-generated README names a local training path as base_model (e.g. /content/rl_work/sft_merged
                # for GRPO), which the Hub rejects: upload a copy with a card that names the Hub base model instead
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.copytree(path, tmp, dirs_exist_ok=True)
                    open(os.path.join(tmp, "README.md"), "w").write(ADAPTER_CARD[name].format(base=a.base, merged=a.push))
                    api.create_repo(repo, private=a.private, exist_ok=True)
                    api.upload_folder(folder_path=tmp, repo_id=repo)
                print("pushed", repo)


if __name__ == "__main__":
    main()
