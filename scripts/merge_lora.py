"""Merge the Qwen SFT LoRA adapter into its base model and save a standalone fp16 model (for llama.cpp / Ollama /
vLLM without --enable-lora). The chat template and tokenizer are saved with it.

    python scripts/merge_lora.py --adapter ~/Downloads/E_alpha64/adapter --out models/sprout-qwen
    python scripts/merge_lora.py --adapter <SFT adapter> --adapter2 <GRPO adapter> --out models/sprout-qwen-grpo
    python /tmp/llama.cpp/convert_hf_to_gguf.py models/sprout-qwen --outtype f16 --outfile models/sprout-qwen-f16.gguf
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", required=True, help="folder with adapter_config.json and adapter_model.safetensors")
ap.add_argument("--adapter2", default=None, help="optional second adapter trained on the merged first one (GRPO)")
ap.add_argument("--out", default="models/sprout-qwen")
a = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)     # merge in fp32, save fp16
for p in model.parameters():                 # macOS CPU misaligned-weights workaround (see rag/README.md)
    p.data = p.data.clone()
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
if a.adapter2:                                # GRPO adapter, trained on top of the merged SFT model
    model = PeftModel.from_pretrained(model, a.adapter2).merge_and_unload()
model = model.to(torch.float16)
tok = AutoTokenizer.from_pretrained(a.adapter if __import__("os").path.exists(f"{a.adapter}/tokenizer_config.json") else a.base)

msgs = [{"role": "system", "content": "You are talking to a child aged 7-9. Generate a friendly response with "
                                      "age-appropriate knowledge."}, {"role": "user", "content": "Why is the sky blue?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
ids = enc["input_ids"] if hasattr(enc, "keys") else enc   # newer transformers return a dict (BatchEncoding)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, do_sample=False)
text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
print("smoke test:", text)
assert text.strip() and "�" not in text, "empty or garbled output: do not use this merge"
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("saved", a.out)
