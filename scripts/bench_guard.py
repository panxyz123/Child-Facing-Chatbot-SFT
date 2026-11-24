"""Latency of a guard model on this machine, as the agent would call it: one request at a time, prompt-only
classification, only the first output line needed (Qwen3Guard: "Safety: Safe/Unsafe/Controversial").

Serve the guard with llama.cpp (OpenAI-compatible), once on CPU and once on the Metal GPU:
    huggingface-cli download Qwen/Qwen3Guard-Gen-0.6B --local-dir models/qwen3guard-0.6b
    python /tmp/llama.cpp/convert_hf_to_gguf.py models/qwen3guard-0.6b --outtype f16 --outfile models/qwen3guard-0.6b-f16.gguf
    llama-server -m models/qwen3guard-0.6b-f16.gguf --jinja -ngl 0  --port 8090     # CPU only
    llama-server -m models/qwen3guard-0.6b-f16.gguf --jinja -ngl 99 --port 8090     # Metal GPU
    python scripts/bench_guard.py --tag qwen3guard06b_cpu --model Qwen3Guard-Gen-0.6B

--model: compare with that model's Colab predictions (eval/guard_models_eval.ipynb, copied to
results/guardrails/guard_models/<model>_prompts.jsonl) to check that the short, locally served output gives the
same verdicts. --colab takes any other predictions file.
"""
import argparse, json, os, random, re, statistics, time
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8090/v1")
ap.add_argument("--tag", required=True)
ap.add_argument("--suite", default="data/safety/suite.jsonl")
ap.add_argument("--n", type=int, default=100, help="prompts, stratified by label and source")
ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=8)
ap.add_argument("--model", default=None, help="e.g. Qwen3Guard-Gen-0.6B: compare with its Colab predictions")
ap.add_argument("--colab", default=None, help="predictions file; default from --model")
a = ap.parse_args()
if a.model and not a.colab:
    a.colab = f"results/guardrails/guard_models/{a.model}_prompts.jsonl"

rows = [json.loads(l) for l in open(a.suite, encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[(r["label"], r["source"])].append(r)
rng = random.Random(0); pick = []
for k in sorted(by):
    rng.shuffle(by[k])
while len(pick) < a.n and any(by.values()):          # round-robin over strata
    for k in sorted(by):
        if by[k] and len(pick) < a.n:
            pick.append(by[k].pop())

from openai import OpenAI
cli = OpenAI(base_url=a.base_url, api_key="none", timeout=120)
msg = lambda t: [{"role": "user", "content": t[:6000]}]
cli.chat.completions.create(model="guard", messages=msg("hello"), max_tokens=a.max_tokens, temperature=0)   # warm up
lat, out = [], {}
for r in pick:
    t0 = time.perf_counter()
    o = cli.chat.completions.create(model="guard", messages=msg(r["text"]), max_tokens=a.max_tokens, temperature=0)
    lat.append(time.perf_counter() - t0)
    text = o.choices[0].message.content or ""
    m = re.search(r"Safety: (Safe|Unsafe|Controversial)", text)
    out[r["id"]] = m.group(1) if m else None
q = lambda p: sorted(lat)[min(len(lat) - 1, int(p / 100 * len(lat)))]
res = dict(tag=a.tag, n=len(pick), max_tokens=a.max_tokens, p50_ms=1000 * q(50), p95_ms=1000 * q(95),
           mean_ms=1000 * statistics.mean(lat), labels=Counter(out.values()),
           unparsed=sum(v is None for v in out.values()))
if a.colab:
    ref = {}
    for l in open(a.colab):
        x = json.loads(l); m = re.search(r"Safety: (Safe|Unsafe|Controversial)", x.get("raw", ""))
        ref[x["id"]] = m.group(1) if m else ("Unsafe" if x["unsafe"] else "Safe")
    both = [i for i in out if i in ref and out[i]]
    res["agree_with_colab"] = sum(out[i] == ref[i] for i in both) / max(len(both), 1)
    res["compared"] = len(both)
print(json.dumps(res, indent=1, default=dict))
os.makedirs("results/guardrails", exist_ok=True)
json.dump(res, open(f"results/guardrails/latency_{a.tag}.json", "w"), indent=1, default=dict)
