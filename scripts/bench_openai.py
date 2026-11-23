"""Load test for any OpenAI-compatible chat server (Ollama, llama.cpp llama-server, vLLM), with the agent's real
prompts. Streams every request and records TTFT, TPOT (time per output token after the first), E2E latency and
output tokens; reports percentiles and throughput per load level.

Two load modes:
  closed  N concurrent users, each sends its next request when the previous one ends   (--concurrency 1,4,8)
  open    Poisson arrivals at a fixed rate, independent of how fast the server answers (--rates 0.5,1,2 req/s)
Open loop is what real traffic looks like (latency grows without bound past capacity); closed loop is the quick
capacity check for single-machine servers.

    python scripts/bench_openai.py --base-url http://localhost:11434/v1 --model sprout --tag ollama
    python scripts/bench_openai.py --base-url http://localhost:8080/v1  --model sprout --tag llamacpp
    -> results/serving/<tag>.json (summary per level) and <tag>_requests.jsonl (one row per request)

SLO for goodput (share of requests meeting all three): --slo-ttft 1.0 --slo-tpot 0.1 --slo-e2e 10 (seconds).
"""
import argparse, asyncio, json, os, random, statistics, time


def load_prompts(paths, n, seed):
    """Agent prompts exported by scripts/export_e2e_prompts.py (messages incl. retrieved facts); rows without
    messages get the agent's plain system prompt. Shuffled with a fixed seed and cycled."""
    rows = []
    for p in paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rows.append(r.get("messages") or [
                {"role": "system", "content": f"You are talking to a child aged {r['age']}. "
                                              "Generate a friendly response with age-appropriate knowledge."},
                {"role": "user", "content": r["instruction"]}])
    random.Random(seed).shuffle(rows)
    return [rows[i % len(rows)] for i in range(n)]


async def one(client, a, msgs, rid, t_sched):
    t0 = time.perf_counter()
    first, n_chunks, usage, err, text = None, 0, None, None, []
    try:
        stream = await client.chat.completions.create(
            model=a.model, messages=msgs, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
            stream=True, stream_options={"include_usage": True}, extra_body=dict({"top_k": a.top_k}, **({"ignore_eos": True} if a.ignore_eos else {})),
            stop=a.stop or None)
        async for ch in stream:
            if getattr(ch, "usage", None):
                usage = ch.usage
            if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter()
                n_chunks += 1
                text.append(ch.choices[0].delta.content)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:120]}"
    t1 = time.perf_counter()
    out_tok = usage.completion_tokens if usage and usage.completion_tokens else n_chunks
    ttft = (first - t0) if first else None
    tpot = (t1 - first) / (out_tok - 1) if first and out_tok > 1 else None
    return dict(id=rid, queue_s=round(t0 - t_sched, 4) if t_sched else 0.0, ttft=ttft, tpot=tpot, e2e=t1 - t0,
                out_tokens=out_tok, prompt_tokens=getattr(usage, "prompt_tokens", None), error=err,
                chars=len("".join(text)))


async def closed_loop(client, a, prompts, conc):
    q = list(enumerate(prompts)); out = []
    async def user():
        while q:
            i, m = q.pop(0)
            out.append(await one(client, a, m, i, None))
    t0 = time.perf_counter()
    await asyncio.gather(*[user() for _ in range(conc)])
    return out, time.perf_counter() - t0


async def open_loop(client, a, prompts, rate):
    rng = random.Random(a.seed); tasks = []
    t0 = time.perf_counter(); t = t0
    for i, m in enumerate(prompts):
        t += rng.expovariate(rate)
        await asyncio.sleep(max(0.0, t - time.perf_counter()))
        tasks.append(asyncio.create_task(one(client, a, m, i, t)))
    out = await asyncio.gather(*tasks)
    return list(out), time.perf_counter() - t0


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 4) if xs else None


def summarize(rows, wall, a, level):
    ok = [r for r in rows if not r["error"] and r["ttft"] is not None]
    # open loop: time waiting in our own client counts too (it is time the user waits)
    e2e = [r["e2e"] + r["queue_s"] for r in ok]
    good = [r for r in ok if r["ttft"] + r["queue_s"] <= a.slo_ttft and (r["tpot"] or 0) <= a.slo_tpot
            and r["e2e"] + r["queue_s"] <= a.slo_e2e]
    toks = sum(r["out_tokens"] for r in ok)
    return dict(level=level, requests=len(rows), errors=len(rows) - len(ok), wall_s=round(wall, 2),
                req_per_s=round(len(ok) / wall, 3), out_tok_per_s=round(toks / wall, 1),
                goodput_req_per_s=round(len(good) / wall, 3), slo_attainment=round(len(good) / max(len(rows), 1), 3),
                ttft_p50=pct([r["ttft"] + r["queue_s"] for r in ok], 50), ttft_p95=pct([r["ttft"] + r["queue_s"] for r in ok], 95),
                tpot_p50=pct([r["tpot"] for r in ok], 50), tpot_p95=pct([r["tpot"] for r in ok], 95),
                e2e_p50=pct(e2e, 50), e2e_p95=pct(e2e, 95),
                mean_out_tokens=round(statistics.mean(r["out_tokens"] for r in ok), 1) if ok else None,
                share_at_max_tokens=round(sum(r["out_tokens"] >= a.max_tokens - 2 for r in ok) / max(len(ok), 1), 3))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="output name, e.g. ollama / llamacpp / vllm_t4")
    ap.add_argument("--prompts", default="results/qwen_rag_v4_prompts.jsonl,results/qwen_norag_questions.jsonl")
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--rates", default="0.25,0.5,1,2")
    ap.add_argument("--requests", type=int, default=24, help="per load level")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=160)      # the agent's max_tokens
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--top-k", dest="top_k", type=int, default=30)
    ap.add_argument("--slo-ttft", dest="slo_ttft", type=float, default=1.0)
    ap.add_argument("--slo-tpot", dest="slo_tpot", type=float, default=0.1)
    ap.add_argument("--slo-e2e", dest="slo_e2e", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore-eos", dest="ignore_eos", action="store_true",
                    help="every answer runs to --max-tokens (vLLM, llama.cpp): equal output lengths across servers; "
                         "check share_at_max_tokens = 1.0, a server that ignores the flag stops early")
    ap.add_argument("--stop", default="</s>,<|user|>,<|system|>",
                    help="comma list sent with every request (TinyLlama chat format, = models/sprout/Modelfile); "
                         "'' for none. Servers differ in which stop strings they add on their own")
    a = ap.parse_args()
    a.stop = [] if a.ignore_eos else [x for x in a.stop.split(",") if x]
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"), timeout=300)
    levels = [float(x) for x in (a.rates if a.mode == "open" else a.concurrency).split(",")]
    prompts = load_prompts(a.prompts.split(","), a.requests, a.seed)
    await one(client, a, prompts[0], -1, None)                                # warm up (model load, caches)
    os.makedirs("results/serving", exist_ok=True)
    summary = dict(tag=a.tag, base_url=a.base_url, model=a.model, mode=a.mode, args=vars(a), levels=[])
    with open(f"results/serving/{a.tag}_requests.jsonl", "w", encoding="utf-8") as f:
        for lv in levels:
            rows, wall = await (open_loop(client, a, prompts, lv) if a.mode == "open"
                                else closed_loop(client, a, prompts, int(lv)))
            for r in rows:
                f.write(json.dumps(dict(r, level=lv)) + "\n")
            s = summarize(rows, wall, a, lv); summary["levels"].append(s)
            print(f"{a.mode} {lv:>5}: {s['req_per_s']:.2f} req/s, {s['out_tok_per_s']:.0f} tok/s, "
                  f"TTFT p50/p95 {s['ttft_p50']}/{s['ttft_p95']} s, TPOT p50 {s['tpot_p50']} s, "
                  f"E2E p95 {s['e2e_p95']} s, SLO met {s['slo_attainment']:.0%}, errors {s['errors']}")
    json.dump(summary, open(f"results/serving/{a.tag}.json", "w"), indent=2)
    print("saved results/serving/%s.json" % a.tag)


if __name__ == "__main__":
    asyncio.run(main())
