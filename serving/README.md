# Serving

How many children can one machine serve within a latency target, and which serving stack gets there?

## Highlights

- **Batching decides capacity, not raw speed.** On a MacBook Air every stack decodes a single request at about
  40 tokens/s, close to the memory-bandwidth limit (about 45 tokens/s for 2.2 GB of fp16 weights at ~100 GB/s).
  Under concurrency, llama.cpp and vLLM (vllm-metal) batch requests and reach **7.1x and 6.1x Ollama's throughput
  at 8 users**, with every request inside the latency target; Ollama's MLX backend serves them one at a time.
- **One load generator for every stack**, driven by the agent's real prompts (retrieved passages included):
  `scripts/bench_openai.py` streams each request and reports TTFT, TPOT, end-to-end latency and goodput (share of
  requests meeting all SLOs), in closed-loop (N users) or open-loop (Poisson arrivals) mode.
- **vLLM vs HF `generate` on a T4** (earlier micro-benchmark, TinyLlama): 2.5x faster decode, 3.0x throughput at 8
  concurrent requests, and 6.3x lower TTFT with prefix caching on a 1.1k-token shared prefix (an upper bound).

## Local stacks on a laptop (`results/serving/`)

MacBook Air (Apple Silicon, Metal GPU), the same TinyLlama SFT fp16 weights in all three, 24 requests per level,
160 output tokens, temperature 0.3, prompts exported from the agent. SLO: TTFT <= 1 s, TPOT <= 0.1 s, E2E <= 10 s.

| users | Ollama 0.40 (MLX) | llama.cpp `llama-server -np 8` | vllm-metal |
|---|---|---|---|
| output tokens/s, 1 | 34 | 40 | 38 |
| output tokens/s, 8 | 37 | **262** | 225 |
| TTFT p95, 8 | 15.7 s | **0.04 s** | 0.26 s |
| TPOT p50, 8 | 27 ms | 31 ms | 34 ms |
| requests meeting the SLO, 8 | 4% | 100% | 100% |

Full tables: `python scripts/analyze_serving.py --runs ollama_stop,llamacpp_fixed,vllm_metal_fixed --baseline
ollama_stop` (written to `results/serving/local_frameworks.md`).

- **Ollama serializes**: throughput stays at ~37 tokens/s and TTFT grows linearly with users, although
  `OLLAMA_NUM_PARALLEL=8` was set and confirmed in the server's environment. This matches a known limitation of the
  MLX runner (ollama/ollama#17280). One user gets a fast answer (TTFT 0.1 s); two users already miss the SLO.
- **llama.cpp and vllm-metal tie up to 4 users**; at 8, llama.cpp is ahead (262 vs 225 tokens/s, TTFT p95 0.04 vs
  0.26 s), likely the per-step overhead of vLLM's Python scheduler and server at this small scale (not profiled).
  vLLM's strengths (paged KV cache under memory pressure, prefix caching, scheduling at high concurrency) are what
  the GPU tests target.
- **Deployment reading**: a single family on a laptop is served well by any of them; a shared server needs a
  batching stack, and on one Mac llama.cpp is the best of the three.

Setup notes: llama.cpp and vllm-metal ran with `--ignore-eos` (every answer exactly 160 tokens); Ollama ignores that
flag and produced its natural lengths (mean ~70 tokens), which favours Ollama, since a serial server's queueing
delay grows with answer length. A fanless laptop throttles under sustained load (TPOT doubled in one run, kept in
`ollama.json` and excluded); runs were repeated after cooling. `ollama_run1.json` is transcribed from console output.

## Next: the production model on GPUs

The numbers above compare stacks with the TinyLlama weights. The production generator is Qwen2.5-1.5B-Instruct +
the Sprout LoRA (`scripts/merge_lora.py`), so every number below is measured with it:

| test | where | question it answers |
|---|---|---|
| local stacks again, with Qwen | MacBook Air | does the ranking hold for the production model |
| agent end to end (guards, router, retrieval) | MacBook Air | is CPU retrieval (~0.9 s p50) the bottleneck before the GPU |
| capacity: closed loop, 1 to 64 users, fixed lengths | T4, RTX Pro 6000 | scaling with memory bandwidth, laptop to datacenter GPU |
| goodput: open loop, Poisson arrivals, natural lengths | T4, RTX Pro 6000 | the highest request rate that meets the SLO, i.e. children per GPU |
| prefix caching on real multi-turn sessions | T4, RTX Pro 6000 | the real gain, against the 6.3x synthetic upper bound |

All stacks then use the same sampling (temperature 0.3, top-p 0.9, top-k 30, repetition penalty 1.05; Ollama takes
the penalty from its Modelfile).
