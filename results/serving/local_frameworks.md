## Setup
- `ollama_stop`: http://localhost:11434/v1, mode closed, max_tokens 160, ignore_eos False, stop ['</s>', '<|user|>', '<|system|>'], requests/level 24
- `llamacpp_fixed`: http://localhost:8080/v1, mode closed, max_tokens 160, ignore_eos True, stop [], requests/level 24
- `vllm_metal_fixed`: http://localhost:8000/v1, mode closed, max_tokens 160, ignore_eos True, stop [], requests/level 24

## output tokens/s (all requests)

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 34 | 40 | 38 |
| 2 | 37 | 80 | 80 |
| 4 | 37 | 154 | 150 |
| 8 | 37 | 262 | 225 |

## requests/s

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 0.51 | 0.25 | 0.23 |
| 2 | 0.49 | 0.50 | 0.50 |
| 4 | 0.53 | 0.96 | 0.94 |
| 8 | 0.50 | 1.64 | 1.41 |

## TTFT p50 (s)

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 0.114 | 0.028 | 0.094 |
| 2 | 2.061 | 0.029 | 0.081 |
| 4 | 5.554 | 0.033 | 0.142 |
| 8 | 14.642 | 0.040 | 0.212 |

## TTFT p95 (s)

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 0.366 | 0.037 | 0.338 |
| 2 | 3.362 | 0.033 | 0.279 |
| 4 | 7.163 | 0.035 | 0.175 |
| 8 | 15.718 | 0.043 | 0.260 |

## TPOT p50 (s/token)

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 0.0268 | 0.0234 | 0.0255 |
| 2 | 0.0261 | 0.0250 | 0.0243 |
| 4 | 0.0261 | 0.0258 | 0.0262 |
| 8 | 0.0266 | 0.0307 | 0.0343 |

## E2E p95 (s)

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 3.15 | 4.45 | 4.69 |
| 2 | 5.44 | 4.03 | 4.41 |
| 4 | 8.70 | 4.27 | 4.44 |
| 8 | 17.59 | 5.09 | 5.95 |

## SLO met

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 100% | 100% | 100% |
| 2 | 4% | 100% | 100% |
| 4 | 4% | 100% | 100% |
| 8 | 4% | 100% | 100% |

## mean output tokens

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 67 | 160 | 160 |
| 2 | 76 | 160 | 160 |
| 4 | 70 | 160 | 160 |
| 8 | 74 | 160 | 160 |

## share at max_tokens

| concurrency | ollama_stop | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|---|
| 1 | 0% | 100% | 100% |
| 2 | 0% | 100% | 100% |
| 4 | 0% | 100% | 100% |
| 8 | 0% | 100% | 100% |

## Throughput relative to `ollama_stop` (output tokens/s)

| level | llamacpp_fixed | vllm_metal_fixed |
|---|---|---|
| 1 | 1.2x | 1.1x |
| 2 | 2.2x | 2.2x |
| 4 | 4.1x | 4.0x |
| 8 | 7.1x | 6.1x |

## Checks
- `ollama_stop`: mean output tokens 67 to 76
- `llamacpp_fixed`: mean output tokens 160 to 160
- `vllm_metal_fixed`: mean output tokens 160 to 160
