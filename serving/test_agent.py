"""
End-to-end smoke test + latency breakdown for the RAG agent (agent.py) on vLLM.

Colab usage (T4, after `pip install vllm langgraph lancedb sentence-transformers langchain-community langchain-openai`):

    # upload rag/lancedb_data/wiki_science_v2_fixed.lance into /content/lancedb_data/
    !python serving/test_agent.py --db-path /content/lancedb_data --start-server

Without --start-server, the script expects a vLLM server already running on VLLM_BASE_URL.
Fails loudly if the LanceDB table can't be opened or retrieval returns empty text,
instead of letting the agent answer silently without context.
"""

import argparse
import json
import os
import subprocess
import sys
import time

QUERIES = [
    "Why is the sky blue?",
    "Why is winter cold?",
    "How do plants make their food?",
    "What are stars made of?",
    "Why do we need to sleep?",
]

parser = argparse.ArgumentParser()
parser.add_argument("--db-path", default="./lancedb_data")
parser.add_argument("--table", default="wiki_science_v2_fixed")
parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="model path for --start-server (e.g. merged_model)")
parser.add_argument("--start-server", action="store_true")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# agent.py reads these at import time
os.environ["LANCE_STORAGE_PATH"] = args.db_path
os.environ["LANCE_TABLE_NAME"] = args.table
os.environ["LLM_BACKEND"] = "vllm"
os.environ["VLLM_BASE_URL"] = f"http://localhost:{args.port}/v1"
os.environ["VLLM_MODEL"] = "tinyllama"

import requests  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent import RAGAgent  # noqa: E402


def wait_for_server(proc=None, timeout_s=600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            sys.exit("vLLM server exited, see vllm_agent.log")
        try:
            if requests.get(f"http://localhost:{args.port}/health", timeout=2).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    sys.exit("vLLM server not reachable")


proc = None
if args.start_server:
    proc = subprocess.Popen(
        ["vllm", "serve", args.model, "--dtype", "half", "--max-model-len", "2048",
         "--gpu-memory-utilization", "0.7", "--port", str(args.port),
         "--served-model-name", "tinyllama"],
        stdout=open("vllm_agent.log", "w"), stderr=subprocess.STDOUT)
wait_for_server(proc)
print("✓ vLLM server up")

try:
    agent = RAGAgent()

    # ---- sanity checks on retrieval ----
    table = agent.retriever.table
    if table is None:
        sys.exit(f"LanceDB table '{args.table}' not found under {args.db_path}")
    print(f"✓ table rows: {table.count_rows()}, columns: {table.schema.names}")
    docs = agent.retriever.retrieve(QUERIES[0])
    if not docs or not any(d["text"] for d in docs):
        sys.exit("Retrieval returned no documents or empty text: check column names")
    print("✓ retrieval sample:", [d["title"] for d in docs])

    # ---- time each stage by wrapping retrieve / generate ----
    timings = {}

    def timed(name, fn):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            timings[name] = (time.perf_counter() - t0) * 1e3
            return out
        return wrapper

    agent.retriever.retrieve = timed("retrieve_ms", agent.retriever.retrieve)
    agent.generator.generate = timed("generate_ms", agent.generator.generate)

    agent.run("warmup question")  # warm up embedding model and server
    rows = []
    for q in QUERIES:
        timings.clear()
        t0 = time.perf_counter()
        print(agent.run(q))
        total = (time.perf_counter() - t0) * 1e3
        rows.append(dict(query=q, total_ms=total, **timings))

    print(f"\n{'query':<34}{'retrieve':>10}{'generate':>10}{'total':>10}  (ms)")
    for r in rows:
        print(f"{r['query']:<34}{r['retrieve_ms']:>10.0f}{r['generate_ms']:>10.0f}{r['total_ms']:>10.0f}")
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ("retrieve_ms", "generate_ms", "total_ms")}
    print(f"{'mean':<34}{avg['retrieve_ms']:>10.0f}{avg['generate_ms']:>10.0f}{avg['total_ms']:>10.0f}")
    print(f"LLM generation share of end-to-end latency: {avg['generate_ms'] / avg['total_ms']:.0%}")
    json.dump(dict(per_query=rows, mean=avg), open("agent_latency.json", "w"), indent=2)
finally:
    if proc is not None:
        proc.terminate()
        proc.wait(60)
