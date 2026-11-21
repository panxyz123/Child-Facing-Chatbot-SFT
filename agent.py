"""Chat with Sprout from the command line (needs a running vLLM server, the LanceDB index and models/router.joblib).

    vllm serve <merged SFT model> --dtype half --max-model-len 2048 --served-model-name sprout
    python agent.py --age 7-9                       # interactive
    python agent.py --age 4-6 "why is the sky blue?"  # one message

The agent itself lives in sprout/ (graph in sprout/agent.py).
"""
import argparse
import json

from sprout import AgentConfig, SproutAgent
from sprout.router import AlwaysRetrieve


def build_agent(cfg: AgentConfig) -> SproutAgent:
    import os
    router = None if os.path.exists(cfg.router_path) else AlwaysRetrieve()
    if router is not None:
        print(f"note: {cfg.router_path} not found, every message goes to retrieval (python scripts/train_router.py)")
    return SproutAgent(cfg, router=router)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?")
    ap.add_argument("--age", default="7-9", choices=["0-3", "4-6", "7-9", "10-13", "14-18"])
    ap.add_argument("--trace", action="store_true", help="print route, sources and per-node latency")
    a = ap.parse_args()
    cfg = AgentConfig()
    cfg.trace_path = cfg.trace_path or "logs/traces.jsonl"
    agent = build_agent(cfg)

    def show(msg):
        out = agent.respond(msg, a.age, session_id="cli")
        print(f"\nSprout: {out['answer']}")
        if a.trace:
            print(json.dumps({k: v for k, v in out.items() if k != "answer"}, indent=1))

    if a.message:
        show(a.message)
        return
    print(f"Sprout (age {a.age}). Empty line to quit.")
    while True:
        msg = input("\nchild: ").strip()
        if not msg:
            break
        show(msg)


if __name__ == "__main__":
    main()
