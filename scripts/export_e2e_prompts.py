"""Export the agent's rag prompts (route -> retrieve -> build_messages, no gate) for questions in a jsonl, so the answers
can be generated elsewhere (e.g. Colab). Retrieval and the router run here.

    python scripts/export_e2e_prompts.py --in results/qwen_norag_questions.jsonl --out results/qwen_rag_prompts.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd())
from sprout import AgentConfig, SproutAgent
from sprout.llm import StubLLM

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--table", default="wiki_v3")
a = ap.parse_args()

cfg = AgentConfig(); cfg.trace_path = None; cfg.db_path, cfg.table = a.db, a.table
agent = SproutAgent(cfg, llm=StubLLM(), token_counter=lambda t: int(len(t.split()) * 1.4) + 1)
n_used = 0
with open(a.out, "w", encoding="utf-8") as f:
    for r in map(json.loads, open(a.inp, encoding="utf-8")):
        s = dict(age=r["age"], text=r["instruction"], message=r["instruction"], history=[])
        s.update(agent.route(s))
        if s["route"] == "knowledge":
            s.update(agent.retrieve(s))
        msgs, _, _ = agent.build_messages(s)
        n_used += s.get("gate") == "used"
        f.write(json.dumps(dict(r, messages=msgs, route=s["route"], gate=s.get("gate"), top_score=s.get("top_score"),
                                sources=[p["title"] for p in s.get("passages", [])]), ensure_ascii=False) + "\n")
print(f"wrote {a.out}; passages used for {n_used} questions")
