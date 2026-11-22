"""Eyeball check, no judge: 10 science questions from the finished run. The existing model's answers are read from
results/rag_e2e; only the new model (whatever the server runs) is generated.

    VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=qwen2.5:1.5b-instruct python scripts/peek_models.py > results/peek_qwen.txt
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "eval"))
import rag_e2e as E

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--table", default="wiki_v3")
a = ap.parse_args()

L = lambda c: {r["id"]: r for r in map(json.loads, open(f"results/rag_e2e/{c}.jsonl", encoding="utf-8"))}
old = {c: L(c) for c in ("norag", "rag")}
ids = [i for i, r in old["rag"].items() if r["domain"] in E.SCIENCE and i in old["norag"]][: a.k]
agents = {c: E.make_agent(c, a) for c in ("norag", "rag")}
print("new model:", os.environ.get("VLLM_MODEL"))
for i in ids:
    r = old["rag"][i]
    print("=" * 80, f"\n[{i}] age {r['age']} | {r['instruction']}")
    for c in ("norag", "rag"):
        print(f"--- OLD {c}\n{old[c][i]['prediction']}")
        o = agents[c].respond(r["instruction"], r["age"], session_id=f"peek-{c}-{i}", log=False)
        print(f"--- NEW {c} (gate={o['gate']}, top={o['top_score']})\n{o['answer']}")
