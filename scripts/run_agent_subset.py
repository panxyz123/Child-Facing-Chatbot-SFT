"""Run the full agent (guards, router, retrieval, regeneration) on a question file with a chosen temperature, for both
conditions, without changing sprout/ (only the config object is overridden). Same model server as rag_e2e.

    VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=sprout \
      python scripts/run_agent_subset.py --temperature 0.3 --tag tinyllama
    -> results/tinyllama_norag_t03.jsonl, results/tinyllama_rag_t03.jsonl (judge_hallucination.py format)
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "eval"))
import rag_e2e as E

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="results/qwen_norag_questions.jsonl")
ap.add_argument("--temperature", type=float, required=True)
ap.add_argument("--tag", required=True, help="model name in the output file names")
ap.add_argument("--conditions", default="norag,rag")
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--table", default="wiki_v3")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
t = f"t{str(a.temperature).replace('.', '')}"
for cond in a.conditions.split(","):
    agent = E.make_agent(cond, a)
    agent.cfg.temperature = agent.llm.params["temperature"] = a.temperature   # ChatLLM copies it at init
    path = f"results/{a.tag}_{cond}_{t}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            o = agent.respond(r["instruction"], r["age"], session_id=f"{cond}-{r['id']}", log=False)
            f.write(json.dumps(dict(r, prediction=o["answer"], condition=f"{a.tag}_{cond}_{t}", route=o["route"],
                                    gate=o["gate"], sources=o["sources"], top_score=o["top_score"]),
                               ensure_ascii=False) + "\n")
    print("wrote", path)
