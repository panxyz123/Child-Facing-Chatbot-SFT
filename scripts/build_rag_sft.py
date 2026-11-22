"""Add retrieved facts to the SFT prompts, using the agent's own prompt code so training and serving match.

The targets stay the original Gemini answers (no regeneration, no API). Retrieval is local.
For router-"knowledge" rows, a share keep the facts in the system prompt (they learn to use them, or to ignore weak
ones, because the target does not depend on them); the rest keep the no-facts prompt, so the model also works
without retrieval. Everything else is written as before.

    python scripts/build_rag_sft.py                      # -> data/children_sft_v3_rag/{train,validation}.jsonl
Each row: the original fields plus messages=[system, user, assistant], with_facts, top_score.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.getcwd())
from sprout import AgentConfig, SproutAgent
from sprout.llm import StubLLM

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/children_sft_v2_splits")
ap.add_argument("--dst", default="data/children_sft_v3_rag")
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--table", default="wiki_v3")
ap.add_argument("--facts-share", type=float, default=0.7, help="share of knowledge rows that get the facts")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

cfg = AgentConfig(); cfg.trace_path = None; cfg.db_path, cfg.table = a.db, a.table
agent = SproutAgent(cfg, llm=StubLLM(), token_counter=lambda t: int(len(t.split()) * 1.4) + 1)
rng = random.Random(a.seed)
os.makedirs(a.dst, exist_ok=True)
for split in ("train", "validation"):                      # test stays untouched
    rows = [json.loads(l) for l in open(os.path.join(a.src, f"{split}.jsonl"), encoding="utf-8")]
    n_facts = n_know = 0
    with open(os.path.join(a.dst, f"{split}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            s = dict(age=r["age"], text=r["instruction"], message=r["instruction"], history=[])
            s.update(agent.route(s))
            with_facts, top = False, None
            if s["route"] == "knowledge":
                n_know += 1
                s.update(agent.retrieve(s)); top = s.get("top_score")
                with_facts = bool(s.get("passages")) and rng.random() < a.facts_share
                if not with_facts:
                    s["passages"] = []
            msgs, _, _ = agent.build_messages(s)
            msgs.append(dict(role="assistant", content=r["response"]))
            n_facts += with_facts
            f.write(json.dumps(dict(r, messages=msgs, route=s["route"], with_facts=with_facts, top_score=top),
                               ensure_ascii=False) + "\n")
    print(f"{split}: {len(rows)} rows, {n_know} routed to knowledge, {n_facts} with facts")
