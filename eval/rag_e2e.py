"""End-to-end: does retrieval reduce hallucination, measured on the same questions, model and sampling?

The questions are the test questions that the router sends to retrieval and that the retrieval benchmark kept
(results/retrieval_v3/per_query.csv). Each condition runs the full Sprout agent once per question:
    norag     the router threshold is raised so nothing is retrieved: the model answers from its own knowledge
    rag       hybrid_rerank over the Simple-Wikipedia-extended index (wiki_v3), top-2 passages in the prompt
    rag_gate<t>   as rag, but passages are used only if the reranker's top-1 score is >= t
Answers are written in the format of src/judge_hallucination.py, which then judges them claim by claim.

    # needs the model server running (e.g. ollama: VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=sprout)
    python eval/rag_e2e.py generate --n 5                    # try the whole chain on 5 questions first
    python eval/rag_e2e.py generate                          # norag and rag on 100 questions, science and other in proportion
    python src/judge_hallucination.py --in results/rag_e2e/norag.jsonl --out results/rag_e2e/norag_judgments.jsonl
    python src/judge_hallucination.py --in results/rag_e2e/rag.jsonl   --out results/rag_e2e/rag_judgments.jsonl
    python eval/rag_e2e.py report                            # paired: the same questions, model and sampling
    python eval/rag_e2e.py generate --conditions rag_gate0.81   # later: a gate threshold
(Without a freshly generated norag, report falls back to the earlier answers in results/hallucination_v2.)
"""
import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
SCIENCE = {"physical science", "life science", "earth and space"}
OUT = "results/rag_e2e"


def load_questions(a):
    qids = [r["qid"] for r in csv.DictReader(open(a.queries, encoding="utf-8"))]
    test = [json.loads(l) for l in open(os.path.join(a.splits, "test.jsonl"), encoding="utf-8")]
    domain = {}
    if os.path.exists(a.domains):
        domain = {json.loads(l)["id"]: json.loads(l)["domain"] for l in open(a.domains, encoding="utf-8")}
    rows = []
    for q in qids:
        t = test[int(q.split("-")[1])]
        rows.append(dict(id=q, split="test", age=t["age"], instruction=t["instruction"], reference=t["response"],
                         domain=domain.get(q, "unknown")))
    if a.n and a.n < len(rows):                 # fixed seed, science and the rest kept in proportion
        rng = random.Random(7)
        sci = [r for r in rows if r["domain"] in SCIENCE]; rest = [r for r in rows if r["domain"] not in SCIENCE]
        k = round(a.n * len(sci) / len(rows))
        rows = rng.sample(sci, min(k, len(sci))) + rng.sample(rest, min(a.n - k, len(rest)))
        rows.sort(key=lambda r: int(r["id"].split("-")[1]))
    return rows


def make_agent(cond, a):
    from sprout import AgentConfig, SproutAgent
    cfg = AgentConfig()
    cfg.trace_path = None
    cfg.db_path, cfg.table = a.db, a.table
    if cond == "norag":
        cfg.knowledge_threshold = 1.1           # a probability never reaches this: no retrieval
    elif cond.startswith("rag_gate"):
        cfg.gate_threshold = float(cond[len("rag_gate"):])
    elif cond != "rag":
        raise SystemExit(f"unknown condition {cond}")
    return SproutAgent(cfg)


def generate(a):
    rows = load_questions(a)
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(rows)} questions ({sum(r['domain'] in SCIENCE for r in rows)} science)")
    for cond in a.conditions.split(","):
        path = os.path.join(OUT, f"{cond}.jsonl")
        done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
        todo = [r for r in rows if r["id"] not in done]
        print(f"{cond}: {len(todo)} to generate")
        if not todo:
            continue
        agent = make_agent(cond, a)
        with open(path, "a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                out = agent.respond(r["instruction"], r["age"], session_id=f"{cond}-{r['id']}", log=False)
                rec = dict(id=r["id"], split="test", age=r["age"], instruction=r["instruction"],
                           reference=r["reference"], prediction=out["answer"], domain=r["domain"], condition=cond,
                           route=out["route"], gate=out["gate"], sources=out["sources"], top_score=out["top_score"],
                           fkgl=out["fkgl"], output_flag=out["output_flag"], regenerations=out["regenerations"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {cond} {i}/{len(todo)}")


def load_baseline(a):
    """The earlier no-retrieval answers and their claim judgments (results/hallucination_v2), same question ids."""
    gens = {}
    for l in open(a.baseline_generations, encoding="utf-8"):
        r = json.loads(l)
        gens[r["id"]] = dict(r, route="chat", gate="not_needed", sources=[], condition="norag")
    judged = {json.loads(l)["id"]: json.loads(l) for l in open(a.domains, encoding="utf-8")}
    for i, g in gens.items():
        g["domain"] = judged.get(i, {}).get("domain", "unknown")
    return gens, judged


def load_judged(cond):
    gens = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(OUT, f"{cond}.jsonl"), encoding="utf-8")}
    jp = os.path.join(OUT, f"{cond}_judgments.jsonl")
    judged = {json.loads(l)["id"]: json.loads(l) for l in open(jp, encoding="utf-8")} if os.path.exists(jp) else {}
    return gens, judged


def boot(deltas, n=5000, seed=0):
    rng = random.Random(seed)
    m = sum(deltas) / len(deltas)
    bs = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n))
    return m, bs[int(.025 * n)], bs[int(.975 * n)]


def report(a):
    conds = [c for c in a.conditions.split(",") if os.path.exists(os.path.join(OUT, f"{c}_judgments.jsonl"))]
    data = {c: load_judged(c) for c in conds}
    if "norag" not in data:                     # no new baseline: use the earlier one, paired by question id
        data["norag"] = load_baseline(a); conds = ["norag"] + conds
        print("baseline 'norag': the earlier answers in results/hallucination_v2 (SFT model, sampled in the "
              "hallucination notebook); the rag answers come from the agent's model and sampling.")
    # same questions in every condition; answers that the guard replaced by a safe reply are left out
    ids = set.intersection(*[set(j) for _, j in data.values()])
    ids = {i for i in ids if all(data[c][0][i]["route"] != "blocked" for c in conds)}
    ids = sorted(ids, key=lambda x: int(x.split("-")[1]))
    dom = {i: data[conds[0]][0][i]["domain"] for i in ids}
    subsets = {"all": ids, "science": [i for i in ids if dom[i] in SCIENCE], "other": [i for i in ids if dom[i] not in SCIENCE]}
    print(f"{len(ids)} questions judged in every condition: {conds}\n")
    for name, sub in subsets.items():
        if not sub:
            continue
        print(f"== {name} (n={len(sub)})")
        for c in conds:
            J = [data[c][1][i] for i in sub]
            err = sum(j["has_error"] for j in J) / len(J)
            cl = sum(j["n_claims"] for j in J); fl = sum(j["n_false"] for j in J)
            used = sum(data[c][0][i]["gate"] == "used" for i in sub) / len(sub)
            print(f"  {c:14s} answers with an error {err:6.1%} | false claims {fl / max(cl, 1):6.1%} ({fl}/{cl}) | "
                  f"passages used in {used:5.1%}")
        if "norag" in conds:
            for c in conds:
                if c == "norag":
                    continue
                d_err = [float(data[c][1][i]["has_error"]) - float(data["norag"][1][i]["has_error"]) for i in sub]
                d_fs = [(data[c][1][i]["false_share"] or 0) - (data["norag"][1][i]["false_share"] or 0) for i in sub]
                m, lo, hi = boot(d_err); m2, lo2, hi2 = boot(d_fs)
                print(f"  {c} minus norag, paired: answers with an error {m:+.1%} [{lo:+.1%}, {hi:+.1%}] | "
                      f"false-claim share per answer {m2:+.1%} [{lo2:+.1%}, {hi2:+.1%}]")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "report"])
    ap.add_argument("--conditions", default="norag,rag")
    ap.add_argument("--n", type=int, default=100, help="questions to use (0 = all that the router retrieves)")
    ap.add_argument("--queries", default="results/retrieval_v3/per_query.csv")
    ap.add_argument("--splits", default="data/children_sft_v2_splits")
    ap.add_argument("--domains", default="results/hallucination_v2/judgments.jsonl")
    ap.add_argument("--baseline-generations", dest="baseline_generations",
                    default="results/hallucination_v2/generations.jsonl")
    ap.add_argument("--db", default="rag/lancedb_data")
    ap.add_argument("--table", default="wiki_v3")
    a = ap.parse_args()
    generate(a) if a.mode == "generate" else report(a)


if __name__ == "__main__":
    main()
