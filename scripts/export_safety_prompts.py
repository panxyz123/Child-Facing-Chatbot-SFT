"""Step 1 of the end-to-end guardrail evaluation (runs on the Mac, model servers off): put every test prompt of the
safety suite through the agent's steps before generation, exactly as SproutAgent does them.

  input rules (sprout/guards.check_input) -> blocked / support prompts get the canned safe reply, no model call
  redaction (personal info removed from the text the model sees)
  router -> retrieval + gate for knowledge questions -> build_messages (system prompt for the child's age)

Generation, the output rules + Qwen3Guard-Gen-0.6B, and the WildGuard judge run on Colab
(eval/safety_e2e_colab.ipynb); eval/safety_e2e_report.py computes the rates.

    python scripts/export_safety_prompts.py            # -> drive/llm_eval_runs/safety_e2e/prompts.jsonl
    python scripts/export_safety_prompts.py --limit 50 # quick check

The suite has no child ages: each prompt gets one of the five age bands at random (seed 7). Test split = the suite
minus results/guardrails/guard_models/dev_ids.json (the dev split was used to pick the guard model).
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.getcwd())
from sprout import AgentConfig, SproutAgent
from sprout import guards
from sprout.config import AGES
from sprout.llm import StubLLM

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="data/safety/suite.jsonl")
ap.add_argument("--dev-ids", default="results/guardrails/guard_models/dev_ids.json")
ap.add_argument("--out", default="drive/llm_eval_runs/safety_e2e/prompts.jsonl")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

dev = set(json.load(open(a.dev_ids)))
rows = [r for r in map(json.loads, open(a.suite, encoding="utf-8")) if r["id"] not in dev]
rng = random.Random(7)
for r in rows:
    r["age"] = rng.choice(AGES)
if a.limit:
    rows = rng.sample(rows, a.limit)
cfg = AgentConfig(); cfg.trace_path = None; cfg.guard_url = None
agent = SproutAgent(cfg, llm=StubLLM())
os.makedirs(os.path.dirname(a.out), exist_ok=True)
n_block = n_know = 0; t0 = time.time()
with open(a.out, "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        g = guards.check_input(r["text"])
        out = dict(r, input_category=g.category, input_action=g.action, redacted=g.action == "redact")
        if not g.passed:                                   # block or support: canned reply, the model never runs
            n_block += 1
            out.update(stage="input_rule", final=guards.safe_reply(g.category, r["age"], cfg.crisis_line), messages=None)
        else:
            s = dict(age=r["age"], text=g.text, message=r["text"], history=[])
            s.update(agent.route(s))
            if s["route"] == "knowledge":
                n_know += 1; s.update(agent.retrieve(s))
            msgs, _, _ = agent.build_messages(s)
            out.update(stage=None, model_text=g.text, route=s["route"], gate=s.get("gate"), messages=msgs)
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
print(f"wrote {a.out}: {len(rows)} prompts, {n_block} stopped by the input rules, {n_know} routed to retrieval")
