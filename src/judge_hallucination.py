"""Claim-level hallucination judge for model answers to knowledge questions (Gemini, structured output).

Input: generations.jsonl from eval/hallucination_analysis.ipynb (one row per question: id, age, instruction,
reference, prediction, ...). For each answer, one Gemini call labels the question (type, domain), splits the answer
into atomic factual claims, and gives each claim a verdict. Evidence is the judge's own knowledge plus the
reference answer; no retrieval yet (this measures the model without RAG).

Runs locally, resumable: every judgment is appended to --out; re-running skips ids already there. Stops on daily
quota or non-retryable errors (re-run later).

    python src/judge_hallucination.py --in generations.jsonl --out results/hallucination_v2/judgments.jsonl
    python src/judge_hallucination.py --in ... --out ... --limit 5        # try a few first
    python src/judge_hallucination.py --in ... --out ... --dry-run        # print one prompt, no API call
    python src/judge_hallucination.py --in ... --out ... --review 50      # CSV of claims for manual labeling

Needs GOOGLE_API_KEY in the environment or in google_gemini_key.env (repo root or its parent).
"""
import argparse
import csv
import json
import os
import random
import time
from typing import List, Literal

MODEL_ID = "gemini-3.5-flash-lite"
DELAY_SECONDS = 2
FATAL = ("404", "NOT_FOUND", "PERMISSION_DENIED", "API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED")

QUESTION_TYPES = ["explanation", "fact", "precise"]
DOMAINS = ["physical science", "life science", "earth and space", "health and body", "history",
           "society and culture", "language and arts", "other"]
VERDICTS = ["correct", "simplified", "false", "unverifiable"]

PROMPT = """You are checking an AI assistant's answer to a child for factual errors.

The child's age group: {age}
The child's question: {question}
A reference answer (one good answer, not the only correct one): {reference}
The assistant's answer to check: {answer}

1. Classify the question.
   - question_type: "explanation" (why / how something happens), "fact" (what something is, a definition or
     description), or "precise" (the answer hinges on a specific number, date, name or place)
   - domain: one of {domains}

2. Split the assistant's answer into atomic factual claims: short statements about the world that can be true or
   false. Skip greetings, praise, questions back to the child, opinions, suggestions, and events inside a story
   that is clearly fictional. At most 10 claims; if there are more, keep the most important ones.

3. Give each claim a verdict, using your own knowledge (the reference is a hint, not the standard: a claim missing
   from the reference is not wrong for that reason):
   - "correct": true as stated
   - "simplified": not precise, but an acceptable simplification for a child of this age that does not teach
     something false (e.g. "plants make their food from sunlight" for a 5-year-old)
   - "false": wrong in a way a knowledgeable adult would correct: invented terms, people, places or facts, wrong
     causes or mechanisms, wrong numbers, dates or names, or a simplification that teaches a misconception
   - "unverifiable": cannot be judged true or false
   Give a short note for every "false" claim saying what is wrong.

If the answer contains no factual claims, return an empty claims list.
"""


def check_fatal(e):
    if any(k in str(e) for k in FATAL):
        raise SystemExit(f"Stopping, not retryable: {str(e)[:300]}")
    if "PerDay" in str(e) or "per day" in str(e).lower():
        raise SystemExit("Stopping: daily quota used up. Re-run after it resets; finished rows are cached.")


def backoff(e, attempt):
    busy = any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
    return 15 * 2 ** attempt if busy else 5


def make_client():
    from dotenv import load_dotenv
    for path in ("google_gemini_key.env", "../google_gemini_key.env"):
        if os.path.exists(path):
            load_dotenv(path)
    from google import genai
    return genai.Client()


def build_prompt(row):
    return PROMPT.format(age=row["age"], question=row["instruction"], reference=row["reference"],
                         answer=row["prediction"], domains=", ".join(f'"{d}"' for d in DOMAINS))


def make_judge(client, model_id):
    from pydantic import BaseModel

    class Claim(BaseModel):
        claim: str
        verdict: Literal["correct", "simplified", "false", "unverifiable"]
        note: str

    class Judgment(BaseModel):
        question_type: Literal["explanation", "fact", "precise"]
        domain: Literal["physical science", "life science", "earth and space", "health and body", "history",
                        "society and culture", "language and arts", "other"]
        claims: List[Claim]

    def judge(row):
        out = client.models.generate_content(
            model=model_id, contents=build_prompt(row),
            config={"response_mime_type": "application/json", "response_schema": Judgment})
        p = out.parsed
        return p.model_dump() if hasattr(p, "model_dump") else p.dict()
    return judge


def summarize(j):
    """Answer-level fields derived from the claim verdicts."""
    v = [c["verdict"] for c in j["claims"]]
    checkable = [x for x in v if x != "unverifiable"]
    return dict(n_claims=len(v), n_false=v.count("false"), n_simplified=v.count("simplified"),
                has_error=v.count("false") > 0,
                false_share=(v.count("false") / len(checkable)) if checkable else None)


def run(rows, judge, out_path, delay=DELAY_SECONDS):
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    pending = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} answers, {len(rows) - len(pending)} already judged, {len(pending)} to go")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gave_up_in_a_row = 0
    for i, r in enumerate(pending, 1):
        for attempt in range(4):
            try:
                j = judge(r)
                rec = dict(id=r["id"], judge_model=judge.model_id if hasattr(judge, "model_id") else None, **j, **summarize(j))
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(pending)}] {r['age']:>5} | {rec['n_false']}/{rec['n_claims']} false | {r['instruction'][:50]}")
                gave_up_in_a_row = 0
                break
            except Exception as e:
                check_fatal(e)
                wait = backoff(e, attempt)
                print(f"  attempt {attempt + 1} failed ({str(e)[:80]}), sleeping {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on {r['id']}")
            gave_up_in_a_row += 1
            if gave_up_in_a_row >= 3:
                raise SystemExit("Stopping: 3 answers in a row failed (quota or outage). Re-run later; progress is saved.")
        time.sleep(delay)


def write_review(rows, out_path, n, seed=42):
    """Sample n judged answers and write their claims to a CSV with an empty human_verdict column."""
    by_id = {r["id"]: r for r in rows}
    judged = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    judged = [j for j in judged if j["claims"]]
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    path = os.path.splitext(out_path)[0] + "_review.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "age", "question", "answer", "claim", "judge_verdict", "judge_note", "human_verdict"])
        for j in sample:
            r = by_id[j["id"]]
            for c in j["claims"]:
                w.writerow([j["id"], r["age"], r["instruction"], r["prediction"], c["claim"], c["verdict"], c["note"], ""])
    print(f"wrote {path}: fill human_verdict with one of {VERDICTS}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="generations.jsonl from the notebook")
    ap.add_argument("--out", required=True, help="judgments.jsonl (appended, resumable)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pending answers")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for the first answer and exit")
    ap.add_argument("--review", type=int, default=0, help="write a CSV of N judged answers for manual labeling")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    if a.review:
        return write_review(rows, a.out, a.review)
    if a.dry_run:
        print(build_prompt(rows[0]))
        return
    judge = make_judge(make_client(), a.model)
    judge.model_id = a.model
    if a.limit:
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if os.path.exists(a.out) else set()
        rows = [r for r in rows if r["id"] not in done][: a.limit]
    run(rows, judge, a.out)


if __name__ == "__main__":
    main()
