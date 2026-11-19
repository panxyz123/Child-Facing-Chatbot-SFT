"""
Build the v2 SFT dataset: 5 age groups x PER_AGE rows, with the age *set* (as a parent / app would set it),
never guessed from the child's message.

Inputs per age group ("typical" inputs for that age):
  - 0-3, 4-6       : real child speech reused from v1 (combined_sft_data.jsonl), topped up with HF questions
  - 7-9, 10-13, 14-18: HF questions written for that age (age read from the HF 'system' field)
Responses:
  - reused from v1 when the v1 row already fits (real child speech in 0-3 / 4-6, or an HF question whose
    v1 Gemini age equals its HF age)
  - otherwise written by Gemini for the set age
Knowledge label:
  - needs_knowledge = True when a correct answer depends on factual knowledge about the world, so a wrong
    fact would be a hallucination (the cases RAG is for). Gemini labels new rows while writing the reply and
    labels reused v1 rows in small batches.
QA:
  - within each age group, responses whose FKGL is an outlier (IQR rule) are dropped; ~10% extra rows are
    generated so each group still reaches PER_AGE

Run from the repo root (the folder that contains data/ and src/):
    python src/build_sft_dataset_v2.py --dry-run     # plan only: counts and number of API calls
    python src/build_sft_dataset_v2.py               # generate (resumable) and assemble
Needs GOOGLE_API_KEY in the environment or in google_gemini_key.env (repo root or its parent).
"""
import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
import pandas as pd

AGES = ["0-3", "4-6", "7-9", "10-13", "14-18"]
YOUNG = ["0-3", "4-6"]
INTENTS = ['narrative', 'scientific question', 'non-scientific question', 'request storytelling']

CONFIG = dict(
    v1_path="data/combined_sft_data.jsonl",
    local_csv="data/sampled_data.csv",
    hf_dataset="nikister/children-assistant",
    out_path="data/children_sft_v2.jsonl",
    cache_path="data/gen_cache_v2.jsonl",       # every Gemini answer is appended here; re-runs resume from it
    knowledge_cache_path="data/knowledge_cache_v2.jsonl",  # needs_knowledge labels for reused v1 rows
    label_batch=25,
    report_path="data/children_sft_v2_report.json",
    per_age=240,
    local_per_young_age=180,                    # real child speech in each of 0-3 and 4-6
    oversample=1.1,                             # extra generations so the FKGL filter can drop outliers
    model_id="gemini-3.5-flash-lite",   # gemini-2.0-flash-lite was retired
    delay_seconds=2,
    seed=42,
)


# =============================================================================
# Age parsing (HF 'system' field, e.g. "Child's age: 4-6", "8-year-old")
# =============================================================================
AGE_BUCKETS = [("0-3", 0, 3), ("4-6", 4, 6), ("7-9", 7, 9), ("10-13", 10, 13), ("14-18", 14, 18)]


def age_to_bucket(age):
    for name, lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return None


def parse_age(system_text):
    if not isinstance(system_text, str):
        return "N/A"
    text = system_text.lower()
    m = re.search(r"(?:aged?\s+)?(\d{1,2})\s*(?:-|to)\s*(\d{1,2})(?=\s*-?\s*years?|\s|,|\.|$)", text)
    if m and ("age" in text[max(0, m.start() - 6):m.start(1)] or "year" in text[m.end():m.end() + 8]):
        bucket = age_to_bucket((int(m.group(1)) + int(m.group(2))) // 2)
        if bucket:
            return bucket
    m = re.search(r"(\d{1,2})\s*-?\s*years?\s*-?\s*olds?|\baged?\s+(\d{1,2})\b", text)
    if m:
        bucket = age_to_bucket(int(m.group(1) or m.group(2)))
        if bucket:
            return bucket
    return "N/A"


# =============================================================================
# Loading
# =============================================================================
def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def load_v1():
    rows = [json.loads(line) for line in open(CONFIG["v1_path"], encoding="utf-8")]
    local = set(norm(t) for t in pd.read_csv(CONFIG["local_csv"], header=None, skiprows=1)[1].astype(str))
    for r in rows:
        r["source"] = "local" if norm(r["instruction"]) in local else "hf"
    return rows


def load_hf():
    from datasets import load_dataset
    df = load_dataset(CONFIG["hf_dataset"], split="train").to_pandas()
    df = df.reset_index().rename(columns={"index": "hf_idx", "user": "instruction"})
    df["age"] = df["system"].apply(parse_age)
    return df[df["age"] != "N/A"][["hf_idx", "instruction", "age"]]


# =============================================================================
# Plan: which rows are reused, which need a Gemini response
# =============================================================================
def make_plan(v1_rows, hf):
    rng = random.Random(CONFIG["seed"])
    per_age = CONFIG["per_age"]
    log = {"v1_rows": len(v1_rows), "v1_by_source_age": {}}
    for src in ("local", "hf"):
        log["v1_by_source_age"][src] = dict(Counter(r["age"] for r in v1_rows if r["source"] == src))

    reused = []  # rows kept with their v1 response

    # 1. real child speech for 0-3 / 4-6 (v1 labels outside 0-6 on child speech are dropped)
    for age in YOUNG:
        pool = [r for r in v1_rows if r["source"] == "local" and r["age"] == age]
        rng.shuffle(pool)
        for r in pool[:CONFIG["local_per_young_age"]]:
            reused.append(dict(r, age_source="v1 gemini (0-6 child speech)", origin="v1"))

    # 2. v1 rows from HF whose v1 (Gemini) age equals the HF age: label, response and source agree
    hf_ages = defaultdict(set)
    for text, age in zip(hf["instruction"], hf["age"]):
        hf_ages[norm(text)].add(age)
    used_text = set(norm(r["instruction"]) for r in v1_rows)  # never pick a v1 input again as a new one
    v1_hf_ok = defaultdict(list)
    n_found_other_age, n_not_found = 0, 0
    for r in v1_rows:
        if r["source"] != "hf":
            continue
        ages = hf_ages.get(norm(r["instruction"]), set())
        if ages == {r["age"]}:
            v1_hf_ok[r["age"]].append(r)
        elif ages:
            n_found_other_age += 1      # text is in HF, but Gemini's age differs from the HF age
        else:
            n_not_found += 1            # text not in HF (Gemini may have rewritten the instruction)
    n_found = sum(len(v) for v in v1_hf_ok.values()) + n_found_other_age
    log["v1_hf_age_matches"] = {a: len(v) for a, v in v1_hf_ok.items()}
    log["v1_hf_dropped_age_mismatch"] = n_found_other_age
    log["v1_hf_dropped_text_not_in_hf"] = n_not_found
    log["v1_gemini_age_accuracy_on_found"] = round(sum(len(v) for v in v1_hf_ok.values()) / max(n_found, 1), 3)

    # 3. quotas per age, filled by v1 HF rows first, then new Gemini generations
    tasks = []
    log["plan"] = {}
    for age in AGES:
        have_local = sum(1 for r in reused if r["age"] == age)
        need_hf = per_age - have_local
        ok = v1_hf_ok.get(age, [])
        rng.shuffle(ok)
        take = ok[:need_hf]
        for r in take:
            reused.append(dict(r, age_source="given (HF), matches v1", origin="v1"))
        remaining = need_hf - len(take)
        n_gen = math.ceil(remaining * CONFIG["oversample"]) if remaining > 0 else 0
        pool = hf[(hf["age"] == age) & (~hf["instruction"].map(norm).isin(used_text))]
        pool = pool.drop_duplicates(subset="instruction").sample(frac=1, random_state=CONFIG["seed"])
        for row in pool.head(n_gen).itertuples():
            tasks.append(dict(key=f"hf{row.hf_idx}", age=age, instruction=row.instruction))
        log["plan"][age] = dict(local_speech=have_local, v1_hf_reused=len(take),
                                new_gemini=n_gen, available_in_hf=len(pool))
    log["api_calls_planned"] = len(tasks)
    return reused, tasks, log


# =============================================================================
# Gemini generation (age is given, never inferred)
# =============================================================================
KNOWLEDGE_DEF = """needs_knowledge = true if a correct answer depends on factual knowledge about the world
(science, nature, animals, the body and health, history, geography, how things work, numbers and facts),
so that stating a wrong fact would be a hallucination. needs_knowledge = false for chit-chat, feelings,
greetings, play, pretend, telling or continuing a story, opinions, requests, or fragments that only need a
friendly reply."""

PROMPT = """You are writing training data for a chatbot that talks with children.
The parent has set the child's age to: {age} years old.
The child says: "{instruction}"

1. Classify the child's message into the closest intent from the allowed list.
2. Decide needs_knowledge. """ + KNOWLEDGE_DEF + """
3. Write the chatbot's reply for a child aged {age}: friendly, educational and safe, with vocabulary,
   sentence length and depth of knowledge suited to that age. Never give inappropriate information to children.
"""

LABEL_PROMPT = """For each child message below, decide needs_knowledge. """ + KNOWLEDGE_DEF + """
Return one item per message with the same id.

{items}
"""


FATAL = ("404", "NOT_FOUND", "PERMISSION_DENIED", "API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED")


def check_fatal(e):
    """Errors that retrying cannot fix (wrong model id, bad key, daily quota used up): stop instead of
    looping over every row. Progress is cached, so just re-run later."""
    if any(k in str(e) for k in FATAL):
        raise SystemExit(f"Stopping, not retryable: {str(e)[:300]}")
    if "PerDay" in str(e) or "per day" in str(e).lower():
        raise SystemExit("Stopping: daily quota used up. Re-run after it resets; finished rows are cached.")


def backoff(e, attempt):
    """Rate limit (429) or overload (503): wait 15, 30, 60, 120 s. Other errors: short wait."""
    busy = any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
    return 15 * 2 ** attempt if busy else 5


def make_client():
    from dotenv import load_dotenv
    for path in ("google_gemini_key.env", "../google_gemini_key.env"):
        if os.path.exists(path):
            load_dotenv(path)
    from google import genai
    return genai.Client()


def gemini_reply(client, age, instruction):
    from pydantic import BaseModel

    class Reply(BaseModel):
        intent: Literal['narrative', 'scientific question', 'non-scientific question', 'request storytelling']
        needs_knowledge: bool
        response: str

    out = client.models.generate_content(
        model=CONFIG["model_id"],
        contents=PROMPT.format(age=age, instruction=instruction),
        config={"response_mime_type": "application/json", "response_schema": Reply},
    )
    return out.parsed.model_dump() if hasattr(out.parsed, "model_dump") else out.parsed.dict()


def gemini_label_batch(client, items):
    """items: list of (id, text). Returns {id: bool}."""
    from pydantic import BaseModel

    class Label(BaseModel):
        id: int
        needs_knowledge: bool

    listing = "\n".join(f"[{i}] {text}" for i, text in items)
    out = client.models.generate_content(
        model=CONFIG["model_id"], contents=LABEL_PROMPT.format(items=listing),
        config={"response_mime_type": "application/json", "response_schema": list[Label]},
    )
    return {x.id: x.needs_knowledge for x in out.parsed}


def run_knowledge_labels(rows, label_fn):
    """Label reused v1 rows (cached by normalized text, resumable)."""
    cache = {}
    if os.path.exists(CONFIG["knowledge_cache_path"]):
        for line in open(CONFIG["knowledge_cache_path"], encoding="utf-8"):
            x = json.loads(line)
            cache[x["text"]] = x["needs_knowledge"]
    todo = sorted({norm(r["instruction"]) for r in rows} - set(cache))
    print(f"knowledge labels: {len(cache)} cached, {len(todo)} to label")
    for start in range(0, len(todo), CONFIG["label_batch"]):
        batch = list(enumerate(todo[start:start + CONFIG["label_batch"]]))
        for attempt in range(4):
            try:
                labels = label_fn(batch)
                with open(CONFIG["knowledge_cache_path"], "a", encoding="utf-8") as f:
                    for i, text in batch:
                        if i in labels:
                            cache[text] = bool(labels[i])
                            f.write(json.dumps(dict(text=text, needs_knowledge=cache[text]), ensure_ascii=False) + "\n")
                break
            except Exception as e:
                check_fatal(e)
                wait = backoff(e, attempt)
                print(f"  label batch failed ({str(e)[:80]}), sleeping {wait}s")
                time.sleep(wait)
        time.sleep(CONFIG["delay_seconds"])
    for r in rows:
        r["needs_knowledge"] = cache.get(norm(r["instruction"]))


def run_generation(tasks, reply_fn):
    done = set()
    if os.path.exists(CONFIG["cache_path"]):
        done = {json.loads(line)["key"] for line in open(CONFIG["cache_path"], encoding="utf-8")}
    pending = [t for t in tasks if t["key"] not in done]
    print(f"Gemini: {len(tasks)} planned, {len(done & {t['key'] for t in tasks})} cached, {len(pending)} to go")
    gave_up_in_a_row = 0
    for i, t in enumerate(pending, 1):
        for attempt in range(4):
            try:
                reply = reply_fn(t["age"], t["instruction"])
                with open(CONFIG["cache_path"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(t, **reply), ensure_ascii=False) + "\n")
                print(f"[{i}/{len(pending)}] {t['age']:>5} | {t['instruction'][:50]}")
                gave_up_in_a_row = 0
                break
            except Exception as e:
                check_fatal(e)
                wait = backoff(e, attempt)
                print(f"  attempt {attempt + 1} failed ({str(e)[:80]}), sleeping {wait}s")
                time.sleep(wait)
        else:
            print(f"  giving up on {t['key']}")
            gave_up_in_a_row += 1
            if gave_up_in_a_row >= 3:
                raise SystemExit("Stopping: 3 rows in a row failed (quota or outage). Re-run later; progress is cached.")
        time.sleep(CONFIG["delay_seconds"])


# =============================================================================
# Assemble + FKGL QA
# =============================================================================
def fkgl(text):
    import textstat
    return float(textstat.flesch_kincaid_grade(text)) if str(text).strip() else float("nan")


def assemble(reused, tasks, log):
    task_keys = {t["key"] for t in tasks}
    generated = []
    if os.path.exists(CONFIG["cache_path"]):
        for line in open(CONFIG["cache_path"], encoding="utf-8"):
            g = json.loads(line)
            if g["key"] in task_keys:
                generated.append(dict(age=g["age"], intent=g["intent"], needs_knowledge=g["needs_knowledge"],
                                      instruction=g["instruction"], response=g["response"], source="hf",
                                      age_source="given (HF)", origin="gemini v2"))
    rows = [dict(age=r["age"], intent=r.get("intent", "N/A"), needs_knowledge=r.get("needs_knowledge"),
                 instruction=r["instruction"], response=r["response"], source=r["source"],
                 age_source=r["age_source"], origin=r["origin"])
            for r in reused] + generated
    for r in rows:
        r["fkgl"] = fkgl(r["response"])

    rng = random.Random(CONFIG["seed"])
    final, log["qa"] = [], {}
    for age in AGES:
        group = [r for r in rows if r["age"] == age and r["response"].strip()]
        f = np.array([r["fkgl"] for r in group])
        q1, q3 = np.nanpercentile(f, 25), np.nanpercentile(f, 75)
        margin = max(1.5 * (q3 - q1), 2.0)  # at least 2 grade levels, so a narrow IQR doesn't drop normal rows
        lo, hi = float(q1 - margin), float(q3 + margin)
        kept = [r for r in group if lo <= r["fkgl"] <= hi]
        # trim the extra generations first, so reused real child speech is kept
        kept.sort(key=lambda r: (r["origin"] == "gemini v2", rng.random()))
        kept = kept[:CONFIG["per_age"]]
        final += kept
        log["qa"][age] = dict(candidates=len(group), fkgl_band=[round(lo, 2), round(hi, 2)],
                              dropped_outliers=len(group) - sum(lo <= r["fkgl"] <= hi for r in group),
                              final=len(kept), final_by_origin=dict(Counter(r["origin"] for r in kept)),
                              needs_knowledge=dict(Counter(str(r["needs_knowledge"]) for r in kept)),
                              fkgl_median=round(float(np.nanmedian([r["fkgl"] for r in kept])), 2))
    rng.shuffle(final)
    with open(CONFIG["out_path"], "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    medians = [log["qa"][a]["fkgl_median"] for a in AGES]
    log["fkgl_median_increases_with_age"] = all(x < y for x, y in zip(medians, medians[1:]))
    log["final_rows"] = len(final)
    short = {a: CONFIG["per_age"] - log["qa"][a]["final"] for a in AGES if log["qa"][a]["final"] < CONFIG["per_age"]}
    if short:
        print("WARNING: groups below per_age (raise 'oversample' and re-run; cached replies are reused):", short)
    with open(CONFIG["report_path"], "w") as f:
        json.dump(log, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return final, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no API calls, no output files")
    args = ap.parse_args()

    v1_rows, hf = load_v1(), load_hf()
    reused, tasks, log = make_plan(v1_rows, hf)
    print(json.dumps({k: log[k] for k in ("v1_by_source_age", "v1_hf_age_matches",
                                           "v1_hf_dropped_age_mismatch", "v1_hf_dropped_text_not_in_hf",
                                           "v1_gemini_age_accuracy_on_found", "plan", "api_calls_planned")}, indent=2))
    if args.dry_run:
        return
    client = make_client()
    run_knowledge_labels(reused, lambda items: gemini_label_batch(client, items))
    run_generation(tasks, lambda age, text: gemini_reply(client, age, text))
    final, log = assemble(reused, tasks, log)
    print(json.dumps(log["qa"], indent=2))
    print("FKGL median increases with age:", log["fkgl_median_increases_with_age"])
    print(f"wrote {len(final)} rows -> {CONFIG['out_path']}  (report: {CONFIG['report_path']})")


if __name__ == "__main__":
    main()
