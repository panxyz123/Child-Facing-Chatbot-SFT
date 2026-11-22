"""Paired on the same questions, all judged by the same claim judge: TinyLlama SFT norag (results/rag_e2e) vs
Qwen SFT norag / rag (results/qwen_*_judgments.jsonl, whichever exist)."""
import json, os
L = lambda p: {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
runs = {"TinyLlama norag": "results/rag_e2e/norag_judgments.jsonl", "TinyLlama rag": "results/rag_e2e/rag_judgments.jsonl",
        "Qwen norag": "results/qwen_norag_judgments.jsonl", "Qwen rag": "results/qwen_rag_judgments.jsonl",
        "Qwen norag t0.3": "results/qwen_norag_t03_judgments.jsonl",
        "Qwen rag t0.3": "results/qwen_rag_t03_judgments.jsonl",
        "TinyLlama norag t0.3": "results/tinyllama_norag_t03_judgments.jsonl",
        "TinyLlama rag t0.3": "results/tinyllama_rag_t03_judgments.jsonl"}
J = {k: L(p) for k, p in runs.items() if os.path.exists(p)}
def stats(names):
    ids = sorted(set.intersection(*(set(J[k]) for k in names)))
    print(f"--- n={len(ids)}: questions judged in {', '.join(names)}")
    for k in names:
        j = J[k]; a = sum(j[i]["has_error"] for i in ids) / len(ids)
        c = sum(j[i]["n_false"] for i in ids) / max(sum(j[i]["n_claims"] for i in ids), 1)
        print(f"{k:22s} answers with an error {a:5.1%}   false claims {c:5.1%}")
    return ids
def pair(x, y):
    if x in J and y in J:
        ids = stats([x, y])
        b = sum(J[x][i]["has_error"] and not J[y][i]["has_error"] for i in ids); w = sum(J[y][i]["has_error"] and not J[x][i]["has_error"] for i in ids)
        print(f"{y} vs {x}: fixed {b}, broken {w}")
pair("TinyLlama norag", "Qwen norag"); pair("Qwen norag", "Qwen rag"); pair("Qwen norag", "Qwen norag t0.3"); pair("Qwen norag t0.3", "Qwen rag t0.3"); pair("Qwen rag", "Qwen rag t0.3")
pair("TinyLlama norag", "TinyLlama norag t0.3"); pair("TinyLlama norag t0.3", "TinyLlama rag t0.3")
pair("TinyLlama norag t0.3", "Qwen norag t0.3"); pair("TinyLlama rag t0.3", "Qwen rag t0.3")
