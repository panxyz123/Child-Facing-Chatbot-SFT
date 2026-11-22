# Retrieval

Knowledge questions (as decided by the router) are answered with passages from a Wikipedia index in LanceDB.

## Highlights

- **Coverage of child questions nearly doubled** by fixing the corpus, not the retriever: adding Simple English
  Wikipedia raised the share of routed questions with an answering passage from **37.0% to 67.4%** (same 181
  questions).
- **Hit@5 0.92 with hybrid search + cross-encoder rerank**, best of six methods on every metric.
- **Evaluation without ground-truth labels**: pooled LLM relevance judging, checked against a second rater, with
  incremental judging so a new method only pays for passages nobody graded yet.
- **Found and worked around a silent reranker bug** (NaN scores from misaligned weights on macOS CPU) that had made
  "rerank" quietly equal to plain hybrid search.

## Index

| table | contents |
|---|---|
| `wiki_science_v2_fixed` | expert English Wikipedia, science fields (millawell/wikipedia_field_of_science), one lead passage per article |
| `wiki_v3` | the above + Simple English Wikipedia leads (wikimedia/wikipedia 20231101.simple, CC BY-SA 4.0) |
| `wiki_v4_para` | in progress: Simple Wikipedia split into paragraph chunks, for article-weighted retrieval (below) |

Embeddings: all-MiniLM-L6-v2. Full-text index for BM25. `add_simple_wikipedia.py` builds `wiki_v3` (optionally
embedding with PySpark; on one machine it gives no speedup over a single process, it is meant for a cluster).

## Results (181 routed test questions, `wiki_v3`)

`retrieval_benchmark.py`: every method returns its top 10; the top 5 of all methods are pooled and graded once
per question by Gemini (2 = answers the question, 1 = on topic, 0 = not relevant).

| method | Hit@5 (grade >= 1) | Precision@1 (grade >= 1) | Success@5 (grade 2) | MRR (grade 2) | nDCG@5 | p50 latency |
|---|---|---|---|---|---|---|
| vector | 0.912 | 0.779 | 0.525 | 0.424 | 0.628 | 107 ms |
| hybrid (alpha 0.5) | 0.878 | 0.691 | 0.536 | 0.406 | 0.584 | 115 ms |
| **hybrid + rerank** | **0.923** | 0.729 | **0.591** | **0.460** | **0.660** | 526 ms |

BM25 alone and hybrid at alpha 0.3 / 0.7 were also run (BM25: Success@5 0.348).

Hybrid + rerank is significantly better than vector and than hybrid at alpha 0.5 (paired bootstrap). The gain over
plain vector search is small for its 5x latency; most of the remaining misses are questions the corpus does not
cover (coverage 67.4%).

## How it was evaluated

- **Routed questions only**: the 192 knowledge questions of the test split, filtered by the router to the 181 the
  agent would actually retrieve for.
- **Judge reliability**: on 60 blind-labeled passages a second rater agreed with Gemini 76.7% (Cohen's kappa 0.65,
  88.3% on relevant / not relevant). Gemini is generous with grade 2, so grade-2 numbers are upper estimates.
- `--stage review` writes blind review files; `--stage agree` computes the agreement.

## Limitations

- Relevance is judged by an LLM, not by human annotation, and recall is only measurable within the pooled passages.
- The reranker bug workaround (cloning the weights at load) is in `retrieval_benchmark.py` and `sprout/retrieval.py`;
  the first benchmark run (`results/retrieval_v2/runs_nan_rerank.jsonl`) is kept for reference.
- Better retrieval did not translate into fewer hallucinated claims end to end (see `eval/README.md`): on these
  common questions the generator mostly answers from its own knowledge.

## In progress: article-weighted paragraph retrieval

A port of the retrieval design from an earlier project, where it reached 0.9+ precision and recall on a labeled
enterprise documentation benchmark (that number belongs to that benchmark, not to this corpus): paragraph-aware chunking, then recall 20 chunks, cross-encoder scores, article score =
sum of score / rank, keep the top 2 articles within 0.8 of the best, take their top 3 chunks and merge adjacent ones.
Files: `chunking.py`, `build_para_colab.ipynb`, `build_para_table.py`, `article_weighted.py` (with a no-article-step
ablation). Numbers on this corpus will differ: different data, no labels, and judged relevance instead.
