# Retrieval

Knowledge questions (as decided by the router) are answered with passages from a Wikipedia index in LanceDB.

## Highlights

- **Coverage of child questions nearly doubled** by fixing the corpus, not the retriever: adding Simple English
  Wikipedia raised the share of routed questions with an answering passage from **37.0% to 67.4%** (same 181
  questions).
- **Paragraph-aware chunking raised top-1 answer rate from 28% to 48%** (60 routed questions, 13 wins / 1 loss,
  p = 0.002): the answer is often in the body of an article, not in its first sentences.
- **Hit@5 0.92 with hybrid search + cross-encoder rerank** on article leads, best of six methods on every metric.
- **Evaluation without ground-truth labels**: pooled LLM relevance judging, checked against a second rater, with
  incremental judging so a new method only pays for passages nobody graded yet.
- **Found and worked around a silent reranker bug** (NaN scores from misaligned weights on macOS CPU) that had made
  "rerank" quietly equal to plain hybrid search.

## Index

| table | contents |
|---|---|
| `wiki_science_v2_fixed` | expert English Wikipedia, science fields (millawell/wikipedia_field_of_science), one lead passage per article |
| `wiki_v3` | the above + Simple English Wikipedia leads (wikimedia/wikipedia 20231101.simple, CC BY-SA 4.0) |
| `wiki_v4_para` | Simple Wikipedia split into paragraph chunks (`chunking.py`, 40 to 150 words, one paragraph of overlap) + the expert leads; used by the agent |

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

## Paragraph retrieval (`wiki_v4_para`, 60 routed questions)

`article_weighted.py` ports the retrieval design of an earlier project, where it reached 0.9+ precision and recall
on a labeled enterprise documentation benchmark (that number belongs to that benchmark, not to this corpus):
paragraph-aware chunking, recall 20 chunks, cross-encoder scores, article score = sum of score / rank, keep the top
2 articles within 0.8 of the best, take their top 3 chunks and merge adjacent ones. Each step is an ablation:

| method | top-1 answers | top-3 | top-5 | MRR | p50 latency (CPU) |
|---|---|---|---|---|---|
| hybrid + rerank on article leads (`wiki_v3`) | 28.3% | 41.7% | 46.7% | 0.349 | 0.53 s |
| **paragraph chunks + MiniLM rerank** | **48.3%** | 56.7% | 63.3% | 0.534 | 0.86 s |
| paragraph chunks + bge-reranker-v2-m3 | 48.3% | 61.7% | 65.0% | 0.551 | 6.07 s |
| + article weighting (MiniLM) | 48.3% | 50.0% | 50.0% | 0.492 | 0.86 s |

- Chunking is the gain (13 wins / 1 loss at top-1). The 25x larger reranker ties at top-1 (2 / 2) and helps a little
  at top-3 (3 / 0, p = 0.25) for 7x the CPU latency. Article weighting adds nothing here: Simple Wikipedia articles
  are short, so there is little to filter, unlike long multi-file documentation (its @3/@5 also use at most 3 spans).
- The agent uses paragraph chunks + MiniLM, with a relevance gate at reranker logit 2.0: it keeps 78% of
  retrievals, raises top-1 answer rate among them to 55% (23% among the dropped) and cuts off-topic top-1 passages.

## Limitations

- **Better retrieval did not reduce hallucinated claims end to end** on 30 common-knowledge test questions (Qwen SFT,
  temperature 0.3): 14.9% of claims false without retrieval, 14.0% with the old index, 18.1% with the new one; the
  differences are within noise (5 answers fixed, 6 broken). On the 20 questions that pass the gate the new index is
  lowest (11.5% vs 12.3% without retrieval); the 10 below it get no passages and differ only by sampling.
  Most errors are not in the passages: the model adds its own claims (see `eval/README.md`).
- A title can fool the reranker: "What are electric cars?" retrieved a chunk about the documentary *What Is the
  Electric Car?* at a high score.
- Young children's phrasing is vague: in the paragraph run only 1 of the 3 questions for ages 0-3 had an answering
  passage (a small sample).
- Relevance is judged by an LLM, not by human annotation, and recall is only measurable within the pooled passages.
- The reranker bug workaround (cloning the weights at load) is in `retrieval_benchmark.py` and `sprout/retrieval.py`;
  the first benchmark run (`results/retrieval_v2/runs_nan_rerank.jsonl`) is kept for reference.

## Future directions

- **Hallucination check before answering.** Sample several answers and drop or regenerate claims they disagree on
  (self-consistency, as in SelfCheckGPT); errors at temperature 0.8 were often sampling slips that change between
  samples. Parallel sampling shares the prompt's KV cache in vLLM, so the extra cost is mostly decode, and the
  existing judged claims can measure a detector offline with no new judge calls. Checking answers against the
  passages with an NLI model is the alternative, but most false claims are not in the passages and child-level
  paraphrase is hard for NLI, so it would flag many correct claims.
- **Query rewriting, then retrieval as a tool.** Let the model rewrite the child's words into a search query
  ("why does his balloon went up" -> "why do balloons float"), measurable with the same benchmark. A full tool-calling
  loop where a 1.5B model writes its own queries is less predictable and adds turns of latency.
- **Questions the model cannot answer alone.** Showing a hallucination benefit from retrieval needs long-tail
  questions that the corpus covers and the model does not.
