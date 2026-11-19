# Data pipeline

Instruction-tuning data for an age-aware chatbot for children: each row is a child's message, the child's age
group, and a reply written for that age.

| | v1 (`yxpan/children_sft_dataset`) | v2 (`yxpan/children_sft_dataset_v2`) |
|---|---|---|
| Age label | guessed by Gemini from the child's message | **set** from the source, as a parent or app would set it |
| Age groups | unbalanced (10-13: 3 rows, 4-6: 503 rows) | balanced, ~240 rows per group |
| Knowledge label | none | `needs_knowledge` on every row |
| Split | by row index: 500 / 100 / 600 | stratified by age and `needs_knowledge`: 700 / 100 / 393 |
| Rows | 1,200 | 1,193 |

Age groups: `0-3`, `4-6`, `7-9`, `10-13`, `14-18`.

## v1 (January 2026)

Script: `src/build_sft_dataset_v1.py` (renamed from `scr/train_data_generate_new.py`; content unchanged, kept exactly
as it was run, including the bug below).

1. **Sources**
   - 1,500 real child utterances (`data/sampled_data.csv`, 300 per original label), taken from a Hugging Face
     child-speech dataset. These are short and often ungrammatical ("horsie", "that is got flowers on it"),
     which is how children actually talk to a chatbot.
   - 800 rows sampled from [`nikister/children-assistant`](https://huggingface.co/datasets/nikister/children-assistant):
     12,000 synthetic questions, 2,000 per age group plus 2,000 persona prompts, with the age in the `system` field
     (`Child's age: 4-6`).
2. Both merged (2,300 rows), shuffled, first 1,200 kept (about 65% child speech, 35% HF questions).
3. One `gemini-2.0-flash-lite` call per row: classify the intent, classify the age, write the reply for that age.

**Issues found in a November 2026 audit**

- **Every age was guessed.** A reversed `rename({'age': 'system'})` dropped the HF age column, so the HF rows were
  also sent with no age and Gemini guessed it from the message alone. On the 354 HF rows that can be matched back
  to the source, Gemini's age agreed with the source age **26.3%** of the time (chance with 5 groups: 20%).
- **Imbalance.** 10-13 had 3 rows in total; most rows were 0-6, because most inputs were toddler speech.
- **Mislabeled teens.** 146 of the 206 rows labeled 14-18 were toddler utterances ("black", "are they tightened")
  answered at a teenage level. This matches the SFT result where 14-18 was the only group that got worse.

## v2 (November 2026)

Principle: the age is a condition given to the model, not something to infer from what the child says.
Inputs should still look like what a child of that age would send.

### Build: `src/build_sft_dataset_v2.py`

1. **0-3 and 4-6**: reuse 180 real child-speech rows per group from v1 (only rows v1 labeled 0-3 or 4-6; child
   speech labeled 7+ is dropped), topped up with HF questions for that age.
2. **7-9, 10-13, 14-18**: HF questions for that age. The age is parsed from the HF `system` field; persona-only
   rows (`Character: ...`) carry no age and are dropped.
3. **Reuse**: a v1 HF row is kept only if its v1 (Gemini) age equals the source age (93 rows), so label, reply and
   source agree. Everything else gets a new reply.
4. **Generation**: `gemini-3.5-flash-lite` writes the reply for the given age and labels `intent` and
   `needs_knowledge` (824 calls). Reused rows get `needs_knowledge` in batches of 25.
5. **QA**: inside each age group, replies whose FKGL is an outlier (IQR rule, at least 2 grade levels of margin)
   are dropped; ~10% extra replies are generated to absorb the drops.

The build is resumable: every Gemini reply is cached (`data/gen_cache_v2.jsonl`, `data/knowledge_cache_v2.jsonl`,
both git-ignored), a re-run only calls the API for what is missing, and it stops when the daily quota runs out.
`--dry-run` prints the plan and the number of API calls without calling the API.

`needs_knowledge` is true when a correct answer depends on facts about the world (science, nature, history,
how things work), so a wrong fact would be a hallucination. These are the questions RAG is for.

### Split and upload: `src/push_sft_dataset_v2.py`

- Same question at the same age (repeated child utterances): one row kept (5 rows dropped).
- Same question at different ages (the HF source asks some questions at several ages, 13 groups): all kept, since
  they teach "answer for the age you were given", and each whole group goes to `train`. `validation` and `test`
  contain no question that appears in `train`.
- Stratified by age, and inside each age by `needs_knowledge`: 140 / 20 / rest per age group.

| split | rows | per age group | knowledge questions (0-3 / 4-6 / 7-9 / 10-13 / 14-18) |
|---|---|---|---|
| train | 700 | 140 | 35 / 40 / 78 / 99 / 90 |
| validation | 100 | 20 | 5 / 6 / 11 / 14 / 13 |
| test | 393 | 77-80 | 19 / 21 / 44 / 56 / 52 |

Median reply FKGL (Flesch-Kincaid grade) by age group: 2.4 / 3.0 / 7.0 / 9.1 / 11.2.

### Fields

| field | meaning |
|---|---|
| `id` | row id in `data/children_sft_v2.jsonl` |
| `age` | age group the reply is written for |
| `intent` | `narrative`, `scientific question`, `non-scientific question`, `request storytelling` |
| `needs_knowledge` | the answer depends on factual knowledge (see above) |
| `instruction` / `response` | the child's message / the reply |
| `source` | `local` (real child speech) or `hf` (`nikister/children-assistant`) |
| `age_source` | where the age came from |
| `origin` | `v1` (reply reused from v1) or `gemini v2` (new reply) |
| `fkgl` | FKGL of the reply |

### Known limitations

- **Reply generator differs by age group.** 0-6 replies are mostly from the v1 generator, 7+ mostly from the v2
  generator. On the same kind of input (HF questions) both write at similar levels for 0-3 and 4-6
  (FKGL 2.7 vs 3.0, 3.7 vs 3.5); for older groups the v2 generator writes 2 to 3.5 grade levels higher
  (v1 has only 16-17 comparable rows there).
- **7-9 replies read above a typical 7-9 reading level** (median FKGL ~7). Tone and content fit the age; long
  sentences in stories push FKGL up.
- **`intent` is coarse and tied to age**: `narrative` only occurs in 0-6, and commands ("close it") end up as
  `non-scientific question`. Use `needs_knowledge`, not `intent`, for RAG routing and hallucination evaluation.
- **`needs_knowledge` is an LLM label** and leans towards marking "why" questions as knowledge; spot-check it before
  treating it as ground truth.

### Reproduce

From the repo root, with `GOOGLE_API_KEY` set (or `google_gemini_key.env` next to the repo):

```bash
python src/build_sft_dataset_v2.py --dry-run   # plan and API call count
python src/build_sft_dataset_v2.py             # generate (resumable) + QA -> data/children_sft_v2.jsonl
python src/push_sft_dataset_v2.py              # split, write data/children_sft_v2_splits/ (dry run)
python src/push_sft_dataset_v2.py --push       # upload to the Hugging Face Hub
```

## Files

| path | |
|---|---|
| `data/sampled_data.csv` | 1,500 real child utterances (v1 input) |
| `data/combined_sft_data.jsonl` | v1 dataset |
| `data/children_sft_v2.jsonl` | v2 dataset before splitting |
| `data/children_sft_v2_report.json` | v2 build report: reuse counts, Gemini age accuracy, QA per age group |
| `data/children_sft_v2_splits/` | v2 train / validation / test as uploaded |
