"""Add Simple English Wikipedia to the retrieval index, as a NEW table (the existing one is left untouched).

Why: the first retrieval benchmark found that most child questions that the router sends to retrieval have no
passage that answers them. The index (wikipedia_field_of_science) holds expert, long-tail English articles; what is
missing are the introductory articles on everyday concepts (thunder, comets, rainbows, daylight saving time...).
Simple English Wikipedia is written for beginners and covers science, health, history and society.

Source:  wikimedia/wikipedia, config 20231101.simple (CC BY-SA 4.0; credit Wikipedia in the README)
Result:  table wiki_v3 = copy of wiki_science_v2_fixed + the Simple English articles
         (same columns: id, topic, search_text "topic: first sentences", vector; ids start with "simple-")

    python rag/add_simple_wikipedia.py --limit 500        # try it on 500 articles first
    python rag/add_simple_wikipedia.py                    # the full run (embedding is the slow part)
"""
import argparse
import os
import re
import shutil
import sys
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # the model the index was built with
SENTENCES = 5
MIN_CHARS = 200
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def lead(text, n=SENTENCES):
    """First n sentences of the article body (the dataset has no title line), whitespace normalised."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return " ".join(SENT.split(text)[:n])


def _load_model(device="cpu", threads=None):
    import torch
    from sentence_transformers import SentenceTransformer
    if threads:
        torch.set_num_threads(threads)
    model = SentenceTransformer(EMBED_MODEL, device=device)
    for p in model.parameters():            # transformers 5.23 + torch 2.15 on macOS CPU: clone, or misaligned
        p.data = p.data.clone()             # float32 weights give wrong results in linear layers
    return model


_WORKER_MODEL = {}


def _embed_partitions(batches, batch_size=256):
    """Runs inside a Spark Python worker: one model per worker process, one thread each, so N workers use N cores
    without oversubscribing them. Input and output are pandas DataFrames (id, search_text) -> (id, vector)."""
    import pandas as pd
    if "m" not in _WORKER_MODEL:
        _WORKER_MODEL["m"] = _load_model("cpu", threads=1)
    for pdf in batches:
        v = _WORKER_MODEL["m"].encode(pdf["search_text"].tolist(), batch_size=batch_size, convert_to_numpy=True)
        yield pd.DataFrame({"id": pdf["id"].values, "vector": [x.astype("float32").tolist() for x in v]})


def embed_single(texts, device, batch):
    return _load_model(device).encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)


def get_spark(workers, batch):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master(f"local[{workers}]").appName("sprout-embed")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(batch * 4))
             .config("spark.driver.memory", "6g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.range(workers).count()            # start the JVM and the executors now, so start-up is timed apart
    return spark


def embed_spark(rows, workers, batch, spark=None):
    """Encode with PySpark: the passages are split into `workers` partitions (one model load per partition) and
    encoded by that many parallel Python workers (local[N] on one machine; on a cluster change master and N)."""
    import pandas as pd
    own = spark is None
    spark = spark or get_spark(workers, batch)
    df = spark.createDataFrame(pd.DataFrame([{"id": r["id"], "search_text": r["search_text"]} for r in rows]))
    out = df.repartition(workers).mapInPandas(lambda it: _embed_partitions(it, batch), schema="id string, vector array<float>")
    res = out.toPandas().set_index("id")
    if own:
        spark.stop()
    return np.array([res.loc[r["id"], "vector"] for r in rows], dtype="float32")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="rag/lancedb_data")
    ap.add_argument("--src", default="wiki_science_v2_fixed")
    ap.add_argument("--dst", default="wiki_v3")
    ap.add_argument("--limit", type=int, default=None, help="only the first N articles, to test the pipeline")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    ap.add_argument("--spark", type=int, default=0, help="encode with PySpark on this many local workers (0 = off)")
    ap.add_argument("--bench", type=int, default=0,
                    help="encode only the first N passages once with and once without Spark, print the speedup, stop")
    a = ap.parse_args(argv)

    import lancedb
    from datasets import load_dataset

    src_dir, dst_dir = (os.path.join(a.db, f"{n}.lance") for n in (a.src, a.dst))
    if os.path.exists(dst_dir):
        sys.exit(f"{dst_dir} exists already: delete it by hand to start again (this script never deletes)")
    db = lancedb.connect(a.db)
    schema = db.open_table(a.src).schema
    names = set(schema.names)
    if not names <= {"id", "topic", "search_text", "vector", "category", "url"}:
        sys.exit(f"unexpected columns in {a.src}: {sorted(names)}; adapt the rows below to this schema first")
    import pyarrow as pa
    for c in names & {"category", "url", "id", "topic", "search_text"}:
        if not (pa.types.is_string(schema.field(c).type) or pa.types.is_large_string(schema.field(c).type)):
            sys.exit(f"column {c} is {schema.field(c).type}, not a string: adapt the rows below to this schema first")
    have = set(db.open_table(a.src).to_pandas()["topic"].str.lower())
    print(f"{a.src}: {len(have)} topics")

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    rows = []
    for r in ds:
        title = r["title"].strip()
        body = lead(r["text"])
        if title.lower() in have or len(body) < MIN_CHARS or title.lower().startswith(("list of", "category:")):
            continue
        row = dict(id=f"simple-{r['id']}", topic=title, search_text=f"{title}: {body}")
        if "category" in names:
            row["category"] = "simple_wikipedia"        # the field-of-science label does not exist for these
        if "url" in names:
            row["url"] = r.get("url") or ""
        rows.append(row)
        if a.limit and len(rows) >= a.limit:
            break
    print(f"{len(rows)} Simple English articles to add")

    if a.bench:
        n = min(a.bench, len(rows)); texts = [r["search_text"] for r in rows[:n]]
        workers = a.spark or max(2, (os.cpu_count() or 4) // 2)
        t0 = time.time(); v1 = embed_single(texts, a.device, a.batch); t_single = time.time() - t0
        t0 = time.time(); spark = get_spark(workers, a.batch); t_start = time.time() - t0
        t0 = time.time(); v2 = embed_spark(rows[:n], workers, a.batch, spark); t_spark = time.time() - t0
        print(f"{n} passages | single process (all cores, torch threads): {n / t_single:.0f} passages/s | "
              f"Spark x{workers}: {n / t_spark:.0f} passages/s excluding {t_start:.0f}s start-up "
              f"({n / (t_spark + t_start):.0f} including) | speedup {t_single / t_spark:.2f}x excluding start-up, "
              f"{t_single / (t_spark + t_start):.2f}x including | max abs diff {np.abs(v1 - v2).max():.2e}")
        return
    t0 = time.time()
    vecs = embed_spark(rows, a.spark, a.batch) if a.spark else embed_single([r["search_text"] for r in rows], a.device, a.batch)
    if not np.isfinite(vecs).all():
        sys.exit("non-finite embeddings: not writing anything")
    for r, v in zip(rows, vecs):
        r["vector"] = v.astype("float32").tolist()
    print(f"embedded in {time.time() - t0:.0f}s")

    shutil.copytree(src_dir, dst_dir)       # the original table stays as it is
    tbl = db.open_table(a.dst)
    tbl.add(rows)
    tbl.create_fts_index("search_text", replace=True)
    print(f"{a.dst}: {tbl.count_rows()} rows. Next: python rag/retrieval_benchmark.py --stage retrieve "
          f"--table {a.dst} --out results/retrieval_v3")


if __name__ == "__main__":
    main()
