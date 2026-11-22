"""Build LanceDB table wiki_v4_para from the Colab output (rag/build_para_colab.ipynb), as a NEW table.

  rows = Simple Wikipedia paragraph chunks (only articles that are in wiki_v3, i.e. not duplicating an expert title)
       + the expert (non-Simple) lead rows of wiki_v3, each as a one-chunk article (article_id = its id)
  columns: id, article_id, topic, section, para_idx, search_text, vector, category, url
  indexes: full-text on search_text; scalar on article_id (fetch an article's chunks by key)

    python rag/build_para_table.py --src ~/Downloads/wiki_para        # sprout-index env
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder with para_chunks.parquet and para_vectors_f16.npy")
ap.add_argument("--db", default="rag/lancedb_data")
ap.add_argument("--base", default="wiki_v3")
ap.add_argument("--dst", default="wiki_v4_para")
a = ap.parse_args()

import lancedb
db = lancedb.connect(a.db)
if a.dst in db.table_names():
    raise SystemExit(f"{a.dst} exists; pick another --dst (tables are never overwritten)")
base = db.open_table(a.base).to_pandas()
simple_ids = set(base.id[base.id.str.startswith("simple-")])
expert = base[~base.id.str.startswith("simple-")].copy()

para = pd.read_parquet(os.path.join(a.src, "para_chunks.parquet"))
vec = np.load(os.path.join(a.src, "para_vectors_f16.npy")).astype("float32")
assert len(para) == len(vec) and np.isfinite(vec).all()
keep = para.article_id.isin(simple_ids).values
para, vec = para[keep].reset_index(drop=True), vec[keep]
para["vector"] = list(vec)
para["category"] = "simple_wikipedia"
print(f"simple: {para.article_id.nunique()} of {len(simple_ids)} wiki_v3 articles, {len(para)} chunks")

expert["article_id"] = expert.id.astype(str); expert["section"] = ""; expert["para_idx"] = 0
for c in ("category", "url"):
    if c not in expert:
        expert[c] = ""
cols = ["id", "article_id", "topic", "section", "para_idx", "search_text", "vector", "category", "url"]
expert = expert[cols]; expert["id"] = expert.id.astype(str)
expert["vector"] = [np.asarray(v, dtype="float32") for v in expert.vector]
df = pd.concat([para[cols], expert], ignore_index=True)
for c in ("id", "article_id", "topic", "section", "search_text", "category", "url"):
    df[c] = df[c].fillna("").astype(str)
df["para_idx"] = df.para_idx.astype("int32")
print(f"total {len(df)} rows ({len(expert)} expert leads)")

t = db.create_table(a.dst, data=df)
t.create_fts_index("search_text", replace=True)
t.create_scalar_index("article_id")
print("built", a.dst, t.count_rows(), "rows")
