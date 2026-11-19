"""Which readability signal should a reward use? Compare candidate signals on the v2 reference answers before
spending another RL run on them.

Signals (all computed per text):
  fkgl          Flesch-Kincaid grade (current reward): syllables per word and words per sentence
  dale_chall    Dale-Chall score: share of words outside a list of words familiar to 4th graders
  coleman_liau  characters per word and sentences per 100 words (no syllable counting)
  aoa_mean      mean age of acquisition of the content words (Kuperman et al. 2012 ratings; independent of this
                dataset and of Gemini)
  aoa_p90       90th percentile of the same: driven by the hardest words in the answer
  zipf_mean     mean word frequency (wordfreq, Zipf scale; lower = rarer words), sign flipped so higher = harder
  combo         mean of the z-scored fkgl, aoa_mean and zipf_mean

Checks:
  1. discrimination  Spearman between the signal and the age group (0-3 ... 14-18) over all reference answers,
                     and the mean AUROC between adjacent age groups. Two caveats: references are Gemini-written, so
                     this measures agreement with the dataset's notion of age, not ground truth; and the v2
                     pipeline dropped references with outlier FKGL for their age group, which inflates FKGL here.
  2. reliability     split-half: sentences of each answer alternately into two halves, correlation of the signal
                     between halves across answers, Spearman-Brown corrected. Independent of any age label: it
                     measures how noisy the signal is on short texts, which is what makes a reward noisy.
  3. length confound Spearman between the signal and the answer's word count within each age group
                     (a good signal should not just measure length).
  4. model answers   if a predictions file is given: Spearman between the signal on the model's answers and the set age.

AoA ratings: download the Kuperman et al. (2012) ratings (xlsx) from http://crr.ugent.be/archives/806 and pass
the file with --aoa.

    pip install textstat wordfreq openpyxl pandas scipy scikit-learn
    python src/validate_readability_signals.py --aoa path/to/AoA_ratings.xlsx
    python src/validate_readability_signals.py --aoa ... --preds path/to/E_alpha64/holdout_preds.jsonl
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

AGES = ["0-3", "4-6", "7-9", "10-13", "14-18"]
SPLITS_DIR = "data/children_sft_v2_splits"
WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def load_aoa(path):
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    word_col = next(c for c in df.columns if c.lower() in ("word", "words"))
    aoa_col = next(c for c in df.columns if c in ("Rating.Mean", "AoA_Kup_lem", "AoA_Kup", "aoa_mean") or "Rating.Mean" in c)
    df = df[[word_col, aoa_col]].dropna()
    return dict(zip(df[word_col].astype(str).str.lower(), pd.to_numeric(df[aoa_col], errors="coerce")))


def lookup(word, aoa):
    """Exact match, else a crude lemma (plural, past tense, -ing); the ratings are for lemmas."""
    if word in aoa:
        return aoa[word]
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e")):
        if word.endswith(suf) and len(word) > len(suf) + 2:
            w = word[: -len(suf)] + rep
            if w in aoa:
                return aoa[w]
    return None


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if WORD.search(s)]


def signals(text, aoa):
    import textstat
    from wordfreq import zipf_frequency
    words = [w.lower() for w in WORD.findall(text)]
    out = dict(words=len(words))
    if not words:
        return out
    out["fkgl"] = textstat.flesch_kincaid_grade(text)
    out["dale_chall"] = textstat.dale_chall_readability_score(text)
    out["coleman_liau"] = textstat.coleman_liau_index(text)
    ages = [a for a in (lookup(w, aoa) for w in words) if a is not None and not np.isnan(a)]
    if ages:
        out["aoa_mean"], out["aoa_p90"] = float(np.mean(ages)), float(np.percentile(ages, 90))
    out["aoa_coverage"] = len(ages) / len(words)
    out["zipf_mean"] = -float(np.mean([zipf_frequency(w, "en") for w in words]))
    return out


SIGNALS = ["fkgl", "dale_chall", "coleman_liau", "aoa_mean", "aoa_p90", "zipf_mean", "combo"]


def add_combo(df, stats=None):
    """combo = mean z-score of fkgl, aoa_mean and zipf_mean; z-score statistics come from the reference answers."""
    cols = ["fkgl", "aoa_mean", "zipf_mean"]
    stats = stats or {c: (df[c].mean(), df[c].std()) for c in cols}
    df["combo"] = np.nanmean(np.column_stack([(df[c] - stats[c][0]) / stats[c][1] for c in cols]), axis=1)
    return stats


def split_half(texts, aoa, min_sentences=2):
    """Per signal: correlation between the two halves of each text (alternate sentences), Spearman-Brown corrected."""
    a, b = [], []
    for t in texts:
        s = sentences(t)
        if len(s) < min_sentences:
            continue
        a.append(signals(" ".join(s[0::2]), aoa))
        b.append(signals(" ".join(s[1::2]), aoa))
    A, B = pd.DataFrame(a), pd.DataFrame(b)
    if len(A) < 10:
        return {}, len(A)
    stats = add_combo(A); add_combo(B, stats)
    rel = {}
    for c in SIGNALS:
        ok = A[c].notna() & B[c].notna()
        r = np.corrcoef(A[c][ok], B[c][ok])[0, 1] if ok.sum() > 10 else np.nan
        rel[c] = 2 * r / (1 + r) if r > -1 else np.nan
    return rel, len(A)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoa", required=True, help="Kuperman et al. (2012) AoA ratings, xlsx or csv")
    ap.add_argument("--splits", default=SPLITS_DIR)
    ap.add_argument("--preds", default=None, help="optional model predictions (jsonl with age, prediction)")
    ap.add_argument("--out", default="results/reward_signals")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    aoa = load_aoa(a.aoa)
    print(f"AoA ratings for {len(aoa)} words")

    rows = []
    for split in ["train", "validation", "test"]:
        for l in open(os.path.join(a.splits, f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            rows.append(dict(split=split, age=r["age"], text=r["response"], **signals(r["response"], aoa)))
    D = pd.DataFrame(rows)
    D["age_idx"] = D.age.map(AGES.index)
    stats = add_combo(D)
    print(f"{len(D)} reference answers; mean AoA coverage {D.aoa_coverage.mean():.0%} of words")

    summary = {}
    for c in SIGNALS:
        ok = D[c].notna()
        rho = spearmanr(D.age_idx[ok], D[c][ok]).correlation
        aucs = []
        for lo, hi in zip(AGES, AGES[1:]):
            m = ok & D.age.isin([lo, hi])
            aucs.append(roc_auc_score(D.age[m] == hi, D[c][m]))
        length = np.nanmean([spearmanr(D.words[(D.age == g) & ok], D[c][(D.age == g) & ok]).correlation for g in AGES])
        summary[c] = dict(spearman_age=rho, adjacent_auroc=float(np.mean(aucs)), length_corr_within_age=length)

    rel, n_rel = split_half(D.text, aoa)
    rel_young, n_young = split_half(D.text[D.age.isin(["0-3", "4-6"])], aoa)
    for c in SIGNALS:
        summary[c]["split_half_all"] = rel.get(c, np.nan)
        summary[c]["split_half_0_6"] = rel_young.get(c, np.nan)

    if a.preds:
        P = pd.DataFrame([dict(age=r["age"], **signals(r["prediction"], aoa)) for r in map(json.loads, open(a.preds, encoding="utf-8"))])
        add_combo(P, stats)
        for c in SIGNALS:
            ok = P[c].notna()
            summary[c]["spearman_age_model_answers"] = spearmanr(P.age[ok].map(AGES.index), P[c][ok]).correlation

    S = pd.DataFrame(summary).T
    pd.set_option("display.width", 160)
    print(f"\nSignal comparison (split-half on {n_rel} answers with 2+ sentences, {n_young} of them ages 0-6)")
    print("higher is better: spearman_age, adjacent_auroc, split_half_*;  closer to 0 is better: length_corr_within_age")
    print(S.round(3).to_string())
    medians = D.groupby("age")[SIGNALS].median().reindex(AGES)
    print("\nMedian per age group (reference answers)")
    print(medians.round(2).to_string())
    S.to_csv(os.path.join(a.out, "signal_comparison.csv"))
    medians.to_csv(os.path.join(a.out, "signal_medians_by_age.csv"))
    D.drop(columns=["text"]).to_csv(os.path.join(a.out, "reference_signals.csv"), index=False)
    print("\nsaved to", a.out)


if __name__ == "__main__":
    main()
