"""Significance of GRPO's age-sensitivity gain, from the age_sweep.json files the GRPO notebooks write
(llm_rl_runs/<exp>/age_sweep.json on Drive: for each of 40 hold-out inputs, the answer at each of 5 set ages).

    python scripts/age_sweep_stats.py ~/Downloads/grpo_qwen_rag/age_sweep.json ~/Downloads/grpo_v2data_dense/age_sweep.json

Per input: Spearman between set age and the answer's FKGL. Reports the mean, a paired bootstrap CI and Wilcoxon test
on the difference, and the share of inputs with rho > 0.5 with an exact test on the inputs that changed.
"""
import json, sys
import numpy as np
import textstat
from scipy.stats import binomtest, spearmanr, wilcoxon

AGES = 5
fk = lambda t: float(textstat.flesch_kincaid_grade(t)) if t.strip() else np.nan


def rhos(texts):
    n = len(texts) // AGES
    return np.array([spearmanr(range(AGES), [fk(t) for t in texts[i * AGES:(i + 1) * AGES]]).correlation
                     for i in range(n)])


for path in sys.argv[1:]:
    S = json.load(open(path))
    a, b = rhos(S["sft"]), rhos(S["sft+grpo"])
    ok = ~np.isnan(a) & ~np.isnan(b); a, b = a[ok], b[ok]; d = b - a
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    up, down = int(((a <= .5) & (b > .5)).sum()), int(((a > .5) & (b <= .5)).sum())
    print(f"{path}\n  n = {ok.sum()} inputs | mean Spearman {a.mean():.3f} -> {b.mean():.3f}, diff {d.mean():+.3f} "
          f"(95% CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]), Wilcoxon p = "
          f"{wilcoxon(d).pvalue:.3f}\n  share with rho > 0.5: {np.mean(a > .5):.1%} -> {np.mean(b > .5):.1%} "
          f"(gained {up}, lost {down}, exact p = {binomtest(up, up + down).pvalue if up + down else 1.0:.3f})")
