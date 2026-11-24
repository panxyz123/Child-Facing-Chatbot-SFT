"""Compare load-test runs of scripts/bench_openai.py: one markdown table per metric (rows = load level, columns =
runs), throughput ratios against a baseline run, sanity checks on output length, and optional plots.

    python scripts/analyze_serving.py --runs ollama_stop,llamacpp_stop,llamacpp_fixed --baseline ollama_stop
    python scripts/analyze_serving.py --runs ... --plot          # -> results/serving/plots/*.png
    python scripts/analyze_serving.py --runs ... --md results/serving/local_frameworks.md

Read the checks first: throughput in tokens/s is comparable across runs only if the output lengths are similar
(mean_out_tokens, share_at_max_tokens); TTFT and TPOT are per request and less sensitive to it.
"""
import argparse, json, os

METRICS = [("out_tok_per_s", "output tokens/s (all requests)", "{:.0f}"), ("req_per_s", "requests/s", "{:.2f}"),
           ("ttft_p50", "TTFT p50 (s)", "{:.3f}"), ("ttft_p95", "TTFT p95 (s)", "{:.3f}"),
           ("tpot_p50", "TPOT p50 (s/token)", "{:.4f}"), ("e2e_p95", "E2E p95 (s)", "{:.2f}"),
           ("slo_attainment", "SLO met", "{:.0%}"), ("mean_out_tokens", "mean output tokens", "{:.0f}"),
           ("share_at_max_tokens", "share at max_tokens", "{:.0%}")]


def load(tag, d):
    s = json.load(open(os.path.join(d, f"{tag}.json")))
    return s, {float(l["level"]): l for l in s["levels"]}


def fmt(v, f):
    return "" if v is None else f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of tags in results/serving")
    ap.add_argument("--baseline", default=None, help="tag to compute throughput ratios against")
    ap.add_argument("--dir", default="results/serving")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--md", default=None, help="also write the tables to this markdown file")
    a = ap.parse_args()
    tags = a.runs.split(",")
    runs = {t: load(t, a.dir) for t in tags}
    levels = sorted({lv for _, by in runs.values() for lv in by})
    out = []
    p = lambda line="": (print(line), out.append(line))

    p("## Setup")
    for t, (s, _) in runs.items():
        ar = s.get("args", {})
        p(f"- `{t}`: {s.get('base_url')}, mode {s.get('mode')}, max_tokens {ar.get('max_tokens')}, "
          f"ignore_eos {ar.get('ignore_eos', False)}, stop {ar.get('stop')}, requests/level {ar.get('requests')}"
          + (f". Note: {s['note'][:160]}..." if s.get("note") else ""))
    for key, name, f in METRICS:
        p(f"\n## {name}\n")
        p("| " + ("concurrency" if runs[tags[0]][0].get("mode") != "open" else "req/s offered") + " | "
          + " | ".join(tags) + " |")
        p("|---" * (len(tags) + 1) + "|")
        for lv in levels:
            p(f"| {lv:g} | " + " | ".join(fmt(runs[t][1].get(lv, {}).get(key), f) for t in tags) + " |")
    if a.baseline:
        p(f"\n## Throughput relative to `{a.baseline}` (output tokens/s)\n")
        p("| level | " + " | ".join(t for t in tags if t != a.baseline) + " |")
        p("|---" * len(tags) + "|")
        base = runs[a.baseline][1]
        for lv in levels:
            b = base.get(lv, {}).get("out_tok_per_s")
            p(f"| {lv:g} | " + " | ".join(
                (f"{runs[t][1][lv]['out_tok_per_s'] / b:.1f}x" if b and lv in runs[t][1] else "")
                for t in tags if t != a.baseline) + " |")
    p("\n## Checks")
    for t, (s, by) in runs.items():
        ar = s.get("args", {}); m = [l.get("mean_out_tokens") for l in by.values() if l.get("mean_out_tokens")]
        shares = [l.get("share_at_max_tokens") for l in by.values() if l.get("share_at_max_tokens") is not None]
        if ar.get("ignore_eos") and shares and min(shares) < 0.95:
            p(f"- `{t}`: ignore_eos was requested but only {min(shares):.0%} of answers reached max_tokens: "
              "the server ignores the flag, so this run has natural lengths, not fixed ones")
        if shares and max(shares) > 0.2 and not ar.get("ignore_eos"):
            p(f"- `{t}`: up to {max(shares):.0%} of answers hit max_tokens: answers are cut, check stop strings "
              "and sampling (e.g. repetition penalty) against the other servers")
        if m:
            p(f"- `{t}`: mean output tokens {min(m):.0f} to {max(m):.0f}")
    if a.md:
        open(a.md, "w").write("\n".join(out) + "\n"); print("wrote", a.md)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(os.path.join(a.dir, "plots"), exist_ok=True)
        for key, name, _ in METRICS[:6]:
            plt.figure(figsize=(5, 3.4))
            for t in tags:
                xs = [lv for lv in levels if lv in runs[t][1] and runs[t][1][lv].get(key) is not None]
                plt.plot(xs, [runs[t][1][lv][key] for lv in xs], "o-", label=t)
            plt.xlabel("concurrency"); plt.ylabel(name); plt.xscale("log", base=2); plt.grid(alpha=0.3)
            if key.startswith(("ttft", "e2e")):
                plt.yscale("log")
            plt.legend(fontsize=8); plt.tight_layout()
            f = os.path.join(a.dir, "plots", f"{key}.png"); plt.savefig(f, dpi=150); plt.close()
        print("plots in", os.path.join(a.dir, "plots"))


if __name__ == "__main__":
    main()
