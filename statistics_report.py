"""Seed-level uncertainty and paired comparisons for the rechecked results."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "processed" / "results.json").read_text(encoding="utf-8"))
per_seed = data["per_seed"]
seeds = sorted(next(iter(per_seed.values())).keys())

summary = {}
for model, values in per_seed.items():
    f1 = np.array([values[s]["f1"] for s in seeds], dtype=float)
    mean = float(f1.mean())
    sd = float(f1.std(ddof=1))
    ci = stats.t.interval(0.95, len(f1) - 1, loc=mean,
                          scale=stats.sem(f1))
    summary[model] = {
        "seeds": seeds,
        "f1_mean": mean,
        "f1_sd": sd,
        "f1_ci95_t": [max(0.0, float(ci[0])), min(1.0, float(ci[1]))],
    }

lgfen = np.array([per_seed["LGFEN"][s]["f1"] for s in seeds])
comparisons = {}
for model in per_seed:
    if model == "LGFEN":
        continue
    other = np.array([per_seed[model][s]["f1"] for s in seeds])
    diff = lgfen - other
    ttest = stats.ttest_rel(lgfen, other)
    try:
        wilcoxon = stats.wilcoxon(lgfen, other, alternative="two-sided")
        w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    except ValueError:
        w_stat, w_p = None, None
    comparisons[model] = {
        "paired_difference_lgfen_minus_model_mean": float(diff.mean()),
        "paired_difference_sd": float(diff.std(ddof=1)),
        "paired_t_stat": float(ttest.statistic),
        "paired_t_p": float(ttest.pvalue),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }

output = {
    "note": "Exploratory seed-level statistics; n=3 paired seeds, no claim of high statistical power.",
    "summary": summary,
    "paired_comparisons": comparisons,
}
(ROOT / "results" / "statistics.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)
for model, row in summary.items():
    lo, hi = row["f1_ci95_t"]
    print(f"{model}: F1={row['f1_mean']:.4f} +/- {row['f1_sd']:.4f}; 95% CI [{lo:.4f}, {hi:.4f}]")
for model, row in comparisons.items():
    print(f"LGFEN - {model}: diff={row['paired_difference_lgfen_minus_model_mean']:.4f}, "
          f"paired t p={row['paired_t_p']:.4f}, Wilcoxon p={row['wilcoxon_p']}")
