import json
from pathlib import Path

import numpy as np
from scipy import stats

data = json.loads(Path("results/multi_seed_results.json").read_text(encoding="utf-8"))
models = data["models"]
summary = data["summary"]
seeds = data["primary_seeds"]
lgfen = np.array([models["LGFEN"][str(s)]["f1"] for s in seeds])
comparisons = {}
for name in ["GCN", "GAT", "GraphSAGE", "CNN", "BiLSTM", "Transformer"]:
    other = np.array([models[name][str(s)]["f1"] for s in seeds])
    t = stats.ttest_rel(lgfen, other)
    comparisons[name] = {
        "mean_difference": float((lgfen - other).mean()),
        "paired_t_p": float(t.pvalue),
    }
output = {"primary_seeds": seeds, "summary": summary, "paired_comparisons": comparisons}
Path("results/multi_seed_statistics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(comparisons, indent=2))
