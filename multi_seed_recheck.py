import json
import time
from pathlib import Path

import numpy as np
import torch

from models import MODELS, LGFEN
from run_experiments import load, set_seed, train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_graphs, val_graphs, test_graphs = load("train"), load("val"), load("test")
primary_seeds = [42, 2026, 7, 123, 999]
extended_seeds = [42, 2026, 7, 123, 999, 11, 22, 33, 44, 55]
results = {"device": str(device), "primary_seeds": primary_seeds, "models": {}}

for name, factory in MODELS.items():
    results["models"][name] = {}
    for seed in primary_seeds:
        set_seed(seed)
        start = time.time()
        metrics = train_one(factory().to(device), train_graphs, val_graphs, test_graphs, seed)
        metrics["elapsed_min"] = round((time.time() - start) / 60, 3)
        results["models"][name][str(seed)] = metrics
        print(name, seed, metrics)

results["models"]["LGFEN_extended"] = {}
for seed in extended_seeds:
    set_seed(seed)
    start = time.time()
    metrics = train_one(LGFEN().to(device), train_graphs, val_graphs, test_graphs, seed)
    metrics["elapsed_min"] = round((time.time() - start) / 60, 3)
    results["models"]["LGFEN_extended"][str(seed)] = metrics
    print("LGFEN_extended", seed, metrics)

for name, values in results["models"].items():
    f1 = np.array([row["f1"] for row in values.values()], dtype=float)
    results.setdefault("summary", {})[name] = {
        "n_seeds": int(len(f1)),
        "f1_mean": float(f1.mean()),
        "f1_sd": float(f1.std(ddof=1)),
        "f1_min": float(f1.min()),
        "f1_max": float(f1.max()),
    }
Path("results/multi_seed_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results["summary"], indent=2))
