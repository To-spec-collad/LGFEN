import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_rate": float(np.mean(pred)),
    }

out = {}
for name, path in {
    "portmap": "external/CIC-DDoS2019/graph_adapt_test.pt",
    "udplag_balanced": "external/CIC-DDoS2019/graph_udplag_balanced_adapt_test.pt",
}.items():
    graphs = torch.load(path, weights_only=False)
    y = torch.cat([g.y for g in graphs]).numpy()
    out[name] = {
        "n": int(len(y)),
        "labels": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "all_benign": metrics(y, np.zeros_like(y)),
        "all_attack": metrics(y, np.ones_like(y)),
    }
Path("results/external_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
