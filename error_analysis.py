import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from models import LGFEN
from run_experiments import load, predict, set_seed, train_one

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, val, test = load("train"), load("val"), load("test")
model = LGFEN().to(device)
train_one(model, train, val, test, seed=42)
rows = []
all_pred, all_y = [], []
with torch.inference_mode():
    for index, graph in enumerate(test):
        graph = graph.to(device)
        logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        y = graph.y.cpu().numpy()
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        rows.append({"window": index, "n": int(len(y)), "f1": float(f1_score(y, pred, zero_division=0)), "cm": cm})
        all_pred.append(pred)
        all_y.append(y)
all_pred = np.concatenate(all_pred)
all_y = np.concatenate(all_y)
output = {
    "seed": 42,
    "overall_cm": confusion_matrix(all_y, all_pred, labels=[0, 1]).tolist(),
    "worst_windows": sorted(rows, key=lambda row: row["f1"])[:5],
    "best_windows": sorted(rows, key=lambda row: row["f1"], reverse=True)[:5],
}
Path("results/error_analysis_seed42.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
