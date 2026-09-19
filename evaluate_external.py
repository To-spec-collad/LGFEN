import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models import LGFEN
from run_experiments import load, predict, set_seed, train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
model = LGFEN().to(device)
train_graphs = load("train")
val_graphs = load("val")
test_graphs = load("test")
train_one(model, train_graphs, val_graphs, test_graphs, seed=42)
external = torch.load("external/CIC-DDoS2019/graph_portmap.pt", weights_only=False)
logits, labels = predict(model, external)
pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
y = labels.cpu().numpy()
metrics = {
    "dataset": "CIC-DDoS2019 Portmap",
    "training": "CIC-IDS2017 seed 42; no external fine-tuning",
    "n_flows": int(len(y)),
    "accuracy": float(accuracy_score(y, pred)),
    "precision": float(precision_score(y, pred, zero_division=0)),
    "recall": float(recall_score(y, pred, zero_division=0)),
    "f1": float(f1_score(y, pred, zero_division=0)),
    "predicted_positive_rate": float(pred.mean()),
}
Path("results/external_portmap_zero_shot.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, indent=2))
