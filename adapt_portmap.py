import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models import LGFEN
from run_experiments import load, predict, set_seed, train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
model = LGFEN().to(device)
train_one(model, load("train"), load("val"), load("test"), seed=42)

prefix = sys.argv[1] if len(sys.argv) > 1 else "portmap"
adapt_train = torch.load(f"external/CIC-DDoS2019/graph_{prefix}_adapt_train.pt", weights_only=False)
adapt_val = torch.load(f"external/CIC-DDoS2019/graph_{prefix}_adapt_val.pt", weights_only=False)
adapt_test = torch.load(f"external/CIC-DDoS2019/graph_{prefix}_adapt_test.pt", weights_only=False)

all_y = torch.cat([g.y for g in adapt_train])
counts = torch.bincount(all_y, minlength=2).float().clamp_min(1)
class_weight = (counts.sum() / (2 * counts)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
best_state = None
best_f1 = -1.0
bad = 0
for epoch in range(20):
    model.train()
    for graph in adapt_train:
        graph = graph.to(device)
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        loss = F.cross_entropy(torch.stack([-logits, logits], dim=1), graph.y,
                               weight=class_weight)
        loss.backward()
        optimizer.step()
    val_logits, val_y = predict(model, adapt_val)
    val_pred = (torch.sigmoid(val_logits) >= 0.5).long().cpu().numpy()
    val_true = val_y.cpu().numpy()
    val_f1 = f1_score(val_true, val_pred, zero_division=0)
    print(f"epoch={epoch} loss={loss.item():.5f} val_f1={val_f1:.5f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
    else:
        bad += 1
        if bad >= 5:
            break

model.load_state_dict(best_state)
logits, labels = predict(model, adapt_test)
pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
y = labels.cpu().numpy()
metrics = {
    "dataset": f"CIC-DDoS2019 {prefix}",
    "training": "CIC-IDS2017 pretraining + labeled CIC-DDoS2019 train adaptation",
    "n_test_flows": int(len(y)),
    "accuracy": float(accuracy_score(y, pred)),
    "precision": float(precision_score(y, pred, zero_division=0)),
    "recall": float(recall_score(y, pred, zero_division=0)),
    "f1": float(f1_score(y, pred, zero_division=0)),
    "predicted_positive_rate": float(pred.mean()),
}
Path(f"results/external_{prefix}_adapted.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, indent=2))
