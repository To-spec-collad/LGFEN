# -*- coding: utf-8 -*-
"""
LGFEN - training, evaluation, ablation, complexity and final confusion matrix.

Usage:
    python run_experiments.py             # main experiments (7 models x 3 seeds)
    python run_experiments.py --ablation  # ablation study (LGFEN variants)
    python run_experiments.py --complexity
    python run_experiments.py --final     # final model (seed 42) + confusion matrix

Outputs are written to `processed/`:
    results.json / ablation_results.json / complexity.json /
    confusion_final.json / confusion_final.png
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

from models import MODELS, LGFEN

OUT = 'processed'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 2026, 7]
EPOCHS = 40
LR = 1e-3
PATIENCE = 8


def load(split):
    """Load the time-window graph list for a split."""
    return torch.load(f'{OUT}/graph_{split}.pt', weights_only=False, map_location=DEVICE)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def focal_loss(logit, y, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy(p, y, reduction='none')
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * ce).mean()


@torch.no_grad()
def predict(model, graphs):
    logits, ys = [], []
    model.eval()
    for g in graphs:
        g = g.to(DEVICE)
        logits.append(model(g.x, g.edge_index, g.edge_attr, g.edge_time))
        ys.append(g.y)
    return torch.cat(logits), torch.cat(ys)


def metrics(model, graphs):
    logits, y = predict(model, graphs)
    pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    y = y.cpu().numpy()
    return {
        'acc': float((pred == y).mean()),
        'prec': float(precision_score(y, pred, zero_division=0)),
        'rec': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
    }


def train_one(model, tr, va, te, seed, use_focal=False):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        tl = 0.0
        for g in tr:
            g = g.to(DEVICE)
            opt.zero_grad()
            logit = model(g.x, g.edge_index, g.edge_attr, g.edge_time)
            loss = focal_loss(logit, g.y.float()) if use_focal \
                else F.binary_cross_entropy_with_logits(logit, g.y.float())
            loss.backward()
            opt.step()
            tl += loss.item()
        m = metrics(model, va)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 5 == 0:
            print(f'   ep{ep} loss={tl:.3f} valF1={m["f1"]:.4f}')
    model.load_state_dict(best_state)
    return metrics(model, te)


def main_experiments():
    tr, va, te = load('train'), load('val'), load('test')
    per_seed, means = {}, {}
    for name, fn in MODELS.items():
        per_seed[name] = {}
        print(f'== {name} ==')
        for s in SEEDS:
            t0 = time.time()
            set_seed(s)
            m = train_one(fn().to(DEVICE), tr, va, te, s)
            per_seed[name][str(s)] = {**m, 'time_min': round((time.time() - t0) / 60, 2)}
            print(f'  seed{s}: acc={m["acc"]:.4f} prec={m["prec"]:.4f} '
                  f'rec={m["rec"]:.4f} f1={m["f1"]:.4f} '
                  f'({per_seed[name][str(s)]["time_min"]}min)')
        means[name] = {k: float(np.mean([per_seed[name][str(s)][k] for s in SEEDS]))
                       for k in ['acc', 'prec', 'rec', 'f1']}
        print(f'{name}: acc={means[name]["acc"]:.4f} prec={means[name]["prec"]:.4f} '
              f'rec={means[name]["rec"]:.4f} f1={means[name]["f1"]:.4f}')
    json.dump({'per_seed': per_seed, 'mean': means},
              open(f'{OUT}/results.json', 'w'), indent=2)
    print('saved -> processed/results.json')


def ablation():
    tr, va, te = load('train'), load('val'), load('test')
    variants = {
        'LGFEN(full)': lambda: LGFEN(),
        'w/o DEG': lambda: LGFEN(use_gate=False),
        'w/o TEB': lambda: LGFEN(use_time=False),
        'w/o ENA': lambda: LGFEN(use_ena=False),
        'w/o GF': lambda: LGFEN(use_alpha=False),
    }
    per_seed, means = {}, {}
    for name, fn in variants.items():
        per_seed[name] = {}
        print(f'== {name} ==')
        for s in SEEDS:
            set_seed(s)
            m = train_one(fn().to(DEVICE), tr, va, te, s)
            per_seed[name][str(s)] = m
            print(f'  seed{s}: f1={m["f1"]:.4f}')
        means[name] = {k: float(np.mean([per_seed[name][str(s)][k] for s in SEEDS]))
                       for k in ['acc', 'prec', 'rec', 'f1']}
        print(f'{name}: acc={means[name]["acc"]:.4f} prec={means[name]["prec"]:.4f} '
              f'rec={means[name]["rec"]:.4f} f1={means[name]["f1"]:.4f}')
    # +Focal loss variant (paper default is BCE)
    name = '+Focal loss'
    per_seed[name] = {}
    print(f'== {name} ==')
    for s in SEEDS:
        set_seed(s)
        m = train_one(LGFEN().to(DEVICE), tr, va, te, s, use_focal=True)
        per_seed[name][str(s)] = m
        print(f'  seed{s}: f1={m["f1"]:.4f}')
    means[name] = {k: float(np.mean([per_seed[name][str(s)][k] for s in SEEDS]))
                   for k in ['acc', 'prec', 'rec', 'f1']}
    print(f'{name}: acc={means[name]["acc"]:.4f} prec={means[name]["prec"]:.4f} '
          f'rec={means[name]["rec"]:.4f} f1={means[name]["f1"]:.4f}')
    json.dump({'per_seed': per_seed, 'mean': means},
              open(f'{OUT}/ablation_results.json', 'w'), indent=2)
    print('saved -> ablation_results.json')


def complexity():
    tr = load('train')
    g = tr[0].to(DEVICE)
    res = {}
    for name, fn in MODELS.items():
        m = fn().to(DEVICE).eval()
        n_params = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                _ = m(g.x, g.edge_index, g.edge_attr, g.edge_time)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.time() - t0) / 20 * 1000
        res[name] = {'params': int(n_params), 'infer_ms': round(infer_ms, 3)}
        print(f'[complexity] {name}: params={n_params} infer={infer_ms:.3f}ms')
    json.dump(res, open(f'{OUT}/complexity.json', 'w'), indent=2)
    print('saved -> complexity.json')


def final_run():
    tr, va, te = load('train'), load('val'), load('test')
    set_seed(42)
    model = LGFEN().to(DEVICE)
    m = train_one(model, tr, va, te, 42)  # BCE, seed 42
    print('FINAL seed42:', m)
    logits, y = predict(model, te)
    pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    y = y.cpu().numpy()
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    cm = [[tn, fp], [fn, tp]]
    json.dump({'cm': cm, 'labels': ['BENIGN', 'MALICIOUS']},
              open(f'{OUT}/confusion_final.json', 'w'), indent=2)
    print('cm [[TN,FP],[FN,TP]]:', cm)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1], ['BENIGN', 'MALICIOUS'])
        ax.set_yticks([0, 1], ['BENIGN', 'MALICIOUS'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha='center', va='center',
                        color='white' if cm[i][j] > max(cm[i]) / 2 else 'black')
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(f'{OUT}/confusion_final.png', dpi=300)
        print('saved confusion_final.png')
    except Exception as e:
        print('matplotlib unavailable, skipped png:', e)
    print('saved -> confusion_final.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--complexity', action='store_true')
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    print('Device:', DEVICE)
    if args.ablation:
        ablation()
    elif args.complexity:
        complexity()
    elif args.final:
        final_run()
    else:
        main_experiments()
