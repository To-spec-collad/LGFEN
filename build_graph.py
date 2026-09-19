# -*- coding: utf-8 -*-
"""
LGFEN - graph construction.

Builds one heterogeneous flow graph per time window:
    * nodes = IP addresses, node features = 158 dims
      (78 dims aggregated from outgoing flows + 78 dims from incoming flows
       + out-degree + in-degree)
    * edges = flows, edge features = 78 dims (CICFlowMeter) + timestamp

The standard scaler is fitted on the TRAIN split only (leakage-free), saved as
`processed/scaler.npz`.

Outputs:
    processed/graph_{train,val,test}.pt   (list of torch_geometric Data)
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

OUT = 'processed'
WINDOWS = 20

info = json.load(open(f'{OUT}/dataset_info.json'))
feat_cols = info['features']

# [1] 标准化（仅 train 拟合，防泄漏）
tr = pd.read_csv(f'{OUT}/train.csv', low_memory=False)
scaler = StandardScaler().fit(tr[feat_cols].values)
np.savez(f'{OUT}/scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
print('[1] scaler fitted on train, dim=', len(feat_cols))


def build_graph(df):
    X = scaler.transform(df[feat_cols].values).astype(np.float32)
    y = (df['Label'].values == 'MALICIOUS').astype(np.int64)
    src = df['Src IP dec'].values.astype(np.int64)
    dst = df['Dst IP dec'].values.astype(np.int64)
    t = df['TimeNum'].values.astype(np.float32)

    # 节点映射
    nodes = np.unique(np.concatenate([src, dst]))
    nid = {int(v): i for i, v in enumerate(nodes)}
    u = np.array([nid[int(v)] for v in src], dtype=np.int64)
    v = np.array([nid[int(w)] for w in dst], dtype=np.int64)

    # 节点特征: 方向聚合均值(78+78) + 出度/入度 = 158维
    gdf = df[['Src IP dec', 'Dst IP dec']].copy()
    gdf[feat_cols] = X
    ms = gdf.groupby('Src IP dec')[feat_cols].mean()
    md = gdf.groupby('Dst IP dec')[feat_cols].mean()
    ds = gdf.groupby('Src IP dec').size()
    dd = gdf.groupby('Dst IP dec').size()
    feats = []
    for ip in nodes:
        a = ms.loc[int(ip)].values if int(ip) in ms.index else np.zeros(len(feat_cols))
        b = md.loc[int(ip)].values if int(ip) in md.index else np.zeros(len(feat_cols))
        feats.append(np.concatenate([a, b, [ds.get(int(ip), 0)], [dd.get(int(ip), 0)]]))
    node_feat = np.stack(feats).astype(np.float32)

    # 时间窗口切分子图
    order = np.argsort(t)
    n = len(t)
    win = int(np.ceil(n / WINDOWS))
    graphs = []
    for w in range(WINDOWS):
        idx = order[w * win:(w + 1) * win]
        if len(idx) == 0:
            continue
        uw, vw, Xw, yw, tw = u[idx], v[idx], X[idx], y[idx], t[idx]
        nw = np.unique(np.concatenate([uw, vw]))
        nidw = {int(x): i for i, x in enumerate(nw)}
        u2 = np.array([nidw[int(x)] for x in uw])
        v2 = np.array([nidw[int(x)] for x in vw])
        # PyG expects tensor-backed fields; keeping NumPy arrays here makes
        # the first model forward fail before training starts.
        g = Data(
            edge_index=torch.from_numpy(np.stack([u2, v2])).long(),
            edge_attr=torch.from_numpy(Xw).float(),
            y=torch.from_numpy(yw).long(),
            x=torch.from_numpy(node_feat[[nidw[int(x)] for x in nw]]).float(),
            edge_time=torch.from_numpy(tw).float(),
        )
        graphs.append(g)
    return graphs, len(nodes), n


for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'{OUT}/{split}.csv', low_memory=False)
    graphs, n_nodes, n_edges = build_graph(df)
    torch.save(graphs, f'{OUT}/graph_{split}.pt')
    print(f'[{split}] graphs={len(graphs)}, nodes={n_nodes}, edges={n_edges}')
    print(f'   node_feat_dim={graphs[0].x.shape[1]}, '
          f'edge_feat_dim={graphs[0].edge_attr.shape[1]}')

print('DONE -> processed/graph_{train,val,test}.pt')
