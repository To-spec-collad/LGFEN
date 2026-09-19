"""Create leakage-free CIC-DDoS2019 adaptation splits and graphs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "external/CIC-DDoS2019"
data_file = sys.argv[1] if len(sys.argv) > 1 else "Portmap.csv"
prefix = sys.argv[2] if len(sys.argv) > 2 else "portmap"
df = pd.read_csv(OUT / data_file, low_memory=False)
df.columns = df.columns.str.strip()
info = json.loads((ROOT / "processed/dataset_info.json").read_text(encoding="utf-8"))
features = info["features"]
aliases = {
    "Total Fwd Packet": "Total Fwd Packets", "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length", "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count", "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size", "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk", "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk", "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate", "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward", "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}
raw = np.zeros((len(df), len(features)), dtype=np.float32)
zero_filled = []
for j, feature in enumerate(features):
    source = feature if feature in df.columns else aliases.get(feature)
    if source in df.columns:
        raw[:, j] = pd.to_numeric(df[source], errors="coerce").fillna(0).to_numpy()
    else:
        zero_filled.append(feature)
raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
y = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int64).to_numpy()
src_values = df["Source IP"].astype(str).to_numpy()
dst_values = df["Destination IP"].astype(str).to_numpy()
flow_key = (src_values + "|" + df["Source Port"].astype(str) + "|" + dst_values + "|" +
            df["Destination Port"].astype(str) + "|" + df["Protocol"].astype(str)).to_numpy()
times = pd.to_datetime(df["Timestamp"], errors="coerce").astype("int64").to_numpy() / 10**9
times = np.nan_to_num(times, nan=np.nanmedian(times)).astype(np.float32)

keys = np.unique(flow_key)
rng = np.random.RandomState(42)
rng.shuffle(keys)
n_train = int(len(keys) * 0.70)
n_val = int(len(keys) * 0.15)
key_sets = {
    "train": set(keys[:n_train]),
    "val": set(keys[n_train:n_train + n_val]),
    "test": set(keys[n_train + n_val:]),
}
indices = {name: np.flatnonzero(np.isin(flow_key, list(values))) for name, values in key_sets.items()}
scaler = StandardScaler().fit(raw[indices["train"]])

def make_graphs(idx):
    X = scaler.transform(raw[idx]).astype(np.float32)
    local_src_values = src_values[idx]
    local_dst_values = dst_values[idx]
    nodes = np.unique(np.concatenate([local_src_values, local_dst_values]))
    node_id = {value: i for i, value in enumerate(nodes)}
    src = np.array([node_id[v] for v in local_src_values], dtype=np.int64)
    dst = np.array([node_id[v] for v in local_dst_values], dtype=np.int64)
    node_features = np.zeros((len(nodes), 2 * len(features) + 2), dtype=np.float32)
    src_sum = np.zeros((len(nodes), len(features)), dtype=np.float64)
    dst_sum = np.zeros_like(src_sum)
    np.add.at(src_sum, src, X)
    np.add.at(dst_sum, dst, X)
    src_count = np.bincount(src, minlength=len(nodes)).clip(min=1)[:, None]
    dst_count = np.bincount(dst, minlength=len(nodes)).clip(min=1)[:, None]
    node_features[:, :len(features)] = (src_sum / src_count).astype(np.float32)
    node_features[:, len(features):2 * len(features)] = (dst_sum / dst_count).astype(np.float32)
    node_features[:, -2] = np.bincount(src, minlength=len(nodes))
    node_features[:, -1] = np.bincount(dst, minlength=len(nodes))
    local_times = times[idx]
    order = np.argsort(local_times)
    size = int(np.ceil(len(idx) / 20))
    graphs = []
    for start in range(0, len(idx), size):
        w = order[start:start + size]
        win_nodes = np.unique(np.concatenate([src[w], dst[w]]))
        local = {int(v): i for i, v in enumerate(win_nodes)}
        edge_index = np.array([[local[int(v)] for v in src[w]],
                               [local[int(v)] for v in dst[w]]], dtype=np.int64)
        graphs.append(Data(
            x=torch.from_numpy(node_features[win_nodes]).float(),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(X[w]).float(),
            edge_time=torch.from_numpy(local_times[w]).float(),
            y=torch.from_numpy(y[idx[w]]).long(),
        ))
    return graphs

for split in ("train", "val", "test"):
    graphs = make_graphs(indices[split])
    torch.save(graphs, OUT / f"graph_{prefix}_adapt_{split}.pt")
    print(split, len(indices[split]), len(graphs))

(OUT / f"adaptation_{prefix}_split.json").write_text(json.dumps({
    "split": {k: int(len(v)) for k, v in indices.items()},
    "label_counts": {k: {str(c): int((y[v] == c).sum()) for c in (0, 1)} for k, v in indices.items()},
    "zero_filled_features": zero_filled,
    "scaler_fit_on": "CIC-DDoS2019 train split only",
}, indent=2), encoding="utf-8")
