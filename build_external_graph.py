"""Build CIC-DDoS2019 Portmap graphs using the CIC-IDS2017 feature schema."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "external/CIC-DDoS2019/Portmap.csv"
OUT = ROOT / "external/CIC-DDoS2019"
info = json.loads((ROOT / "processed/dataset_info.json").read_text(encoding="utf-8"))
features = info["features"]
scaler = np.load(ROOT / "processed/scaler.npz")

aliases = {
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}
missing_directional = {"Fwd RST Flags", "Bwd RST Flags"}

df = pd.read_csv(DATA, low_memory=False)
df.columns = df.columns.str.strip()
resolved = {}
for feature in features:
    source = feature if feature in df.columns else aliases.get(feature)
    if source in df.columns:
        resolved[feature] = source
    elif feature in missing_directional:
        resolved[feature] = None
    else:
        raise KeyError(f"No external column for {feature}")

raw = np.zeros((len(df), len(features)), dtype=np.float32)
for j, feature in enumerate(features):
    if resolved[feature] is not None:
        raw[:, j] = pd.to_numeric(df[resolved[feature]], errors="coerce").fillna(0).to_numpy()
raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
X = ((raw - scaler["mean"]) / np.where(scaler["scale"] == 0, 1, scaler["scale"])).astype(np.float32)
y = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int64).to_numpy()

node_values = pd.concat([df["Source IP"].astype(str), df["Destination IP"].astype(str)], ignore_index=True).unique()
node_id = {value: i for i, value in enumerate(node_values)}
src = df["Source IP"].astype(str).map(node_id).to_numpy(np.int64)
dst = df["Destination IP"].astype(str).map(node_id).to_numpy(np.int64)

src_mean = pd.DataFrame(X).groupby(src).mean()
dst_mean = pd.DataFrame(X).groupby(dst).mean()
node_features = np.zeros((len(node_values), 2 * len(features) + 2), dtype=np.float32)
for node in range(len(node_values)):
    if node in src_mean.index:
        node_features[node, :len(features)] = src_mean.loc[node].to_numpy()
    if node in dst_mean.index:
        node_features[node, len(features):2 * len(features)] = dst_mean.loc[node].to_numpy()
node_features[:, -2] = np.bincount(src, minlength=len(node_values))
node_features[:, -1] = np.bincount(dst, minlength=len(node_values))

timestamps = pd.to_datetime(df["Timestamp"], errors="coerce").astype("int64").to_numpy() / 10**9
timestamps = np.nan_to_num(timestamps, nan=np.nanmedian(timestamps)).astype(np.float32)
order = np.argsort(timestamps)
windows = []
window_size = int(np.ceil(len(df) / 20))
for start in range(0, len(df), window_size):
    idx = order[start:start + window_size]
    nodes = np.unique(np.concatenate([src[idx], dst[idx]]))
    local = {int(value): i for i, value in enumerate(nodes)}
    edge_index = np.array([[local[int(v)] for v in src[idx]],
                           [local[int(v)] for v in dst[idx]]], dtype=np.int64)
    windows.append(Data(
        x=torch.from_numpy(node_features[nodes]).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(X[idx]).float(),
        edge_time=torch.from_numpy(timestamps[idx]).float(),
        y=torch.from_numpy(y[idx]).long(),
    ))

torch.save(windows, OUT / "graph_portmap.pt")
(OUT / "schema_mapping.json").write_text(json.dumps({
    "dataset": "CIC-DDoS2019 Portmap",
    "rows": len(df),
    "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    "missing_directional_features_filled_zero": sorted(missing_directional),
    "aliases": aliases,
}, indent=2), encoding="utf-8")
print(f"saved {len(windows)} windows, {len(df)} flows, nodes={len(node_values)}")
