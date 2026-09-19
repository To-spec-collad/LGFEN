# -*- coding: utf-8 -*-
"""
LGFEN model zoo.

* LGFEN: Log-aware Graph Feature-Enhanced Network
    - DEG: directional edge gating
    - TEB: temporal encoding branch
    - ENA: edge normalization augmentation
    - GF:  gated fusion of edge and node representations
* GNN baselines: GCN, GAT, GraphSAGE
* Non-graph baselines: CNN-1D, BiLSTM, Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


# ---------- 时间正余弦编码 ----------
class SinusoidalTime(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        # Each frequency contributes a sine and cosine term, so the
        # encoding has 2 * dim channels.
        self.dim = dim
        self.fc = nn.Linear(2 * dim, 2 * dim)

    def forward(self, t):
        f = 2 ** torch.arange(self.dim, device=t.device).float() * 0.05
        pe = torch.cat([torch.sin(t.unsqueeze(-1) * f),
                        torch.cos(t.unsqueeze(-1) * f)], -1)
        return self.fc(pe)


# ---------- LGFEN: 方向边门控DEG + 时间分支TEB + 门控融合GF ----------
class LGFEN(nn.Module):
    def __init__(self, in_node=158, in_edge=78, hid=64,
                 use_gate=True, use_time=True, use_ena=True, use_alpha=True):
        super().__init__()
        self.use_gate = use_gate
        self.use_time = use_time
        self.use_ena = use_ena
        self.use_alpha = use_alpha

        self.proj = nn.Linear(in_node, hid)                       # 节点特征投影
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(in_edge + 2 * hid, in_edge), nn.Sigmoid())   # DEG
        if use_time:
            self.time_enc = SinusoidalTime(8)
            self.time_fc = nn.Linear(16, in_edge)                 # TEB
        if use_ena:
            # ENA concatenates directional mean edge features to each node.
            self.ena_proj = nn.Linear(in_node + 2 * in_edge, hid)
        self.conv1 = GCNConv(hid, hid)
        self.conv2 = GCNConv(hid, hid)
        if use_alpha:
            self.alpha = nn.Sequential(
                nn.Linear(in_edge + 2 * hid, 1), nn.Sigmoid())  # scalar GF gate
        # z_e always contains one edge vector and one summed endpoint vector.
        fused_dim = in_edge + hid
        self.head = nn.Sequential(
            nn.Linear(fused_dim + 2 * hid, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        h0 = F.relu(self.proj(x))
        hu, hv = h0[edge_index[0]], h0[edge_index[1]]
        xe = edge_attr
        if self.use_gate:
            g = self.gate(torch.cat([hu, hv, edge_attr], -1))     # 方向感知门控
            xe = edge_attr * (1 + g)                              # 残差门控增强
        if self.use_time and edge_time is not None:
            xe = xe + self.time_fc(self.time_enc(edge_time))      # 时间分支
        if self.use_ena:
            # ENA feeds enhanced edge features back into node representations.
            src, dst = edge_index[0], edge_index[1]
            src_sum = torch.zeros((x.size(0), xe.size(1)), device=xe.device,
                                  dtype=xe.dtype)
            dst_sum = torch.zeros_like(src_sum)
            src_sum.index_add_(0, src, xe)
            dst_sum.index_add_(0, dst, xe)
            src_count = torch.bincount(src, minlength=x.size(0)).clamp_min(1)
            dst_count = torch.bincount(dst, minlength=x.size(0)).clamp_min(1)
            src_mean = src_sum / src_count.to(xe.dtype).unsqueeze(1)
            dst_mean = dst_sum / dst_count.to(xe.dtype).unsqueeze(1)
            h = F.relu(self.ena_proj(torch.cat([x, src_mean, dst_mean], dim=-1)))
        else:
            h = h0
        h1 = F.relu(self.conv1(h, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        hu2, hv2 = h2[edge_index[0]], h2[edge_index[1]]
        if self.use_alpha:
            a = self.alpha(torch.cat([hu2, hv2, xe], -1))         # 门控融合
            ze = torch.cat([a * xe, (1 - a) * (hu2 + hv2)], -1)
        else:
            ze = torch.cat([xe, hu2 + hv2], -1)
        return self.head(torch.cat([ze, hu2, hv2], -1)).squeeze(-1)


# ---------- 图基线（GCN/GAT/GraphSAGE共用骨架） ----------
class GNNBase(nn.Module):
    def __init__(self, conv_cls, in_node=158, in_edge=78, hid=32, heads=1):
        super().__init__()
        self.proj = nn.Linear(in_node, hid)
        if heads > 1:
            # Each GAT layer emits ``out_channels * heads`` when concat=True.
            # Keep the second layer's input equal to the first layer output.
            c1 = conv_cls(hid, hid, heads=heads)
            c2 = conv_cls(hid * heads, hid, heads=heads)
        else:
            c1 = conv_cls(hid, hid)
            c2 = conv_cls(hid, hid)
        self.conv1, self.conv2 = c1, c2
        d = hid * heads
        self.head = nn.Sequential(
            nn.Linear(in_edge + 2 * d, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        h = F.relu(self.proj(x))
        h1 = F.relu(self.conv1(h, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        hu, hv = h2[edge_index[0]], h2[edge_index[1]]
        return self.head(torch.cat([edge_attr, hu, hv], -1)).squeeze(-1)


# ---------- 非图基线 ----------
class CNN1D(nn.Module):
    def __init__(self, in_edge=78):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * 19, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        return self.net(edge_attr.unsqueeze(1)).squeeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, in_edge=78):
        super().__init__()
        self.lstm = nn.LSTM(in_edge, 32, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        out, _ = self.lstm(edge_attr.unsqueeze(1))
        return self.head(out[:, -1]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, in_edge=78):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, in_edge))
        self.enc = nn.TransformerEncoderLayer(
            d_model=in_edge, nhead=6, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(in_edge, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        h = self.enc(edge_attr.unsqueeze(1) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)


MODELS = {
    'LGFEN': lambda: LGFEN(),
    'GCN': lambda: GNNBase(GCNConv),
    'GAT': lambda: GNNBase(GATConv, heads=2),
    'GraphSAGE': lambda: GNNBase(SAGEConv),
    'CNN': lambda: CNN1D(),
    'BiLSTM': lambda: BiLSTM(),
    'Transformer': lambda: TransformerModel(),
}
