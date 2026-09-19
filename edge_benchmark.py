"""Benchmark LGFEN on the current device or a target edge device.

Usage on a Raspberry Pi/Jetson after copying the project and processed graph:
    python edge_benchmark.py --warmup 20 --runs 100
"""
import argparse
import json
import platform
import time
from pathlib import Path

import torch

from models import LGFEN

parser = argparse.ArgumentParser()
parser.add_argument("--graph", default="processed/graph_test.pt")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--runs", type=int, default=100)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load(args.graph, weights_only=False, map_location=device)
graph = graphs[0].to(device)
model = LGFEN().to(device).eval()

with torch.inference_mode():
    for _ in range(args.warmup):
        model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.runs):
        model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
    if device.type == "cuda":
        torch.cuda.synchronize()

result = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "torch": torch.__version__,
    "device": str(device),
    "graph": args.graph,
    "edges_per_window": int(graph.edge_attr.shape[0]),
    "runs": args.runs,
    "warmup": args.warmup,
    "mean_ms": (time.perf_counter() - start) / args.runs * 1000,
}
Path("results/edge_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
