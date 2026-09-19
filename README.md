# LGFEN

Lightweight Graph Feature Enhancement Network for malicious traffic detection on CIC-IDS2017.

LGFEN models network flows as a graph (IP addresses as nodes, flows as edges) and learns a
per-edge malicious/benign prediction with a lightweight message-passing backbone. It combines:

- **DEG** - directional edge gating: direction-aware gating of edge features from both endpoint embeddings;
- **TEB** - temporal encoding branch: sinusoidal time encoding added to edge features;
- **ENA** - edge normalization augmentation: layer normalization on augmented edge features;
- **GF** - gated fusion: data-driven fusion of edge and node representations before the classifier head.

## News / Status

- PeerJ manuscript under review (manuscript ID available after submission).

## Dataset

CIC-IDS2017 (Intrusion Detection Evaluation Dataset), five daily PCAP-derived CSV files
(flow-level features, 78 CICFlowMeter features per flow).

Raw file SHA-256:

| file       | SHA-256                                                          |
|------------|------------------------------------------------------------------|
| monday.csv | c493314d7f2cd5614193d3fd8de6d0731edfe1383ccf24df0323c8c6899ba7ab |
| tuesday.csv| 664b615572fdbd88ae24d01f33a538983277359151e9fd5172d40da092362f71 |
| wednesday.csv| 87c6ab27cef32df2dcac315ecd992f918484e71856f32b0d2a55eb171a774d60 |
| thursday.csv| e3cac669f495df33da7ae1e8fcc06ed511d3161b97a7ba9263f2d476cf467a0e |
| friday.csv | e16fa2655766ed685fe3e43455d7e4024a81f8a1965c1eabf56e53dc9f07fb6b |

Download into `hf_cicids/` (mirror):

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='bvk/CICIDS-2017', repo_type='dataset', local_dir='hf_cicids')"
```

## Pipeline

```bash
# 1. clean + balanced split (150k/class, 5-tuple leakage-free 70/15/15)
python clean_data.py            # -> processed/{train,val,test}.csv + dataset_info.json

# 2. build graphs (IP nodes, flow edges, 20 time windows, scaler on train only)
python build_graph.py           # -> processed/graph_{train,val,test}.pt + scaler.npz

# 3. experiments
python run_experiments.py               # 7 models x 3 seeds -> results.json
python run_experiments.py --ablation    # LGFEN variants      -> ablation_results.json
python run_experiments.py --complexity  # params + inference  -> complexity.json
python run_experiments.py --final       # final model + CM    -> confusion_final.json/png
```

## Main results (3-seed mean F1 on test, 44,757 flows)

| Model       | F1     | Acc    | Params | Infer (ms) |
|-------------|--------|--------|--------|-----------|
| LGFEN       | 0.9932 | 0.9932 | 95,152 | 1.68      |
| GCN         | 0.9738 | 0.9740 | 16,417 | 1.04      |
| GAT         | 0.9738 | 0.9738 | 24,929 | 1.32      |
| GraphSAGE   | 0.9663 | 0.9662 | 18,465 | 0.62      |
| CNN-1D      | 0.9922 | 0.9922 | 40,673 | 0.44      |
| BiLSTM      | 0.9930 | 0.9930 | 30,785 | 0.29      |
| Transformer | 0.9976 | 0.9976 | 50,333 | 0.34      |

Ablation (ΔF1 vs. full, 3-seed mean): w/o DEG −1.09, w/o TEB 0.00, w/o ENA −0.69,
w/o GF −0.57, +Focal loss −0.61 (paper default is BCE).

Final model (seed 42, BCE): confusion matrix [[TN,FP],[FN,TP]] =
[[22,363, 47], [182, 22,165]], Acc 0.9949, Prec 0.9979, Rec 0.9919, F1 0.9949.

All numbers above are the as-reported experimental results archived in `results/`.
Re-running the scripts may produce slightly different figures depending on the
software/hardware environment (experiments were run on an NVIDIA A10 GPU with
PyTorch 2.7 / torch-geometric).

## Environment

- NVIDIA A10 (24 GB), CUDA, PyTorch 2.7, torch-geometric (Linux)
- Training time ~2-4 minutes per model (40 epochs, early stopping, 20 subgraphs)

## License

MIT (see LICENSE).
