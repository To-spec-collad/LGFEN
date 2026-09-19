# LGFEN Reproducibility Package

This package contains the corrected implementation and rechecked results for the PeerJ Computer Science manuscript.

## Main protocol

- CIC-IDS2017: 150,000 benign and 150,000 malicious flows, sampled with `random_state=42`.
- Five-tuple group split: 70% train, 15% validation, 15% test.
- LGFEN main comparison: seeds `42, 2026, 7, 123, 999`.
- Extended LGFEN stability run: ten seeds, including `11, 22, 33, 44, 55`.
- Ablation: three seeds, explicitly reported separately from the five-seed main comparison.

## Commands

```powershell
python clean_data.py
python build_graph.py
python run_experiments.py
python run_experiments.py --ablation
python multi_seed_recheck.py
python multi_seed_statistics.py
python error_analysis.py
python edge_benchmark.py --warmup 20 --runs 100
```

The raw datasets are not redistributed. Download instructions and SHA-256 checksums are in `README.md`. The external CIC-DDoS2019 scripts document feature harmonization, five-tuple splits, target-only scaling, zero-shot evaluation, and supervised target-domain adaptation.

## Interpretation

LGFEN improves ordinary graph baselines on CIC-IDS2017 but does not outperform CNN, BiLSTM, or Transformer. Direct zero-shot transfer to CIC-DDoS2019 fails under domain shift; labeled target-domain adaptation recovers performance on Portmap and UDP-lag scenarios. No embedded edge-hardware measurement is claimed; `edge_benchmark.py` is provided for future Raspberry Pi/Jetson testing.
