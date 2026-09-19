# -*- coding: utf-8 -*-
"""
LGFEN - data cleaning and balanced split for CIC-IDS2017.

1. Merge the five daily CSV files of CIC-IDS2017.
2. Deduplicate and sanitise infinite/missing values.
3. Map labels to binary BENIGN / MALICIOUS.
4. Extract the standard 78 CICFlowMeter feature columns.
5. Balance-sample 150,000 records per class.
6. Split by 5-tuple flow key (70/15/15) to prevent leakage.

Outputs (written to `processed/`):
    train.csv / val.csv / test.csv
    dataset_info.json
"""
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = 'hf_cicids'
FILES = ['monday.csv', 'tuesday.csv', 'wednesday.csv', 'thursday.csv', 'friday.csv']
OUT = 'processed'
TARGET_PER_CLASS = 150000

os.makedirs(OUT, exist_ok=True)

# [1] 读取合并
frames = [pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False) for f in FILES]
df = pd.concat(frames, ignore_index=True)
print('[1] merged:', df.shape)

# [2] 去重
df = df.drop_duplicates()
print('[2] after dedup:', len(df))

# [3] 清理无穷值/缺失值
df = df.replace([np.inf, -np.inf], np.nan)
num = df.select_dtypes(include=[np.number]).columns
df[num] = df[num].fillna(0)
print('[3] remaining NaN:', int(df.isna().sum().sum()))

# [4] 标签统一为二分类 BENIGN / MALICIOUS
df['Label'] = df['Label'].astype(str).str.strip().apply(
    lambda x: 'BENIGN' if x.upper() == 'BENIGN' else 'MALICIOUS')
print('[4] label dist:')
print(df['Label'].value_counts())

# [5] 提取标准78个CICFlowMeter特征列（Flow Duration -> Idle Min）
cols = list(df.columns)
i0 = cols.index('Flow Duration')
i1 = cols.index('Idle Min')
feat_cols = cols[i0:i1 + 1]
print('[5] standard features:', len(feat_cols), feat_cols[0], '->', feat_cols[-1])

# [6] 时间戳转数值（图的时间分支用）
df['TimeNum'] = pd.to_datetime(df['Timestamp'], errors='coerce').astype('int64') // 10**9
df['TimeNum'] = df['TimeNum'].fillna(df['TimeNum'].median())

# [7] 流标识（5元组）用于防泄漏划分
flow_cols = ['Src IP dec', 'Src Port', 'Dst IP dec', 'Dst Port', 'Protocol']
df['FlowKey'] = df[flow_cols].astype(str).agg('_'.join, axis=1)
print('[7] unique flows:', df['FlowKey'].nunique())

# [8] 类别平衡子集采样（每类 TARGET_PER_CLASS 条）
parts = []
for lab in ['BENIGN', 'MALICIOUS']:
    sub = df[df['Label'] == lab]
    sub = sub.sample(TARGET_PER_CLASS, random_state=42)
    parts.append(sub)
df = pd.concat(parts, ignore_index=True)
print('[8] subsampled:', df.shape)
print(df['Label'].value_counts())

# [9] 按 FlowKey 防泄漏划分 70/15/15
keys = df['FlowKey'].unique()
rng = np.random.RandomState(42)
rng.shuffle(keys)
n = len(keys)
n_tr = int(n * 0.7)
n_va = int(n * 0.15)
tr_keys = set(keys[:n_tr])
va_keys = set(keys[n_tr:n_tr + n_va])
te_keys = set(keys[n_tr + n_va:])
df_train = df[df['FlowKey'].isin(tr_keys)]
df_val = df[df['FlowKey'].isin(va_keys)]
df_test = df[df['FlowKey'].isin(te_keys)]
print('[9] split:', len(df_train), len(df_val), len(df_test))

# [10] 保存
for name, sub in [('train', df_train), ('val', df_val), ('test', df_test)]:
    sub = sub.drop(columns=['FlowKey'])
    sub.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
info = {
    'features': feat_cols,
    'n_train': int(len(df_train)),
    'n_val': int(len(df_val)),
    'n_test': int(len(df_test)),
    'label_dist': df['Label'].value_counts().to_dict(),
}
with open(os.path.join(OUT, 'dataset_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print('[10] saved -> processed/{train,val,test}.csv + dataset_info.json')
