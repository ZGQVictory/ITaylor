#!/usr/bin/env python3
"""
数据准备：合并seq/surf预测，拆分calibration/meta-learning数据集。
所有输出文件保存在 ./calibration_meta_data/ 文件夹中。
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR  = os.path.join(BASE, "fold_calibration_seqonly")
SURF_DIR = os.path.join(BASE, "fold_calibration_surfonly")
OUT_DIR  = os.path.join(BASE, "calibration_meta_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. 合并各折CSV ─────────────────────────────────────────────────────────────

def merge_folds(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "fold_*_val_predictions.csv")))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['fold'] = int(os.path.basename(f).split('_')[1])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


seq_df  = merge_folds(SEQ_DIR)
surf_df = merge_folds(SURF_DIR)

# 去重：同一样本可能因supplement出现在多个fold，保留fold编号最小的（原始预测）
KEY_DEDUP = ['id', 'id_hla', 'id_epitope', 'id_tcr', 'label']
seq_df  = seq_df.sort_values('fold').drop_duplicates(KEY_DEDUP, keep='first').reset_index(drop=True)
surf_df = surf_df.sort_values('fold').drop_duplicates(KEY_DEDUP, keep='first').reset_index(drop=True)
print(f"After dedup — seq: {len(seq_df)}, surf: {len(surf_df)}")

seq_df.to_csv(os.path.join(OUT_DIR, "sequence_val_data.csv"), index=False)
surf_df.to_csv(os.path.join(OUT_DIR, "surface_val_data.csv"), index=False)
print(f"sequence_val_data: {len(seq_df)} rows")
print(f"surface_val_data:  {len(surf_df)} rows")


# ── 2. 取并集，构建 sequence_surface_val_data.csv，并按surf参考文件过滤 ──────────

id_cols = ['id', 'id_hla', 'id_epitope', 'hla_allele', 'Epitope', 'hla_alpha123_mature',
           'tcra_variable', 'tcrb_variable', 'label', 'id_tcr']

# 并集 merge
merged = pd.merge(
    seq_df.rename(columns={'logit': 'sequence_logit', 'prediction': 'sequence_prediction',
                            'fold': 'fold_seq'}),
    surf_df.rename(columns={'logit': 'surface_logit',  'prediction': 'surface_prediction',
                             'fold': 'fold_surf'}),
    on=id_cols,
    how='outer'
)
merged['fold'] = merged['fold_seq'].combine_first(merged['fold_surf']).astype(int)
merged = merged[id_cols + ['fold', 'sequence_logit', 'sequence_prediction',
                            'surface_logit', 'surface_prediction']]

# 构建surf参考集：output_with_pdb_files.csv (Epitope, hla_allele)
#               + negative_phla_pandora_with_ids.csv (peptide→Epitope, hla_allele)
surf_pos = pd.read_csv(os.path.join(SURF_DIR, "output_with_pdb_files.csv"),
                       usecols=['Epitope', 'hla_allele'])
surf_neg = pd.read_csv(os.path.join(SURF_DIR, "negative_phla_pandora_with_ids.csv"),
                       usecols=['peptide', 'hla_allele']).rename(columns={'peptide': 'Epitope'})
surf_ref = pd.concat([surf_pos, surf_neg], ignore_index=True).drop_duplicates()

# 只保留在surf参考集中能找到的 Epitope+hla_allele
merged = merged.merge(surf_ref, on=['Epitope', 'hla_allele'], how='inner')

merged.to_csv(os.path.join(OUT_DIR, "sequence_surface_val_data.csv"), index=False)
print(f"sequence_surface_val_data: {len(merged)} rows")


# ── 3. 拆分 25% calibration / 75% meta-learning ───────────────────────────────

calib_df, meta_df = train_test_split(
    merged, test_size=0.75, random_state=42, stratify=merged['label']
)

calib_df.to_csv(os.path.join(OUT_DIR, "calibration_data.csv"), index=False)
meta_df.to_csv(os.path.join(OUT_DIR, "meta_learning_data.csv"), index=False)
print(f"calibration_data:   {len(calib_df)} rows  (pos={int(calib_df['label'].sum())})")
print(f"meta_learning_data: {len(meta_df)} rows  (pos={int(meta_df['label'].sum())})")


# ── 4. Calibration 工具 ────────────────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return ece


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        return torch.sigmoid(logits / torch.exp(self.log_T))

    def get_T(self):
        return torch.exp(self.log_T).item()


class PlattScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.B     = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        return torch.sigmoid(torch.exp(self.log_A) * logits + self.B)

    def get_params(self):
        return torch.exp(self.log_A).item(), self.B.item()


def fit_calibrator(model, logits, labels, lr=0.01, max_iter=100):
    logits_t = torch.FloatTensor(logits)
    labels_t = torch.FloatTensor(labels)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.BCELoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(model(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return model(logits_t).numpy()


# ── 5. 对 sequence 和 surface 分别 calibration，对比分析并出图 ─────────────────

def compute_metrics(probs, labels):
    ece = compute_ece(probs, labels)
    mce = compute_mce(probs, labels)
    bce = -np.mean(labels * np.log(np.clip(probs, 1e-7, 1-1e-7)) +
                   (1-labels) * np.log(np.clip(1-probs, 1e-7, 1-1e-7)))
    auc   = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return dict(ece=ece, mce=mce, bce=bce, auc=auc, auprc=auprc)


def plot_reliability(ax, probs, labels, title, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    conf_list, acc_list, cnt_list = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() > 0:
            conf_list.append(probs[mask].mean())
            acc_list.append(labels[mask].mean())
            cnt_list.append(mask.sum())
    ax2 = ax.twinx()
    ax2.bar(range(len(cnt_list)), cnt_list, alpha=0.2, color='gray')
    ax2.set_ylabel('Count', fontsize=8)
    ax.plot(conf_list, acc_list, 'o-', lw=2, ms=5)
    ax.plot([0,1],[0,1],'--', color='gray', lw=1.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def compute_mce(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() > 0:
            mce = max(mce, abs(probs[mask].mean() - labels[mask].mean()))
    return mce


all_results = []

for name, logit_col in [("sequence", "sequence_logit"), ("surface", "surface_logit")]:
    valid = calib_df[logit_col].notna()
    logits = calib_df.loc[valid, logit_col].values
    labels = calib_df.loc[valid, 'label'].values
    print(f"[{name}] calibration samples: {valid.sum()} / {len(calib_df)}")

    orig_probs = torch.sigmoid(torch.FloatTensor(logits)).numpy()

    ts       = TemperatureScaling()
    ts_probs = fit_calibrator(ts, logits, labels)
    T        = ts.get_T()

    ps       = PlattScaling()
    ps_probs = fit_calibrator(ps, logits, labels)
    A, B     = ps.get_params()

    m_orig = compute_metrics(orig_probs, labels)
    m_ts   = compute_metrics(ts_probs,   labels)
    m_ps   = compute_metrics(ps_probs,   labels)

    # 保存参数
    pd.DataFrame([{
        'temperature_T': T, 'platt_A': A, 'platt_B': B,
        **{f'orig_{k}': v  for k,v in m_orig.items()},
        **{f'temp_{k}': v  for k,v in m_ts.items()},
        **{f'platt_{k}': v for k,v in m_ps.items()},
    }]).to_csv(os.path.join(OUT_DIR, f"{name}_calibration_params.csv"), index=False)

    # 对比表
    cmp = pd.DataFrame([
        {'method': 'Original',    **m_orig},
        {'method': 'Temperature', **m_ts},
        {'method': 'Platt',       **m_ps},
    ])
    cmp.to_csv(os.path.join(OUT_DIR, f"{name}_calibration_comparison.csv"), index=False)
    all_results.append((name, orig_probs, ts_probs, ps_probs, T, A, B, m_orig, m_ts, m_ps))

    print(f"\n[{name}]")
    print(cmp.to_string(index=False, float_format='%.4f'))

    # ── Reliability diagrams (1行3列) ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_reliability(axes[0], orig_probs, labels,
                     f"Original\nECE={m_orig['ece']:.4f}")
    plot_reliability(axes[1], ts_probs,   labels,
                     f"Temperature (T={T:.4f})\nECE={m_ts['ece']:.4f}")
    plot_reliability(axes[2], ps_probs,   labels,
                     f"Platt (A={A:.4f}, B={B:.4f})\nECE={m_ps['ece']:.4f}")
    fig.suptitle(f"{name.capitalize()} Model — Reliability Diagrams", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_reliability_diagrams.png"), dpi=150)
    plt.close()

    # ── ECE/MCE/BCE 对比柱状图 ─────────────────────────────────────────────────
    metrics_to_plot = ['ece', 'mce', 'bce']
    methods = ['Original', 'Temperature', 'Platt']
    vals = [[m_orig[k], m_ts[k], m_ps[k]] for k in metrics_to_plot]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for ax, metric, v in zip(axes, metrics_to_plot, vals):
        bars = ax.bar(methods, v, color=colors)
        ax.bar_label(bars, fmt='%.4f', fontsize=8)
        ax.set_title(metric.upper(), fontsize=10)
        ax.set_ylim(0, max(v) * 1.3)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f"{name.capitalize()} Model — Calibration Metrics Comparison",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_metrics_comparison.png"), dpi=150)
    plt.close()

# ── 跨模型对比图（sequence vs surface，按方法） ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics_to_plot = ['ece', 'mce', 'bce']
x = np.arange(3)  # Original / Temperature / Platt
width = 0.35

for ax, metric in zip(axes, metrics_to_plot):
    seq_vals  = [all_results[0][7][metric], all_results[0][8][metric], all_results[0][9][metric]]
    surf_vals = [all_results[1][7][metric], all_results[1][8][metric], all_results[1][9][metric]]
    b1 = ax.bar(x - width/2, seq_vals,  width, label='Sequence', color='#4C72B0')
    b2 = ax.bar(x + width/2, surf_vals, width, label='Surface',  color='#DD8452')
    ax.bar_label(b1, fmt='%.3f', fontsize=7)
    ax.bar_label(b2, fmt='%.3f', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(['Original','Temperature','Platt'], fontsize=8)
    ax.set_title(metric.upper()); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(seq_vals + surf_vals) * 1.35)

fig.suptitle("Sequence vs Surface — Calibration Comparison", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "seq_vs_surf_calibration_comparison.png"), dpi=150)
plt.close()

print(f"\n所有文件已保存至: {OUT_DIR}")
