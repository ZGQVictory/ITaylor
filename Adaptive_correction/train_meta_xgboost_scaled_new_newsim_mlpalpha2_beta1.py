#!/usr/bin/env python3
"""
方案F-scaled-new-newsim-mlpalpha2: alpha 由小型 MLP(z_surf_cal - z_seq_cal, z_seq_cal) 动态生成，
beta 保持标量。alpha_i 无区间限制，per-sample。

g*     = sigmoid(beta * delta_hat_z)
alpha_i = MLP([z_surf - z_seq, z_seq])     # 无界，per-sample，2维输入
z_star = z_seq + g * alpha_i * (z_surf - z_seq)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "calibration_meta_data")
DATA_PATH = os.path.join(DATA_DIR, "meta_aggregated_dataset.csv")


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def ensure_calibrated_logits(df):
    needs_save = False

    if 'seq_cal_logit' not in df.columns or 'surf_cal_logit' not in df.columns:
        seq_params = pd.read_csv(os.path.join(DATA_DIR, "sequence_calibration_params.csv")).iloc[0]
        surf_params = pd.read_csv(os.path.join(DATA_DIR, "surface_calibration_params.csv")).iloc[0]

        if 'seq_cal_logit' not in df.columns:
            df['seq_cal_logit'] = seq_params['platt_A'] * df['sequence_logit'] + seq_params['platt_B']
            needs_save = True
            print("Added seq_cal_logit to meta_aggregated_dataset.csv")

        if 'surf_cal_logit' not in df.columns:
            df['surf_cal_logit'] = surf_params['platt_A'] * df['surface_logit'] + surf_params['platt_B']
            needs_save = True
            print("Added surf_cal_logit to meta_aggregated_dataset.csv")

    if 'seq_cal_prob' not in df.columns:
        df['seq_cal_prob'] = sigmoid_np(df['seq_cal_logit'].values)
        needs_save = True
        print("Added seq_cal_prob to meta_aggregated_dataset.csv")

    if 'surf_cal_prob' not in df.columns:
        df['surf_cal_prob'] = sigmoid_np(df['surf_cal_logit'].values)
        needs_save = True
        print("Added surf_cal_prob to meta_aggregated_dataset.csv")

    if needs_save:
        df.to_csv(DATA_PATH, index=False)
        print(f"Updated → {DATA_PATH}")

    return df


def ensure_signed_product_z(df):
    if 'signed_product_z' in df.columns and df['signed_product_z'].notna().sum() > 0:
        return df

    raw_product = df['seq_cal_logit'].values * df['surf_cal_logit'].values
    signed_product = np.sign(raw_product) * np.log1p(np.abs(raw_product))
    sp_mean = np.nanmean(signed_product)
    sp_std = np.nanstd(signed_product) + 1e-8
    df['signed_product_z'] = (signed_product - sp_mean) / sp_std
    pd.DataFrame([{
        'mean': sp_mean,
        'std': sp_std,
    }]).to_csv(os.path.join(DATA_DIR, "signed_product_zscore_params.csv"), index=False)
    df.to_csv(DATA_PATH, index=False)
    print("Added/updated signed_product_z in meta_aggregated_dataset.csv")
    print("Saved → signed_product_zscore_params.csv")
    return df


df = pd.read_csv(DATA_PATH)
df = ensure_calibrated_logits(df)
df = ensure_signed_product_z(df)

FEATURE_COLS = [
    'seq_cal_prob', 'surf_cal_prob',
    'entropy_seq', 'entropy_surf',
    'phla_dope_per_res',
    'tcr_lDDT', 'tcr_pTM', 'tcr_ipTM',
    'sim_epitope_same_hla_max', 'sim_epitope_other_hla_max', 'sim_epitope_other_hla_top5_mean',
    'sim_cdr3a_same_hla_max', 'sim_cdr3a_other_hla_max', 'sim_cdr3a_other_hla_top5_mean',
    'sim_cdr3b_same_hla_max', 'sim_cdr3b_other_hla_max', 'sim_cdr3b_other_hla_top5_mean',
    'sim_tcra_full_same_hla_max', 'sim_tcra_full_other_hla_max', 'sim_tcra_full_other_hla_top5_mean',
    'sim_tcrb_full_same_hla_max', 'sim_tcrb_full_other_hla_max', 'sim_tcrb_full_other_hla_top5_mean',
    'prob_diff', 'entropy_diff', 'prob_product_log', 'signed_product_z',
]

REQUIRED_COLS = FEATURE_COLS + [
    'label', 'sequence_logit', 'surface_logit',
    'seq_cal_logit', 'surf_cal_logit', 'seq_cal_prob', 'surf_cal_prob',
]

df = df.dropna(subset=REQUIRED_COLS)


def bce_sample(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


y_all    = df['label'].values.astype(np.float32)
bce_seq  = bce_sample(df['seq_cal_prob'].values, y_all)
bce_surf = bce_sample(df['surf_cal_prob'].values, y_all)
delta    = (bce_seq - bce_surf).astype(np.float32)

X          = df[FEATURE_COLS].values.astype(np.float32)
z_seq_cal  = df['seq_cal_logit'].values.astype(np.float32)
z_surf_cal = df['surf_cal_logit'].values.astype(np.float32)

idx_tr, idx_val = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_all
)
X_tr, X_val         = X[idx_tr], X[idx_val]
delta_tr, delta_val = delta[idx_tr], delta[idx_val]
y_tr, y_val         = y_all[idx_tr], y_all[idx_val]
zs_tr, zs_val       = z_seq_cal[idx_tr], z_seq_cal[idx_val]
zf_tr, zf_val       = z_surf_cal[idx_tr], z_surf_cal[idx_val]

# ── Stage 1: XGBoost regression on Delta_i ────────────────────────────────────
print("Training XGBoost regressor on Delta_i...")
xgb_path = os.path.join(DATA_DIR, "xgboost_meta_scaled_new_newsim.json")
model_xgb = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='rmse', early_stopping_rounds=30,
    random_state=42, n_jobs=-1,
)
model_xgb.fit(X_tr, delta_tr, eval_set=[(X_val, delta_val)], verbose=50)
model_xgb.save_model(xgb_path)
print(f"  Saved XGBoost model → {xgb_path}")

delta_hat_tr  = model_xgb.predict(X_tr).astype(np.float32)
delta_hat_val = model_xgb.predict(X_val).astype(np.float32)

# z-score normalize delta_hat so gate input is centered at 0
dh_mean = delta_hat_tr.mean()
dh_std  = delta_hat_tr.std() + 1e-8
delta_hat_tr_z  = (delta_hat_tr  - dh_mean) / dh_std
delta_hat_val_z = (delta_hat_val - dh_mean) / dh_std
pd.DataFrame([{'mean': float(dh_mean), 'std': float(dh_std)}]).to_csv(
    os.path.join(DATA_DIR, "delta_hat_zscore_params_new_newsim.csv"), index=False)
print(f"  delta_hat z-score: mean={dh_mean:.4f}, std={dh_std:.4f}")

# ── Stage 2: 学习 AlphaNet(MLP)，beta 固定为 1.0，输入扩展为 top10 features ───
# 取 XGBoost top10 重要特征
top10_cols = pd.Series(model_xgb.feature_importances_, index=FEATURE_COLS) \
               .nlargest(10).index.tolist()
print(f"\nTop10 features for AlphaNet: {top10_cols}")

# z-score normalize top10 features (fit on train)
X_top10_tr  = X_tr[:, [FEATURE_COLS.index(c) for c in top10_cols]]
X_top10_val = X_val[:, [FEATURE_COLS.index(c) for c in top10_cols]]
feat_mean = X_top10_tr.mean(axis=0)
feat_std  = X_top10_tr.std(axis=0) + 1e-8
X_top10_tr_z  = (X_top10_tr  - feat_mean) / feat_std
X_top10_val_z = (X_top10_val - feat_mean) / feat_std
pd.DataFrame([dict(zip(top10_cols, feat_mean))]).to_csv(
    os.path.join(DATA_DIR, "alpha_top10_feat_mean_beta1.csv"), index=False)
pd.DataFrame([dict(zip(top10_cols, feat_std))]).to_csv(
    os.path.join(DATA_DIR, "alpha_top10_feat_std_beta1.csv"), index=False)
pd.DataFrame({'feature': top10_cols}).to_csv(
    os.path.join(DATA_DIR, "alpha_top10_feat_names_beta1.csv"), index=False)

print("\nLearning AlphaNet(MLP) with beta=1.0 fixed...")

class AlphaNet(nn.Module):
    """输入: [z_surf - z_seq, z_seq, top10_feats] (12维)，输出: per-sample alpha (无界)"""
    def __init__(self, in_dim=12, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.ones_(self.net[2].bias)

    def forward(self, diff, z_seq, feats):
        x = torch.cat([diff.unsqueeze(-1), z_seq.unsqueeze(-1), feats], dim=-1)
        return self.net(x).squeeze(-1)

s_tr_t   = torch.FloatTensor(delta_hat_tr_z)
s_val_t  = torch.FloatTensor(delta_hat_val_z)
zs_tr_t  = torch.FloatTensor(zs_tr)
zs_val_t = torch.FloatTensor(zs_val)
zf_tr_t  = torch.FloatTensor(zf_tr)
zf_val_t = torch.FloatTensor(zf_val)
y_tr_t   = torch.FloatTensor(y_tr)
y_val_t  = torch.FloatTensor(y_val)
feat_tr_t  = torch.FloatTensor(X_top10_tr_z)
feat_val_t = torch.FloatTensor(X_top10_val_z)

diff_tr_t  = zf_tr_t  - zs_tr_t   # z_surf_cal - z_seq_cal
diff_val_t = zf_val_t - zs_val_t

BETA_FIXED = 1.0
alpha_net = AlphaNet(in_dim=12, hidden=32)
optimizer = optim.Adam(alpha_net.parameters(), lr=0.01)
criterion = nn.BCELoss()

best_val_loss = float('inf')
best_state = None
train_losses, val_losses = [], []
alpha_mean_history = []

for epoch in range(1000):
    alpha_net.train()
    optimizer.zero_grad()
    alpha_i = alpha_net(diff_tr_t, zs_tr_t, feat_tr_t)
    g       = torch.sigmoid(torch.tensor(BETA_FIXED) * s_tr_t)
    z_star  = zs_tr_t + g * alpha_i * diff_tr_t
    loss    = criterion(torch.sigmoid(z_star), y_tr_t)
    loss.backward()
    optimizer.step()

    alpha_net.eval()
    with torch.no_grad():
        alpha_i_val = alpha_net(diff_val_t, zs_val_t, feat_val_t)
        g_val       = torch.sigmoid(torch.tensor(BETA_FIXED) * s_val_t)
        z_star_val  = zs_val_t + g_val * alpha_i_val * diff_val_t
        val_loss    = criterion(torch.sigmoid(z_star_val), y_val_t).item()

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    alpha_mean_history.append(alpha_i.detach().mean().item())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in alpha_net.state_dict().items()}

    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1}: train={loss.item():.4f}, val={val_loss:.4f}, "
              f"alpha_mean={alpha_i.detach().mean().item():.4f}")

print(f"\nbeta=1.0 (fixed), best_val_loss={best_val_loss:.4f}")

# ── 评估 ──────────────────────────────────────────────────────────────────────
alpha_net.load_state_dict(best_state)
alpha_net.eval()
with torch.no_grad():
    alpha_i_val_f = alpha_net(diff_val_t, zs_val_t, feat_val_t)
    g_val_f       = torch.sigmoid(torch.tensor(BETA_FIXED) * s_val_t)
    z_star_val_f  = zs_val_t + g_val_f * alpha_i_val_f * diff_val_t
    y_hat_val     = torch.sigmoid(z_star_val_f).numpy()

auc_final   = roc_auc_score(y_val, y_hat_val)
auprc_final = average_precision_score(y_val, y_hat_val)
auc_seq     = roc_auc_score(y_val, torch.sigmoid(zs_val_t).numpy())
auc_surf    = roc_auc_score(y_val, torch.sigmoid(zf_val_t).numpy())
g_mean, g_std = g_val_f.mean().item(), g_val_f.std().item()
alpha_mean_val = alpha_i_val_f.mean().item()
alpha_std_val  = alpha_i_val_f.std().item()

print(f"\nFinal fusion val AUC:   {auc_final:.4f}")
print(f"Final fusion val AUPRC: {auprc_final:.4f}")
print(f"Baseline seq-cal AUC:   {auc_seq:.4f}")
print(f"Baseline surf-cal AUC:  {auc_surf:.4f}")
print(f"g* mean={g_mean:.4f}, std={g_std:.4f}")
print(f"alpha_i mean={alpha_mean_val:.4f}, std={alpha_std_val:.4f}")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 7, figsize=(35, 4))

axes[0].plot(train_losses, label='Train', lw=1.5)
axes[0].plot(val_losses,   label='Val',   lw=1.5)
axes[0].axvline(np.argmin(val_losses), color='red', linestyle='--', lw=1,
                label=f'Best epoch={np.argmin(val_losses)}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].set_title('[mlpalpha2_beta1] Loss Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(alpha_mean_history, color='steelblue', lw=1.5, label='alpha_i mean (train)')
axes[1].axhline(1.0, color='green', linestyle='--', lw=1, label='beta=1.0 (fixed)')
axes[1].set_xlabel('Epoch'); axes[1].set_title('[mlpalpha2_beta1] alpha_i mean')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].hist(g_val_f.numpy(), bins=50, color='steelblue', edgecolor='white')
axes[2].set_xlabel('g*'); axes[2].set_ylabel('Count')
axes[2].set_title(f'[mlpalpha2_beta1] g* dist  mean={g_mean:.3f}, std={g_std:.3f}')
axes[2].grid(alpha=0.3)

axes[3].hist(alpha_i_val_f.numpy(), bins=50, color='darkorange', edgecolor='white')
axes[3].set_xlabel('alpha_i'); axes[3].set_ylabel('Count')
axes[3].set_title(f'[mlpalpha2_beta1] alpha_i dist  mean={alpha_mean_val:.3f}, std={alpha_std_val:.3f}')
axes[3].grid(alpha=0.3)

diff_val_np  = diff_val_t.numpy()
zs_val_np    = zs_val_t.numpy()
g_val_np     = g_val_f.numpy()
alpha_val_np = alpha_i_val_f.numpy()

# gating g* 2D map
sc4 = axes[4].scatter(diff_val_np, zs_val_np, c=g_val_np,
                      cmap='RdBu_r', alpha=0.5, s=8, vmin=0, vmax=1)
plt.colorbar(sc4, ax=axes[4], label='g*')
axes[4].set_xlabel('z_surf - z_seq'); axes[4].set_ylabel('z_seq')
axes[4].set_title(f'[mlpalpha2_beta1] Gating g*  mean={g_mean:.3f}, std={g_std:.3f}')
axes[4].grid(alpha=0.3)

# alpha_i 2D map
sc5 = axes[5].scatter(diff_val_np, zs_val_np, c=alpha_val_np,
                      cmap='RdBu_r', alpha=0.5, s=8)
plt.colorbar(sc5, ax=axes[5], label='alpha_i')
axes[5].set_xlabel('z_surf - z_seq'); axes[5].set_ylabel('z_seq')
axes[5].set_title(f'[mlpalpha2_beta1] AlphaNet alpha_i  mean={alpha_mean_val:.3f}, std={alpha_std_val:.3f}')
axes[5].grid(alpha=0.3)

feat_imp = pd.Series(model_xgb.feature_importances_, index=FEATURE_COLS).sort_values()
axes[6].barh(feat_imp.index, feat_imp.values, color='steelblue')
axes[6].set_title('[mlpalpha2_beta1] XGBoost Feature Importance\n(Delta regression)')
axes[6].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "meta_mlpalpha2_beta1_training_curves.png"), dpi=150)
plt.close()
print("Saved → meta_mlpalpha2_beta1_training_curves.png")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(best_state, os.path.join(DATA_DIR, "alpha_net_mlpalpha2_beta1.pt"))
print("Saved → alpha_net_mlpalpha2_beta1.pt")

pd.DataFrame({
    'idx': idx_val, 'label': y_val,
    'delta_hat': delta_hat_val,
    'g_star': g_val_f.numpy(),
    'alpha_i': alpha_i_val_f.numpy(),
    'z_seq_cal': zs_val,
    'z_surf_cal': zf_val,
    'z_star': z_star_val_f.numpy(),
    'y_hat': y_hat_val,
}).to_csv(os.path.join(DATA_DIR, "meta_mlpalpha2_beta1_val_predictions.csv"), index=False)

pd.DataFrame([{
    'method': 'mlpalpha2_beta1',
    'best_beta': BETA_FIXED,
    'val_auc_fusion': auc_final, 'val_auprc_fusion': auprc_final,
    'val_auc_seq_cal': auc_seq, 'val_auc_surf_cal': auc_surf,
    'g_star_mean': g_mean, 'g_star_std': g_std,
    'alpha_i_mean': alpha_mean_val, 'alpha_i_std': alpha_std_val,
}]).to_csv(os.path.join(DATA_DIR, "meta_mlpalpha2_beta1_params.csv"), index=False)

print(f"Saved to {DATA_DIR}/")
