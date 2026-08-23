#!/usr/bin/env python3
"""
构建 meta_aggregated_dataset.csv
特征：calibrated scores, entropy, structure reliability, sequence familiarity, conflict
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os, re, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import pairwise2
from Bio.Align import substitution_matrices

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "calibration_meta_data")
SURF_DIR = os.path.join(BASE, "fold_calibration_surfonly")
SEQ_DIR  = os.path.join(BASE, "fold_calibration_seqonly")
DB_DIR   = os.path.join(BASE, "../data/Database_stage1")

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# ── 读取 calibration 参数 ──────────────────────────────────────────────────────
def load_platt(name):
    p = pd.read_csv(os.path.join(DATA_DIR, f"{name}_calibration_params.csv")).iloc[0]
    return float(p['platt_A']), float(p['platt_B'])

seq_A, seq_B   = load_platt("sequence")
surf_A, surf_B = load_platt("surface")

# ── 读取 meta_learning_data ────────────────────────────────────────────────────
meta = pd.read_csv(os.path.join(DATA_DIR, "meta_learning_data.csv"))

# ── 1. Calibrated scores & entropy ────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-x))
def entropy(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(p * np.log(p) + (1-p) * np.log(1-p))

meta['seq_cal_logit']  = seq_A  * meta['sequence_logit'] + seq_B
meta['surf_cal_logit'] = surf_A * meta['surface_logit']  + surf_B
meta['seq_cal_prob']   = sigmoid(meta['seq_cal_logit'].values)
meta['surf_cal_prob']  = sigmoid(meta['surf_cal_logit'].values)
meta['entropy_seq']    = entropy(meta['seq_cal_prob'].values)
meta['entropy_surf']   = entropy(meta['surf_cal_prob'].values)

# ── 2. pHLA structure reliability (DOPE per residue) ─────────────────────────
def parse_dope(pdb_path):
    dope = None
    residues = set()
    with open(pdb_path) as f:
        for line in f:
            if 'DOPE score:' in line:
                dope = float(line.split()[-1])
            if line.startswith('ATOM'):
                residues.add((line[21], line[22:26].strip()))
    n_res = max(len(residues), 1)
    return dope / n_res if dope is not None else None

# 构建 (peptide, hla_allele) -> pdb_file 映射，覆盖正样本和负样本
# 正样本：output_with_pdb_files.csv (id == id_epitope == id_hla)
pos_pdb_df = pd.read_csv(os.path.join(SURF_DIR, "output_with_pdb_files.csv"))
phla_to_pdb = {(row['Epitope'], row['hla_allele']): row['pdb_file']
               for _, row in pos_pdb_df.iterrows()}

# 负样本：negative_phla_pandora_with_ids.csv (id_epitope != id_hla)
neg_pdb_df = pd.read_csv(os.path.join(SURF_DIR, "negative_phla_pandora_with_ids.csv"))
for _, row in neg_pdb_df.iterrows():
    phla_to_pdb[(row['peptide'], row['hla_allele'])] = row['pdb_file']

# 验证fold CSV中每行都能找到对应PDB
for fold_num in range(5):
    fold_path = os.path.join(SURF_DIR, f"fold_{fold_num}_val_predictions.csv")
    fdf = pd.read_csv(fold_path)
    for _, row in fdf.iterrows():
        key = (row['Epitope'], row['hla_allele'])
        if key not in phla_to_pdb:
            raise ValueError(f"fold_{fold_num}: no PDB found for Epitope={row['Epitope']}, hla_allele={row['hla_allele']}, id_epitope={row.get('id_epitope')}, id_hla={row.get('id_hla')}")

# 读取所有unique pdb的DOPE
pdb_dope = {}
for pdb_file in set(phla_to_pdb.values()):
    path = os.path.join(SURF_DIR, "PANDORA_pdb_output", pdb_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PANDORA pdb: {path}")
    pdb_dope[pdb_file] = parse_dope(path)
print(f"PANDORA DOPE loaded: {len(pdb_dope)} unique files")

meta['phla_dope_per_res'] = [
    pdb_dope.get(phla_to_pdb.get((ep, hla)))
    for ep, hla in zip(meta['Epitope'], meta['hla_allele'])
]

# ── 3. TCR structure reliability (tFold) ──────────────────────────────────────
TCR_DIR = os.path.join(SURF_DIR, "tcr_only_tFold")

def parse_tfold(id_tcr):
    path = os.path.join(TCR_DIR, f"tcr_{int(id_tcr):06d}.pdb")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing tFold pdb: {path}")
    lddt, ptm, iptm = None, None, None
    with open(path) as f:
        for line in f:
            if 'lDDT-Ca score:' in line:
                lddt = float(line.split()[-1])
            elif 'ipTM score:' in line:
                iptm = float(line.split()[-1])
            elif 'pTM score:' in line:
                ptm = float(line.split()[-1])
    return lddt, ptm, iptm

unique_tcr = meta['id_tcr'].unique()
tcr_cache = {}
for tid in unique_tcr:
    tcr_cache[tid] = parse_tfold(tid)
print(f"tFold scores loaded: {len(tcr_cache)} entries")

meta['tcr_lDDT'] = meta['id_tcr'].map(lambda x: tcr_cache[x][0])
meta['tcr_pTM']  = meta['id_tcr'].map(lambda x: tcr_cache[x][1])
meta['tcr_ipTM'] = meta['id_tcr'].map(lambda x: tcr_cache[x][2])

# ── Z-score normalization for structure features ───────────────────────────────
struct_cols = ['phla_dope_per_res', 'tcr_lDDT', 'tcr_pTM', 'tcr_ipTM']
zscore_params = {}
for col in struct_cols:
    mu  = meta[col].mean()
    std = meta[col].std()
    zscore_params[col] = {'mean': mu, 'std': std}
    meta[col] = (meta[col] - mu) / std

pd.DataFrame(zscore_params).T.rename_axis('feature').reset_index().to_csv(
    os.path.join(DATA_DIR, "structure_zscore_params.csv"), index=False
)
print(f"Z-score params saved → structure_zscore_params.csv")
print(pd.DataFrame(zscore_params).T.to_string())

# ── 4. Sequence familiarity (positive-only, leave-one-fold-out, BLOSUM62) ─────
# 读取数据库中的CDR3信息
db = pd.read_csv(os.path.join(DB_DIR, "09_FINAL_deduped_reindexed_cleaned.csv"),
                 usecols=['id', 'CDR3α', 'CDR3β'])
db = db.rename(columns={'CDR3α': 'cdr3a', 'CDR3β': 'cdr3b'}).set_index('id')

# 读取MHC pseudo序列
pseudo_df = pd.read_csv(os.path.join(DB_DIR, "MHC_psuedo.dat"),
                        sep=r'\s+', header=None, names=['allele','pseudo'], engine='python')
pseudo_map = pseudo_df.set_index('allele')['pseudo'].to_dict()

def normalize_allele(a):
    return ':'.join(a.replace('*','').split(':')[:2])

# 读取5个fold的seqonly csv，构建pool
fold_dfs = {}
for f in sorted(glob.glob(os.path.join(SEQ_DIR, "fold_*_val_predictions.csv"))):
    fold_num = int(os.path.basename(f).split('_')[1])
    fold_dfs[fold_num] = pd.read_csv(f)

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'

# 预计算BLOSUM62矩阵（20×20），加速向量编码
_BL_MAT = np.array([[BLOSUM62.get((a,b), BLOSUM62.get((b,a), 0))
                     for b in AA_ORDER] for a in AA_ORDER], dtype=float)
_AA2IDX = {a: i for i, a in enumerate(AA_ORDER)}

def blosum_vec(seq):
    if not isinstance(seq, str) or not seq:
        return np.zeros(20)
    idx = [_AA2IDX[a] for a in seq if a in _AA2IDX]
    if not idx:
        return np.zeros(20)
    vec = _BL_MAT[idx].sum(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def blosum_sim_exact(s1, s2, _aligner=[None]):
    """精确BLOSUM62归一化相似度，复用aligner实例"""
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    if _aligner[0] is None:
        from Bio.Align import PairwiseAligner
        a = PairwiseAligner()
        a.substitution_matrix = BLOSUM62
        a.open_gap_score = -10
        a.extend_gap_score = -0.5
        _aligner[0] = a
    aligner = _aligner[0]
    score = aligner.score(s1, s2)
    denom = max(aligner.score(s1, s1), aligner.score(s2, s2))
    return float(score / denom) if denom > 0 else 0.0

def build_vec_index(seqs):
    """对unique序列建向量索引，返回 (unique_seqs, matrix)"""
    unique = [s for s in dict.fromkeys(seqs) if isinstance(s, str) and s]
    mat = np.stack([blosum_vec(s) for s in unique]) if unique else np.zeros((0, 20))
    return unique, mat

def max_sim_vec(query, pool_mat):
    """向量余弦相似度最大值"""
    if pool_mat.shape[0] == 0:
        return 0.0
    qv = blosum_vec(query)
    if np.linalg.norm(qv) == 0:
        return 0.0
    sims = pool_mat @ qv  # 已归一化，直接点积=余弦
    return float(np.max(sims))

def max_sim_exact_cached(query, pool_seqs, cache):
    """精确比对+缓存（用于unique数少的列）"""
    if not isinstance(query, str) or not query:
        return 0.0
    best = 0.0
    for p in pool_seqs:
        key = (query, p)
        if key not in cache:
            cache[key] = blosum_sim_exact(query, p)
        best = max(best, cache[key])
    return best

# 准备数据
meta = meta.join(db, on='id_tcr', how='left')
meta['hla_pseudo'] = meta['hla_allele'].apply(normalize_allele).map(pseudo_map)

all_fold_ids = {}
for fn, fdf in fold_dfs.items():
    id_col = 'id_hla' if 'id_hla' in fdf.columns else 'id'
    ep_col = 'id_epitope' if 'id_epitope' in fdf.columns else id_col
    all_fold_ids[fn] = set(zip(fdf[id_col], fdf[ep_col], fdf['id_tcr']))
id_to_fold = {}
for fn, triples in all_fold_ids.items():
    for triple in triples:
        id_to_fold[triple] = fn

# 构建每个fold的positive-only pool（含CDR3和pseudo）
pool_data = {}
for fold_num, fdf in fold_dfs.items():
    fdf = fdf.copy()
    fdf['fold'] = fold_num
    fdf = fdf[fdf['label'] == 1].copy()
    fdf = fdf.join(db, on='id_tcr', how='left')
    fdf['hla_pseudo'] = fdf['hla_allele'].apply(normalize_allele).map(pseudo_map)
    pool_data[fold_num] = fdf

vec_cols = ['cdr3a', 'cdr3b', 'tcra_variable', 'tcrb_variable']
exact_cols = ['Epitope', 'hla_pseudo']
sim_specs = [
    ('Epitope', 'sim_epitope'),
    ('hla_pseudo', 'sim_hla_pseudo'),
    ('cdr3a', 'sim_cdr3a'),
    ('cdr3b', 'sim_cdr3b'),
    ('tcra_variable', 'sim_tcra_full'),
    ('tcrb_variable', 'sim_tcrb_full'),
]
stratified_sim_cols = {'Epitope', 'cdr3a', 'cdr3b', 'tcra_variable', 'tcrb_variable'}

print("Building positive-only leave-one-fold-out BLOSUM62 indices...")
fold_indices = {}
for fold_num in tqdm(fold_dfs, desc="Build fold indices"):
    pool = pd.concat([df for fn, df in pool_data.items() if fn != fold_num], ignore_index=True)
    fold_indices[fold_num] = {'pool': pool, 'vec': {}, 'exact': {}}
    for col in tqdm(vec_cols, desc=f"fold {fold_num} vector cols", leave=False):
        seqs = [s for s in pool[col].dropna().unique() if isinstance(s, str)]
        mat = np.stack([blosum_vec(s) for s in seqs]) if seqs else np.zeros((0, 20))
        hla_sets = pool.dropna(subset=[col]).groupby(col)['hla_allele'].agg(lambda x: set(x))
        fold_indices[fold_num]['vec'][col] = {'seqs': seqs, 'mat': mat, 'hla_sets': hla_sets}
    for col in tqdm(exact_cols, desc=f"fold {fold_num} exact cols", leave=False):
        pool_seqs = [s for s in pool[col].dropna().unique() if isinstance(s, str)]
        meta_seqs = [s for s in meta[col].dropna().unique() if isinstance(s, str)]
        sim_matrix = {
            q: {p: blosum_sim_exact(q, p) for p in pool_seqs}
            for q in tqdm(meta_seqs, desc=f"fold {fold_num} {col} align", leave=False)
        }
        hla_sets = pool.dropna(subset=[col]).groupby(col)['hla_allele'].agg(lambda x: set(x))
        fold_indices[fold_num]['exact'][col] = {'pool_seqs': pool_seqs, 'sim_matrix': sim_matrix, 'hla_sets': hla_sets}
    print(f"  fold {fold_num}: {len(pool)} positive reference rows")

def _top5_mean(values):
    if values.size == 0:
        return 0.0
    k = min(5, values.size)
    return float(np.mean(np.partition(values, -k)[-k:]))

def _compute_sim_col_batched(col, res_col):
    """批量计算一列的 global/same_hla/other_hla 相似度，按 fold 分组处理"""
    results = np.zeros((len(meta), 4), dtype=float)

    for fold_num, fold_meta in meta.groupby('fold'):
        fidx = fold_indices[fold_num]
        row_indices = fold_meta.index

        if col in vec_cols:
            col_idx = fidx['vec'][col]
            pool_seqs = col_idx['seqs']
            pool_mat  = col_idx['mat']          # (P, 20)
            hla_sets  = col_idx['hla_sets']

            # 批量向量化：(N, 20) @ (20, P) → (N, P)
            queries = fold_meta[col].tolist()
            q_vecs  = np.stack([blosum_vec(q) for q in queries])  # (N, 20)
            sim_mat = q_vecs @ pool_mat.T                          # (N, P)

        else:
            col_idx  = fidx['exact'][col]
            pool_seqs = col_idx['pool_seqs']
            sim_matrix = col_idx['sim_matrix']
            hla_sets   = col_idx['hla_sets']

            queries = fold_meta[col].tolist()
            # (N, P) from precomputed dict
            sim_mat = np.array([
                [sim_matrix.get(q, {}).get(p, 0.0) for p in pool_seqs]
                for q in queries
            ], dtype=float)

        if sim_mat.shape[1] == 0:
            continue

        # global max
        global_max = sim_mat.max(axis=1)  # (N,)

        if col not in stratified_sim_cols:
            results[row_indices, 0] = global_max
            continue

        # 预计算 pool 中每个序列属于哪些 HLA 的 boolean 矩阵
        # 对每个 query row，same/other mask 取决于 row['hla_allele']
        query_hlas = fold_meta['hla_allele'].tolist()
        unique_hlas = list(dict.fromkeys(query_hlas))

        # pool_same[h][j] = True if pool_seqs[j] appears under hla h
        hla_same_mask = {
            h: np.array([h in hla_sets.get(s, set()) for s in pool_seqs], dtype=bool)
            for h in unique_hlas
        }
        hla_other_mask = {
            h: np.array([any(x != h for x in hla_sets.get(s, set())) for s in pool_seqs], dtype=bool)
            for h in unique_hlas
        }

        same_max   = np.zeros(len(queries))
        other_max  = np.zeros(len(queries))
        other_top5 = np.zeros(len(queries))

        for i, h in enumerate(query_hlas):
            sm = hla_same_mask[h]
            om = hla_other_mask[h]
            row_sims = sim_mat[i]
            same_sims  = row_sims[sm]
            other_sims = row_sims[om]
            same_max[i]   = same_sims.max()  if same_sims.size  else 0.0
            other_max[i]  = other_sims.max() if other_sims.size else 0.0
            other_top5[i] = _top5_mean(other_sims)

        results[row_indices, 0] = global_max
        results[row_indices, 1] = same_max
        results[row_indices, 2] = other_max
        results[row_indices, 3] = other_top5

    return results

print("Computing sequence familiarity (positive-only, leave-one-fold-out, batched)...")
for col, res_col in tqdm(sim_specs, desc="Similarity features"):
    results = _compute_sim_col_batched(col, res_col)
    meta[res_col] = results[:, 0]
    if col in stratified_sim_cols:
        meta[f'{res_col}_same_hla_max']        = results[:, 1]
        meta[f'{res_col}_other_hla_max']        = results[:, 2]
        meta[f'{res_col}_other_hla_top5_mean']  = results[:, 3]
    print(f"  {res_col} done")

# ── 5. Conflict features ───────────────────────────────────────────────────────
meta['prob_diff']          = meta['seq_cal_prob'] - meta['surf_cal_prob']
meta['entropy_diff']       = meta['entropy_seq']  - meta['entropy_surf']
meta['direction_conflict'] = ((meta['sequence_logit'] > 0) != (meta['surface_logit'] > 0)).astype(int)

# prob_product: log变换后z-score（避免两小概率相乘极小）
_pp = np.log(meta['seq_cal_prob'].values * meta['surf_cal_prob'].values + 1e-10)
_pp_mean, _pp_std = np.nanmean(_pp), np.nanstd(_pp)
meta['prob_product_log'] = (_pp - _pp_mean) / _pp_std

# logit_product: 原始乘积，保留方向语义
_lp = meta['sequence_logit'].values * meta['surface_logit'].values
meta['logit_product'] = _lp

# 保存zscore参数（仅prob_product_log）
pd.DataFrame({
    'feature': ['prob_product_log'],
    'mean': [_pp_mean],
    'std':  [_pp_std],
}).to_csv(os.path.join(DATA_DIR, "conflict_zscore_params.csv"), index=False)
print("conflict zscore params saved")

# ── 6. 保存 ───────────────────────────────────────────────────────────────────
feature_cols = [
    'id', 'id_hla', 'id_epitope', 'id_tcr', 'label', 'fold',
    'sequence_logit', 'surface_logit',
    'seq_cal_prob', 'surf_cal_prob',
    'entropy_seq', 'entropy_surf',
    'phla_dope_per_res',
    'tcr_lDDT', 'tcr_pTM', 'tcr_ipTM',
    'sim_epitope',
    'sim_epitope_same_hla_max', 'sim_epitope_other_hla_max', 'sim_epitope_other_hla_top5_mean',
    'sim_hla_pseudo',
    'sim_cdr3a',
    'sim_cdr3a_same_hla_max', 'sim_cdr3a_other_hla_max', 'sim_cdr3a_other_hla_top5_mean',
    'sim_cdr3b',
    'sim_cdr3b_same_hla_max', 'sim_cdr3b_other_hla_max', 'sim_cdr3b_other_hla_top5_mean',
    'sim_tcra_full',
    'sim_tcra_full_same_hla_max', 'sim_tcra_full_other_hla_max', 'sim_tcra_full_other_hla_top5_mean',
    'sim_tcrb_full',
    'sim_tcrb_full_same_hla_max', 'sim_tcrb_full_other_hla_max', 'sim_tcrb_full_other_hla_top5_mean',
    'prob_diff', 'entropy_diff', 'prob_product_log', 'logit_product', 'direction_conflict',
]
out = meta[feature_cols]
out.to_csv(os.path.join(DATA_DIR, "meta_aggregated_dataset.csv"), index=False)
print(f"Saved {len(out)} rows, {len(feature_cols)-4} features → meta_aggregated_dataset.csv")
print(out[feature_cols[4:]].describe().to_string())

