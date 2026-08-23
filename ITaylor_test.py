#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By default:
# python ITaylor_test.py --seq_model_dir ./sequence_weight/neg_ratio_10 --surf_model_dir ./surface_weight/neg_ratio_10 --test_dir ./data/Database_stage1/test_outputs --database_csv ./data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv --mhc_pseudo ./data/Database_stage1/MHC_psuedo.dat --imfp_dir ./data/Database_stage2/imfp --output_dir ./predictions/ITaylor_test
# Preflight only: python ITaylor_test.py --preflight-only

"""Standalone ITaylor test pipeline.

All relative paths are resolved from this file's directory. Run
``python ITaylor_test.py --preflight-only`` before a full GPU run.

Runtime packages: numpy, pandas, torch, xgboost, scikit-learn, biopython,
tqdm, and evolutionaryscale/esm (plus an available/cached esmc_600m model).
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "Adaptive_correction"
CALIB_DIR = ANALYSIS_DIR / "calibration_meta_data"
SURF_DIR = ANALYSIS_DIR / "fold_calibration_surfonly"
SEQ_DIR = ANALYSIS_DIR / "fold_calibration_seqonly"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(CALIB_DIR))

# Populated only after argument parsing and preflight, so --help stays lightweight.
np = pd = torch = xgb = nn = None
roc_auc_score = average_precision_score = None
substitution_matrices = None
tqdm = None
BLOSUM62 = None
_BL_MAT = None
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA2IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

FEATURE_COLS_N5 = [
    "seq_cal_prob", "surf_cal_prob", "entropy_seq", "entropy_surf",
    "phla_dope_per_res", "tcr_lDDT", "tcr_pTM", "tcr_ipTM",
    "sim_epitope_same_hla_max", "sim_epitope_other_hla_max",
    "sim_epitope_other_hla_top5_mean", "sim_cdr3a_same_hla_max",
    "sim_cdr3a_other_hla_max", "sim_cdr3a_other_hla_top5_mean",
    "sim_cdr3b_same_hla_max", "sim_cdr3b_other_hla_max",
    "sim_cdr3b_other_hla_top5_mean", "sim_tcra_full_same_hla_max",
    "sim_tcra_full_other_hla_max", "sim_tcra_full_other_hla_top5_mean",
    "sim_tcrb_full_same_hla_max", "sim_tcrb_full_other_hla_max",
    "sim_tcrb_full_other_hla_top5_mean", "prob_diff", "entropy_diff",
    "prob_product_log", "signed_product_z",
]

TEST_REQUIRED_COLUMNS = {
    "id", "id_tcr", "hla_allele", "Epitope", "hla_alpha123_mature",
    "tcra_variable", "tcrb_variable",
}

PYTHON_DEPENDENCIES = {
    "numpy": "numpy", "pandas": "pandas", "torch": "torch",
    "xgboost": "xgboost", "scikit-learn": "sklearn",
    "biopython": "Bio", "tqdm": "tqdm", "esm": "esm",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def has_five_fold_models(path: Path) -> bool:
    return all((path / f"fold_{fold}" / "best_model.pt").is_file()
               for fold in range(5))


def resolve_model_dir(path: Path) -> Path:
    """Accept a fold root or one unambiguous wrapper directory around it."""
    if has_five_fold_models(path):
        return path
    if path.is_dir():
        candidates = [child for child in path.iterdir()
                      if child.is_dir() and has_five_fold_models(child)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(str(p) for p in candidates)
            raise ValueError(f"Multiple five-fold model directories under {path}: {names}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ITaylor sequence/surface meta-fusion test")
    parser.add_argument("--seq_model_dir", default="sequence_weight/neg_ratio_10")
    parser.add_argument("--surf_model_dir", default="surface_weight/neg_ratio_10")
    parser.add_argument("--test_dir", default="data/Database_stage1/test_outputs")
    parser.add_argument(
        "--database_csv",
        default="data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv")
    parser.add_argument("--mhc_pseudo", default="data/Database_stage1/MHC_psuedo.dat")
    parser.add_argument("--imfp_dir", default="data/Database_stage2/imfp")
    parser.add_argument("--output_dir", default="predictions/ITaylor_test")
    parser.add_argument("--seq_device", default="cuda:0")
    parser.add_argument("--surf_device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--esm_model", default="esmc_600m")
    parser.add_argument("--pattern", default="TEST_*.csv")
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="validate paths, files, CSV columns, and Python packages, then exit")
    return parser


def normalize_args(args):
    for name in ("seq_model_dir", "surf_model_dir", "test_dir", "database_csv",
                 "mhc_pseudo", "imfp_dir", "output_dir"):
        setattr(args, name, resolve_path(getattr(args, name)))
    args.seq_model_dir = resolve_model_dir(args.seq_model_dir)
    args.surf_model_dir = resolve_model_dir(args.surf_model_dir)
    return args


def _csv_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle)))


def preflight(args) -> tuple[list[str], list[str], list[Path]]:
    errors, notes = [], []

    for package, module in PYTHON_DEPENDENCIES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"Python package missing: {package} (import name: {module})")

    for label, path in (("sequence model", args.seq_model_dir),
                        ("surface model", args.surf_model_dir)):
        for fold in range(5):
            model = path / f"fold_{fold}" / "best_model.pt"
            if not model.is_file():
                errors.append(f"Missing {label} weight: {model}")

    required_files = [
        ROOT / "predict_stage1_logging.py",
        ROOT / "predict_stage2_surfonly_logging.py",
        ROOT / "Network_v3.py",
        ROOT / "MaSIFCore_v2.py",
        ANALYSIS_DIR / "apply_structure_zscore.py",
        args.database_csv,
        args.mhc_pseudo,
        SURF_DIR / "output_with_pdb_files.csv",
        CALIB_DIR / "sequence_calibration_params.csv",
        CALIB_DIR / "surface_calibration_params.csv",
        CALIB_DIR / "structure_zscore_params.csv",
        CALIB_DIR / "conflict_zscore_params.csv",
        CALIB_DIR / "signed_product_zscore_params.csv",
        CALIB_DIR / "xgboost_meta_scaled_new_newsim.json",
        CALIB_DIR / "delta_hat_zscore_params_new_newsim.csv",
        CALIB_DIR / "alpha_top10_feat_names_beta1.csv",
        CALIB_DIR / "alpha_top10_feat_mean_beta1.csv",
        CALIB_DIR / "alpha_top10_feat_std_beta1.csv",
        CALIB_DIR / "alpha_net_mlpalpha2_beta1.pt",
    ]
    required_files.extend(SEQ_DIR / f"fold_{fold}_val_predictions.csv"
                          for fold in range(5))
    for path in required_files:
        if not path.is_file():
            errors.append(f"Missing required file: {path}")

    tcr_fold_dir = SURF_DIR / "tcr_only_tFold"
    if not tcr_fold_dir.is_dir():
        errors.append(f"Missing tFold directory: {tcr_fold_dir}")

    pandora_dir = SURF_DIR / "PANDORA_pdb_output"
    if not pandora_dir.is_dir():
        archive = SURF_DIR / "PANDORA_pdb_output.tar.gz"
        suffix = f" Archive found at {archive}; extract it on the target machine." if archive.is_file() else ""
        errors.append(f"Missing extracted PANDORA directory: {pandora_dir}.{suffix}")

    if not args.imfp_dir.is_dir():
        errors.append(f"Missing MaSIF imfp directory: {args.imfp_dir}")

    test_files = sorted(args.test_dir.glob(args.pattern)) if args.test_dir.is_dir() else []
    if not args.test_dir.is_dir():
        errors.append(f"Missing test directory: {args.test_dir}")
    elif not test_files:
        errors.append(f"No test files match {args.pattern!r} in {args.test_dir}")
    for path in test_files:
        try:
            columns = _csv_columns(path)
        except Exception as exc:
            errors.append(f"Cannot read CSV header {path}: {exc}")
            continue
        missing = sorted(TEST_REQUIRED_COLUMNS - columns)
        if missing:
            errors.append(f"{path.name} missing columns: {', '.join(missing)}")
    notes.append(f"Resolved sequence model directory: {args.seq_model_dir}")
    notes.append(f"Resolved surface model directory: {args.surf_model_dir}")
    notes.append("ESM-C esmc_600m must be cached locally or downloadable at runtime")
    return errors, notes, test_files


def print_preflight(errors: list[str], notes: list[str]) -> None:
    print("ITaylor preflight")
    for note in notes:
        print(f"[INFO] {note}")
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        print(f"Preflight failed with {len(errors)} issue(s).")
    else:
        print("Preflight passed.")


def import_runtime_dependencies() -> None:
    global np, pd, torch, xgb, nn, roc_auc_score, average_precision_score
    global substitution_matrices, tqdm, BLOSUM62, _BL_MAT
    try:
        import numpy as _np
        import pandas as _pd
        import torch as _torch
        import torch.nn as _nn
        import xgboost as _xgb
        from sklearn.metrics import average_precision_score as _ap
        from sklearn.metrics import roc_auc_score as _roc
        from Bio.Align import substitution_matrices as _substitution_matrices
        from tqdm import tqdm as _tqdm
    except ImportError as exc:
        raise RuntimeError(f"Runtime dependency import failed: {exc}") from exc
    np, pd, torch, nn, xgb = _np, _pd, _torch, _nn, _xgb
    roc_auc_score, average_precision_score = _roc, _ap
    substitution_matrices, tqdm = _substitution_matrices, _tqdm
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
    _BL_MAT = np.array([
        [BLOSUM62.get((a, b), BLOSUM62.get((b, a), 0)) for b in AA_ORDER]
        for a in AA_ORDER
    ], dtype=float)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ITaylor_test")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "ITaylor_test.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


def predict_seq_logits(df, predictor, batch_size, logger):
    epitopes = df["Epitope"].tolist()
    hlas = df["hla_alpha123_mature"].tolist()
    tcras = df["tcra_variable"].tolist()
    tcrbs = df["tcrb_variable"].tolist()
    logger.info("  Extracting ESM embeddings once...")
    batches = [predictor._prepare_batch_input(epitopes[i:i + batch_size],
                hlas[i:i + batch_size], tcras[i:i + batch_size],
                tcrbs[i:i + batch_size]) for i in range(0, len(df), batch_size)]
    fold_results = []
    for fold_idx, model in enumerate(predictor.models):
        logits = []
        for batch in batches:
            with torch.no_grad():
                output = model(
                    peptide_emb=batch["peptide_emb"], hla_emb=batch["hla_emb"],
                    tcra_emb=batch["tcra_emb"], tcrb_emb=batch["tcrb_emb"],
                    peptide_mask=batch["peptide_mask"], hla_mask=batch["hla_mask"],
                    tcra_mask=batch["tcra_mask"], tcrb_mask=batch["tcrb_mask"],
                    mode="seq_only")
            logits.append(output["logit"].squeeze(-1).detach().cpu().numpy())
        fold_results.append(np.concatenate(logits))
        logger.info("  sequence fold %d done", fold_idx)
    return np.stack(fold_results).mean(axis=0)


def predict_surf_logits(df, predictor, batch_size, logger):
    pmhc_ids, tcr_ids = df["id"].tolist(), df["id_tcr"].tolist()
    logger.info("  Loading MaSIF data once...")
    batches = [predictor._prepare_batch_input(pmhc_ids[i:i + batch_size],
                tcr_ids[i:i + batch_size]) for i in range(0, len(df), batch_size)]
    fold_results = []
    for fold_idx, model in enumerate(predictor.models):
        logits = []
        for batch in batches:
            with torch.no_grad():
                output = model(
                    peptide_emb=None, hla_emb=None, tcra_emb=None, tcrb_emb=None,
                    peptide_mask=None, hla_mask=None, tcra_mask=None, tcrb_mask=None,
                    phla_masifs=batch["phla_masifs"], tcr_masifs=batch["tcr_masifs"],
                    mode="surf_only")
            logits.append(output["logit"].squeeze(-1).detach().cpu().numpy())
        fold_results.append(np.concatenate(logits))
        logger.info("  surface fold %d done", fold_idx)
    return np.stack(fold_results).mean(axis=0)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def entropy(probabilities):
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load_platt(name: str):
    row = pd.read_csv(CALIB_DIR / f"{name}_calibration_params.csv").iloc[0]
    return float(row["platt_A"]), float(row["platt_B"])


def parse_dope(path: Path):
    dope, residues = None, set()
    with path.open() as handle:
        for line in handle:
            if "DOPE score:" in line:
                dope = float(line.split()[-1])
            if line.startswith("ATOM"):
                residues.add((line[21], line[22:26].strip()))
    return dope / max(len(residues), 1) if dope is not None else None


def parse_tfold(id_tcr):
    path = SURF_DIR / "tcr_only_tFold" / f"tcr_{int(id_tcr):06d}.pdb"
    lddt = ptm = iptm = None
    with path.open() as handle:
        for line in handle:
            if "lDDT-Ca score:" in line:
                lddt = float(line.split()[-1])
            elif "ipTM score:" in line:
                iptm = float(line.split()[-1])
            elif "pTM score:" in line:
                ptm = float(line.split()[-1])
    return lddt, ptm, iptm


def build_structure_features(df, logger):
    from apply_structure_zscore import apply_zscore, load_zscore_params
    pdb_df = pd.read_csv(SURF_DIR / "output_with_pdb_files.csv")
    pdb_map = pdb_df.set_index("id")["pdb_file"].to_dict()
    pdb_dope = {name: parse_dope(SURF_DIR / "PANDORA_pdb_output" / name)
                for name in pdb_df["pdb_file"].unique()}
    result = df.copy()
    result["phla_dope_per_res"] = result["id"].map(
        {row_id: pdb_dope[pdb] for row_id, pdb in pdb_map.items()})
    logger.info("  DOPE loaded for %d samples", result["phla_dope_per_res"].notna().sum())
    tcr_cache = {tcr_id: parse_tfold(tcr_id) for tcr_id in result["id_tcr"].unique()}
    result["tcr_lDDT"] = result["id_tcr"].map(lambda value: tcr_cache[value][0])
    result["tcr_pTM"] = result["id_tcr"].map(lambda value: tcr_cache[value][1])
    result["tcr_ipTM"] = result["id_tcr"].map(lambda value: tcr_cache[value][2])
    params = load_zscore_params(str(CALIB_DIR / "structure_zscore_params.csv"))
    return apply_zscore(result, params)


def blosum_vec(sequence):
    if not isinstance(sequence, str) or not sequence:
        return np.zeros(20)
    indices = [_AA2IDX[aa] for aa in sequence if aa in _AA2IDX]
    if not indices:
        return np.zeros(20)
    vector = _BL_MAT[indices].sum(axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def blosum_sim_exact(left, right, cache=[None]):
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return 0.0
    if cache[0] is None:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.substitution_matrix = BLOSUM62
        aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
        cache[0] = aligner
    score = cache[0].score(left, right)
    denominator = max(cache[0].score(left, left), cache[0].score(right, right))
    return float(score / denominator) if denominator > 0 else 0.0


def hla_stratified(similarity, pool_hlas, query_hlas):
    query_hlas = np.asarray(query_hlas)
    same_max = np.zeros(len(query_hlas))
    other_max = np.zeros(len(query_hlas))
    other_top5 = np.zeros(len(query_hlas))
    for hla in dict.fromkeys(query_hlas.tolist()):
        rows = np.where(query_hlas == hla)[0]
        same = np.array([hla in values for values in pool_hlas], dtype=bool)
        other = np.array([any(value != hla for value in values) for values in pool_hlas], dtype=bool)
        if same.any():
            same_max[rows] = similarity[rows][:, same].max(axis=1)
        if other.any():
            values = similarity[rows][:, other]
            other_max[rows] = values.max(axis=1)
            k = min(5, values.shape[1])
            other_top5[rows] = np.partition(values, -k, axis=1)[:, -k:].mean(axis=1)
    return same_max, other_max, other_top5


def build_seq_familiarity(df, args, logger):
    database = pd.read_csv(args.database_csv, usecols=["id", "CDR3α", "CDR3β"])
    database = database.rename(columns={"CDR3α": "cdr3a", "CDR3β": "cdr3b"}).set_index("id")
    pseudo_df = pd.read_csv(args.mhc_pseudo, sep=r"\s+", header=None,
                            names=["allele", "pseudo"], engine="python")
    pseudo_map = pseudo_df.set_index("allele")["pseudo"].to_dict()
    pool = pd.concat([pd.read_csv(path) for path in sorted(
        glob.glob(str(SEQ_DIR / "fold_*_val_predictions.csv")))], ignore_index=True)
    pool = pool[pool["label"] == 1].copy().join(database, on="id_tcr", how="left")
    normalize = lambda allele: ":".join(allele.replace("*", "").split(":")[:2])
    pool["hla_pseudo"] = pool["hla_allele"].apply(normalize).map(pseudo_map)
    result = df.copy().join(database, on="id_tcr", how="left")
    result["hla_pseudo"] = result["hla_allele"].apply(normalize).map(pseudo_map)
    query_hlas = result["hla_allele"].tolist()

    vector_columns = [("cdr3a", "sim_cdr3a"), ("cdr3b", "sim_cdr3b"),
                      ("tcra_variable", "sim_tcra_full"),
                      ("tcrb_variable", "sim_tcrb_full")]
    for column, output in vector_columns:
        pool_sequences = [s for s in pool[column].dropna().unique()
                          if isinstance(s, str) and s]
        if not pool_sequences:
            for suffix in ("", "_same_hla_max", "_other_hla_max", "_other_hla_top5_mean"):
                result[output + suffix] = 0.0
            continue
        pool_matrix = np.stack([blosum_vec(sequence) for sequence in pool_sequences])
        pool_hlas = [set(pool.loc[pool[column] == sequence, "hla_allele"].dropna())
                     for sequence in pool_sequences]
        query_matrix = np.stack([blosum_vec(sequence)
                                 for sequence in result[column].fillna("")])
        similarity = query_matrix @ pool_matrix.T
        result[output] = similarity.max(axis=1)
        same, other, top5 = hla_stratified(similarity, pool_hlas, query_hlas)
        result[output + "_same_hla_max"] = same
        result[output + "_other_hla_max"] = other
        result[output + "_other_hla_top5_mean"] = top5
    logger.info("  Vector similarities done")

    for column, output in (("Epitope", "sim_epitope"),
                           ("hla_pseudo", "sim_hla_pseudo")):
        pool_sequences = [s for s in pool[column].dropna().unique()
                          if isinstance(s, str) and s]
        queries = [q for q in dict.fromkeys(result[column].fillna("").tolist()) if q]
        query_similarity = {
            query: np.array([blosum_sim_exact(query, candidate) if candidate != query else 0.0
                             for candidate in pool_sequences])
            for query in tqdm(queries, desc=f"Exact sim {column}", leave=False)
        }
        result[output] = [query_similarity[q].max() if q in query_similarity else 0.0
                          for q in result[column].fillna("")]
        if column == "Epitope":
            pool_hlas = [set(pool.loc[pool[column] == sequence, "hla_allele"].dropna())
                         for sequence in pool_sequences]
            similarity = np.stack([query_similarity.get(q, np.zeros(len(pool_sequences)))
                                   for q in result[column].fillna("")])
            same, other, top5 = hla_stratified(similarity, pool_hlas, query_hlas)
            result[output + "_same_hla_max"] = same
            result[output + "_other_hla_max"] = other
            result[output + "_other_hla_top5_mean"] = top5
    logger.info("  Exact similarities done")
    return result


def build_meta_features(df, args, logger):
    seq_a, seq_b = load_platt("sequence")
    surf_a, surf_b = load_platt("surface")
    result = df.copy()
    result["seq_cal_logit"] = seq_a * result["sequence_logit"] + seq_b
    result["surf_cal_logit"] = surf_a * result["surface_logit"] + surf_b
    result["seq_cal_prob"] = sigmoid(result["seq_cal_logit"].values)
    result["surf_cal_prob"] = sigmoid(result["surf_cal_logit"].values)
    result["entropy_seq"] = entropy(result["seq_cal_prob"].values)
    result["entropy_surf"] = entropy(result["surf_cal_prob"].values)
    result = build_structure_features(result, logger)
    result = build_seq_familiarity(result, args, logger)
    result["prob_diff"] = result["seq_cal_prob"] - result["surf_cal_prob"]
    result["entropy_diff"] = result["entropy_seq"] - result["entropy_surf"]
    params = pd.read_csv(CALIB_DIR / "conflict_zscore_params.csv").set_index("feature")
    product = np.log(result["seq_cal_prob"].values * result["surf_cal_prob"].values + 1e-10)
    result["prob_product_log"] = ((product - params.loc["prob_product_log", "mean"])
                                  / params.loc["prob_product_log", "std"])
    result["logit_product"] = result["sequence_logit"] * result["surface_logit"]
    signed_params = pd.read_csv(CALIB_DIR / "signed_product_zscore_params.csv").iloc[0]
    raw = result["seq_cal_logit"].values * result["surf_cal_logit"].values
    compressed = np.sign(raw) * np.log1p(np.abs(raw))
    result["signed_product_z"] = ((compressed - float(signed_params["mean"]))
                                  / float(signed_params["std"]))
    return result


def apply_learned_fusion(df, logger):
    class AlphaNet2D10(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(12, 32), nn.Tanh(), nn.Linear(32, 1))

        def forward(self, diff, z_seq, features):
            values = torch.cat([diff.unsqueeze(-1), z_seq.unsqueeze(-1), features], dim=-1)
            return self.net(values).squeeze(-1)

    features = df[FEATURE_COLS_N5].values.astype(np.float32)
    delta_model = xgb.XGBRegressor()
    delta_model.load_model(str(CALIB_DIR / "xgboost_meta_scaled_new_newsim.json"))
    delta_hat = delta_model.predict(features).astype(np.float32)
    delta_params = pd.read_csv(CALIB_DIR / "delta_hat_zscore_params_new_newsim.csv").iloc[0]
    delta_z = (delta_hat - float(delta_params["mean"])) / float(delta_params["std"])
    z_seq = df["seq_cal_logit"].values.astype(np.float32)
    z_surf = df["surf_cal_logit"].values.astype(np.float32)

    top10 = pd.read_csv(CALIB_DIR / "alpha_top10_feat_names_beta1.csv")["feature"].tolist()
    means = pd.read_csv(CALIB_DIR / "alpha_top10_feat_mean_beta1.csv").iloc[0][top10].values.astype(np.float32)
    stds = pd.read_csv(CALIB_DIR / "alpha_top10_feat_std_beta1.csv").iloc[0][top10].values.astype(np.float32)
    if np.any(stds == 0):
        raise ValueError("N5 top-10 feature standard deviation contains zero")
    top10_z = (df[top10].values.astype(np.float32) - means) / stds
    invalid = ~np.isfinite(top10_z)
    if invalid.any():
        affected_rows = int(invalid.any(axis=1).sum())
        logger.warning(
            "  ITaylor AlphaNet: mean-imputing %d non-finite standardized values "
            "across %d samples", int(invalid.sum()), affected_rows)
        top10_z[invalid] = 0.0

    alpha_model = AlphaNet2D10()
    state = torch.load(CALIB_DIR / "alpha_net_mlpalpha2_beta1.pt", map_location="cpu")
    alpha_model.load_state_dict(state)
    alpha_model.eval()
    diff = torch.from_numpy(z_surf - z_seq)
    with torch.no_grad():
        alpha = alpha_model(diff, torch.from_numpy(z_seq), torch.from_numpy(top10_z)).numpy()
    gate = sigmoid(delta_z)
    fused_logit = z_seq + gate * alpha * (z_surf - z_seq)
    logger.info("  [ITaylor fusion] beta=1.0, gate mean=%.4f, alpha mean=%.4f",
                gate.mean(), alpha.mean())
    return sigmoid(fused_logit), fused_logit, gate, alpha


def prepare_input_dataframe(path: Path):
    df = pd.read_csv(path)
    missing = TEST_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
    for column in ("id", "id_tcr"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.all(values == np.floor(values)):
            raise ValueError(f"Column {column} must contain integer identifiers")
        df[column] = values.astype(np.int64)
    return df


def process_file(path, args, logger, seq_predictor, surf_predictor):
    logger.info("=" * 70)
    logger.info("Processing %s", path.name)
    df = prepare_input_dataframe(path)
    logger.info("  %d samples", len(df))
    with ThreadPoolExecutor(max_workers=2) as executor:
        seq_future = executor.submit(predict_seq_logits, df, seq_predictor,
                                     args.batch_size, logger)
        surf_future = executor.submit(predict_surf_logits, df, surf_predictor,
                                      args.batch_size, logger)
        df["sequence_logit"] = seq_future.result()
        df["surface_logit"] = surf_future.result()
    df = build_meta_features(df, args, logger)
    probability, fused_logit, gate, alpha = apply_learned_fusion(df, logger)
    df["g_star"], df["alpha_i"] = gate, alpha
    df["fusion_logit"], df["ITaylor_score"] = fused_logit, probability

    similarity_columns = [column for column in FEATURE_COLS_N5 if column.startswith("sim_")]
    save_columns = [
        "id", "id_tcr", "sequence_logit", "surface_logit", "seq_cal_prob",
        "surf_cal_prob", "seq_cal_logit", "surf_cal_logit", "entropy_seq",
        "entropy_surf", "phla_dope_per_res", "tcr_lDDT", "tcr_pTM", "tcr_ipTM",
        *similarity_columns, "prob_diff", "entropy_diff", "prob_product_log",
        "logit_product", "signed_product_z", "g_star", "alpha_i",
        "fusion_logit", "ITaylor_score",
    ]
    if "label" in df.columns:
        save_columns.insert(2, "label")
    output_csv = args.output_dir / f"{path.stem}_meta_N5_predictions.csv"
    df[[column for column in save_columns if column in df.columns]].to_csv(output_csv, index=False)
    logger.info("  Saved %s", output_csv)

    metrics = {}
    if "label" in df.columns:
        labels = df["label"].values
        for name, scores in (("seq", sigmoid(df["sequence_logit"].values)),
                             ("surf", sigmoid(df["surface_logit"].values)),
                             ("ITaylor_score", probability)):
            metrics[name] = {
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
            }
        with (args.output_dir / f"{path.stem}_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    return metrics


def run(args, test_files):
    import_runtime_dependencies()
    logger = setup_logging(args.output_dir)
    logger.info("ITaylor pipeline")
    logger.info("Sequence weights: %s", args.seq_model_dir)
    logger.info("Surface weights: %s", args.surf_model_dir)
    from predict_stage1_logging import Stage1Predictor
    from predict_stage2_surfonly_logging import Stage2SurfOnlyPredictor
    seq_predictor = Stage1Predictor(
        model_dir=str(args.seq_model_dir), device=args.seq_device,
        esm_model_name=args.esm_model, logger=logger)
    surf_predictor = Stage2SurfOnlyPredictor(
        model_dir=str(args.surf_model_dir), imfp_dir=str(args.imfp_dir),
        device=args.surf_device, logger=logger)
    all_metrics = {}
    for path in test_files:
        try:
            all_metrics[path.name] = process_file(
                path, args, logger, seq_predictor, surf_predictor)
        except Exception as exc:
            logger.error("Failed %s: %s", path.name, exc, exc_info=True)
    summary = args.output_dir / "summary_metrics_N5.json"
    with summary.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    logger.info("Summary saved to %s", summary)


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    errors, notes, test_files = preflight(args)
    print_preflight(errors, notes)
    if args.preflight_only:
        return 1 if errors else 0
    if errors:
        print("Fix the preflight issues above, or use CLI path overrides.", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args, test_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
