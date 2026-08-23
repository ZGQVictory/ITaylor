#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example:
# python ITaylor_predict.py --epitope TLMSAMTNL --hla_allele "HLA-A*02:01" --hla_sequence GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEP  --tcra KEVEQNSGPLSVPEGAIASLNCTYSDRGSQSFFWYRQYSGKSPELIMFIYSNGDKEDGRFTAQLNKASQYVSLLIRDSQPSDSATYLCAVNNARLMFGDGTQLVVKP --tcrb GVTQTPKHLITATGQRVTLRCSPRSGDLSVYWYQQSLDQGLQFLIQYYNGEERAKGNILERFSAQQFPDLHSELNLSSLELGDSALYFCASSVAGSPEAFFGQGTRLTVV --cdr3a CAVNNARLMF  --cdr3b CASSVAGSPEAFF --pmhc_masif_dir ./example/9NMU_pHLA01/ --tcr_masif_dir ./example/9NMU_TCR01 --seq_model_dir ./sequence_weight/neg_ratio_10 --surf_model_dir ./surface_weight/neg_ratio10
"""Predict one ITayor_score from sequences and MaSIF features."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import ITaylor_test as core


ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "Adaptive_correction" / "calibration_meta_data"
SEQ_POOL_DIR = ROOT / "Adaptive_correction" / "fold_calibration_seqonly"
MASIF_FEATURES = ("charge", "ddc", "hbond", "hphob", "si")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the ITaylor score for one pHLA-TCR sample")
    parser.add_argument("--epitope", required=True)
    parser.add_argument("--hla_allele", required=True,
                        help="HLA allele, for example HLA-A*02:01")
    parser.add_argument("--hla_sequence", required=True,
                        help="mature HLA alpha1/alpha2/alpha3 sequence")
    parser.add_argument("--tcra", required=True, help="TCR alpha variable-region sequence")
    parser.add_argument("--tcrb", required=True, help="TCR beta variable-region sequence")
    parser.add_argument("--cdr3a", required=True, help="TCR alpha CDR3 sequence")
    parser.add_argument("--cdr3b", required=True, help="TCR beta CDR3 sequence")
    parser.add_argument("--pmhc_masif_dir", required=True,
                        help="directory directly containing p1_*.npy files")
    parser.add_argument("--tcr_masif_dir", required=True,
                        help="directory directly containing p2_*.npy files")
    parser.add_argument("--seq_model_dir", default="sequence_weight/neg_ratio_10")
    parser.add_argument("--surf_model_dir", default="surface_weight/neg_ratio10")
    parser.add_argument(
        "--database_csv",
        default="data/Database_stage1/09_FINAL_deduped_reindexed_cleaned.csv")
    parser.add_argument("--seq_device", default="cuda:0")
    parser.add_argument("--surf_device", default="cuda:1")
    parser.add_argument("--esm_model", default="esmc_600m")
    parser.add_argument(
        "--phla_dope_per_res", type=float, default=None,
        help="raw DOPE per residue; default is training mean minus one std")
    parser.add_argument("--tcr_lddt", type=float, default=1.0)
    parser.add_argument("--tcr_ptm", type=float, default=1.0)
    parser.add_argument("--tcr_iptm", type=float, default=1.0)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="validate inputs and required files without loading the models")
    return parser


def normalize_args(args):
    for name in ("pmhc_masif_dir", "tcr_masif_dir", "seq_model_dir",
                 "surf_model_dir", "database_csv"):
        setattr(args, name, resolve_path(getattr(args, name)))
    args.seq_model_dir = core.resolve_model_dir(args.seq_model_dir)
    args.surf_model_dir = core.resolve_model_dir(args.surf_model_dir)
    return args


def required_masif_files(folder: Path, prefix: str) -> list[Path]:
    return [
        folder / f"{prefix}_rho_wrt_center.npy",
        folder / f"{prefix}_theta_wrt_center.npy",
        folder / f"{prefix}_mask.npy",
        *(folder / f"{prefix}_input_feat_{name}.npy" for name in MASIF_FEATURES),
    ]


def preflight(args) -> list[str]:
    errors = []
    for package, module in core.PYTHON_DEPENDENCIES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"Python package missing: {package} (import name: {module})")

    for label, model_dir in (("sequence", args.seq_model_dir),
                             ("surface", args.surf_model_dir)):
        for fold in range(5):
            path = model_dir / f"fold_{fold}" / "best_model.pt"
            if not path.is_file():
                errors.append(f"Missing {label} model weight: {path}")

    required = [
        ROOT / "ITaylor_test.py",
        ROOT / "predict_stage1_logging.py",
        ROOT / "predict_stage2_surfonly_logging.py",
        ROOT / "Network_v3.py",
        ROOT / "MaSIFCore_v2.py",
        args.database_csv,
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
        *(SEQ_POOL_DIR / f"fold_{fold}_val_predictions.csv" for fold in range(5)),
        *required_masif_files(args.pmhc_masif_dir, "p1"),
        *required_masif_files(args.tcr_masif_dir, "p2"),
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"Missing required file: {path}")

    for name in ("epitope", "hla_allele", "hla_sequence", "tcra", "tcrb",
                 "cdr3a", "cdr3b"):
        if not getattr(args, name).strip():
            errors.append(f"--{name} must not be empty")
    return errors


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("ITaylor_predict")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_direct_masif(pmhc_dir: Path, tcr_dir: Path, device):
    from predict_stage2_surfonly_logging import (
        _ensure_patch_mask, _pad_or_trunc_1d_mask, _pad_or_trunc_2d,
        _safe_np_load,
    )

    np, torch = core.np, core.torch

    def load_side(folder: Path, prefix: str, patches: int):
        rho = _safe_np_load(folder / f"{prefix}_rho_wrt_center.npy").astype(np.float32)
        theta = _safe_np_load(folder / f"{prefix}_theta_wrt_center.npy").astype(np.float32)
        mask = _ensure_patch_mask(_safe_np_load(folder / f"{prefix}_mask.npy"))
        rho = _pad_or_trunc_2d(rho, patches)
        theta = _pad_or_trunc_2d(theta, patches)
        mask = _pad_or_trunc_1d_mask(mask, patches)
        result = []
        for name in MASIF_FEATURES:
            feature = _safe_np_load(folder / f"{prefix}_input_feat_{name}.npy").astype(np.float32)
            feature = _pad_or_trunc_2d(feature, patches)
            result.append({
                "input_feat": torch.from_numpy(feature).float().unsqueeze(0).to(device),
                "rho_coords": torch.from_numpy(rho).float().unsqueeze(0).to(device),
                "theta_coords": torch.from_numpy(theta).float().unsqueeze(0).to(device),
                "mask": torch.from_numpy(mask).bool().unsqueeze(0).to(device),
            })
        return result

    return {
        "phla_masifs": load_side(pmhc_dir, "p1", 13),
        "tcr_masifs": load_side(tcr_dir, "p2", 16),
    }


def predict_sequence_logit(args, predictor) -> float:
    torch = core.torch
    batch = predictor._prepare_batch_input(
        [args.epitope], [args.hla_sequence], [args.tcra], [args.tcrb])
    values = []
    for model in predictor.models:
        with torch.no_grad():
            output = model(
                peptide_emb=batch["peptide_emb"], hla_emb=batch["hla_emb"],
                tcra_emb=batch["tcra_emb"], tcrb_emb=batch["tcrb_emb"],
                peptide_mask=batch["peptide_mask"], hla_mask=batch["hla_mask"],
                tcra_mask=batch["tcra_mask"], tcrb_mask=batch["tcrb_mask"],
                mode="seq_only")
        values.append(float(output["logit"].reshape(-1)[0].detach().cpu()))
    return float(core.np.mean(values))


def predict_surface_logit(args, predictor) -> float:
    torch = core.torch
    batch = load_direct_masif(args.pmhc_masif_dir, args.tcr_masif_dir,
                              predictor.device)
    values = []
    for model in predictor.models:
        with torch.no_grad():
            output = model(
                peptide_emb=None, hla_emb=None, tcra_emb=None, tcrb_emb=None,
                peptide_mask=None, hla_mask=None, tcra_mask=None, tcrb_mask=None,
                phla_masifs=batch["phla_masifs"],
                tcr_masifs=batch["tcr_masifs"], mode="surf_only")
        values.append(float(output["logit"].reshape(-1)[0].detach().cpu()))
    return float(core.np.mean(values))


def blosum_vector(sequence: str):
    np = core.np
    indices = [core._AA2IDX[aa] for aa in sequence if aa in core._AA2IDX]
    if not indices:
        return np.zeros(20)
    vector = core._BL_MAT[indices].sum(axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def unique_pool(pool, column: str):
    sequence_hlas = {}
    for sequence, hla in zip(pool[column], pool["hla_allele"]):
        if not isinstance(sequence, str) or not sequence:
            continue
        sequence_hlas.setdefault(sequence, set())
        if isinstance(hla, str) and hla:
            sequence_hlas[sequence].add(hla)
    return list(sequence_hlas), list(sequence_hlas.values())


def stratified_summaries(similarities, hla_sets, query_hla: str):
    np = core.np
    same_mask = np.array([query_hla in values for values in hla_sets], dtype=bool)
    other_mask = np.array([any(hla != query_hla for hla in values)
                           for values in hla_sets], dtype=bool)
    same_max = float(similarities[same_mask].max()) if same_mask.any() else 0.0
    if not other_mask.any():
        return same_max, 0.0, 0.0
    other = similarities[other_mask]
    k = min(5, len(other))
    return same_max, float(other.max()), float(np.partition(other, -k)[-k:].mean())


def vector_similarity_features(pool, query: str, query_hla: str,
                               pool_column: str, output_prefix: str) -> dict:
    np = core.np
    sequences, hla_sets = unique_pool(pool, pool_column)
    if not sequences:
        same = other = top5 = 0.0
    else:
        matrix = np.stack([blosum_vector(sequence) for sequence in sequences])
        similarities = matrix @ blosum_vector(query)
        same, other, top5 = stratified_summaries(similarities, hla_sets, query_hla)
    return {
        f"{output_prefix}_same_hla_max": same,
        f"{output_prefix}_other_hla_max": other,
        f"{output_prefix}_other_hla_top5_mean": top5,
    }


def epitope_similarity_features(pool, query: str, query_hla: str) -> dict:
    from Bio.Align import PairwiseAligner
    np = core.np
    sequences, hla_sets = unique_pool(pool, "Epitope")
    if not sequences:
        same = other = top5 = 0.0
    else:
        aligner = PairwiseAligner()
        aligner.substitution_matrix = core.BLOSUM62
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        query_self = aligner.score(query, query)
        values = []
        for sequence in sequences:
            if sequence == query:
                values.append(0.0)
                continue
            denominator = max(query_self, aligner.score(sequence, sequence))
            score = aligner.score(query, sequence)
            values.append(float(score / denominator) if denominator > 0 else 0.0)
        same, other, top5 = stratified_summaries(
            np.asarray(values, dtype=float), hla_sets, query_hla)
    return {
        "sim_epitope_same_hla_max": same,
        "sim_epitope_other_hla_max": other,
        "sim_epitope_other_hla_top5_mean": top5,
    }


def build_similarity_features(args, logger) -> dict:
    pd = core.pd
    pool = pd.concat([
        pd.read_csv(SEQ_POOL_DIR / f"fold_{fold}_val_predictions.csv")
        for fold in range(5)
    ], ignore_index=True)
    pool = pool[pool["label"] == 1].copy()
    database = pd.read_csv(args.database_csv, usecols=["id", "CDR3α", "CDR3β"])
    database = database.rename(columns={"CDR3α": "cdr3a", "CDR3β": "cdr3b"})
    pool = pool.join(database.set_index("id"), on="id_tcr", how="left")

    features = epitope_similarity_features(pool, args.epitope, args.hla_allele)
    for query, column, prefix in (
        (args.cdr3a, "cdr3a", "sim_cdr3a"),
        (args.cdr3b, "cdr3b", "sim_cdr3b"),
        (args.tcra, "tcra_variable", "sim_tcra_full"),
        (args.tcrb, "tcrb_variable", "sim_tcrb_full"),
    ):
        features.update(vector_similarity_features(
            pool, query, args.hla_allele, column, prefix))
    logger.info("Similarity features computed from the positive reference pool")
    return features


def structure_features(args) -> dict:
    pd = core.pd
    params = pd.read_csv(CALIB_DIR / "structure_zscore_params.csv").set_index("feature")
    dope_mean = float(params.loc["phla_dope_per_res", "mean"])
    dope_std = float(params.loc["phla_dope_per_res", "std"])
    raw = {
        "phla_dope_per_res": (dope_mean - dope_std
                              if args.phla_dope_per_res is None
                              else args.phla_dope_per_res),
        "tcr_lDDT": args.tcr_lddt,
        "tcr_pTM": args.tcr_ptm,
        "tcr_ipTM": args.tcr_iptm,
    }
    return {
        name: (float(value) - float(params.loc[name, "mean"]))
              / float(params.loc[name, "std"])
        for name, value in raw.items()
    }


def build_fusion_row(args, sequence_logit: float, surface_logit: float,
                     logger):
    np, pd = core.np, core.pd
    seq_a, seq_b = core.load_platt("sequence")
    surf_a, surf_b = core.load_platt("surface")
    seq_cal_logit = seq_a * sequence_logit + seq_b
    surf_cal_logit = surf_a * surface_logit + surf_b
    seq_prob = float(core.sigmoid(np.asarray([seq_cal_logit]))[0])
    surf_prob = float(core.sigmoid(np.asarray([surf_cal_logit]))[0])
    entropy_seq = float(core.entropy(np.asarray([seq_prob]))[0])
    entropy_surf = float(core.entropy(np.asarray([surf_prob]))[0])
    row = {
        "sequence_logit": sequence_logit,
        "surface_logit": surface_logit,
        "seq_cal_logit": seq_cal_logit,
        "surf_cal_logit": surf_cal_logit,
        "seq_cal_prob": seq_prob,
        "surf_cal_prob": surf_prob,
        "entropy_seq": entropy_seq,
        "entropy_surf": entropy_surf,
        "prob_diff": seq_prob - surf_prob,
        "entropy_diff": entropy_seq - entropy_surf,
    }
    row.update(structure_features(args))
    row.update(build_similarity_features(args, logger))

    conflict = pd.read_csv(CALIB_DIR / "conflict_zscore_params.csv").set_index("feature")
    product = np.log(seq_prob * surf_prob + 1e-10)
    row["prob_product_log"] = (
        product - float(conflict.loc["prob_product_log", "mean"])) / float(
            conflict.loc["prob_product_log", "std"])
    signed_params = pd.read_csv(
        CALIB_DIR / "signed_product_zscore_params.csv").iloc[0]
    raw_signed = seq_cal_logit * surf_cal_logit
    compressed = np.sign(raw_signed) * np.log1p(abs(raw_signed))
    row["signed_product_z"] = (
        compressed - float(signed_params["mean"])) / float(signed_params["std"])
    return pd.DataFrame([row])


def run(args) -> float:
    core.import_runtime_dependencies()
    logger = configure_logger()
    from predict_stage1_logging import Stage1Predictor
    from predict_stage2_surfonly_logging import Stage2SurfOnlyPredictor

    logger.info("Loading sequence ensemble")
    sequence_predictor = Stage1Predictor(
        model_dir=str(args.seq_model_dir), device=args.seq_device,
        esm_model_name=args.esm_model, logger=logger)
    logger.info("Loading surface ensemble")
    surface_predictor = Stage2SurfOnlyPredictor(
        model_dir=str(args.surf_model_dir), imfp_dir=str(ROOT),
        device=args.surf_device, logger=logger)
    sequence_logit = predict_sequence_logit(args, sequence_predictor)
    surface_logit = predict_surface_logit(args, surface_predictor)
    logger.info("Sequence logit=%.6f, surface logit=%.6f",
                sequence_logit, surface_logit)
    frame = build_fusion_row(args, sequence_logit, surface_logit, logger)
    probability, _, _, _ = core.apply_learned_fusion(frame, logger)
    return float(probability[0])


def main() -> int:
    try:
        args = normalize_args(build_parser().parse_args())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    errors = preflight(args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.preflight_only:
        print("Preflight passed.", file=sys.stderr)
        return 0
    try:
        probability = run(args)
    except Exception as exc:
        print(f"ERROR: prediction failed: {exc}", file=sys.stderr)
        return 3
    print(f"ITayor_score={probability:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
