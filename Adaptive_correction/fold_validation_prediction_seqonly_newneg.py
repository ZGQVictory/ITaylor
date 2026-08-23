#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fold_validation_prediction_seqonly_v2.py

Uses predict_stage1_logging.py's ESM generation logic instead of preloading embeddings.
More memory efficient for large-scale validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class FoldSplitConfig:
    data_dir: str
    neg_ratio: int
    n_folds: int
    random_seed: int
    neg_csv_path: str
    pos_csv_path: Optional[str]


def _load_fold_split_config(model_root: Path) -> FoldSplitConfig:
    cfg_path = model_root / "config.json"
    if not cfg_path.exists():
        cfg_path = model_root / "fold_0" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_root} or {model_root}/fold_0")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    required = ["data_dir", "neg_ratio", "n_folds", "random_seed"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"config.json missing keys {missing}: {cfg_path}")

    neg_csv_path = cfg.get("neg_csv_path")
    if not neg_csv_path:
        data_dir = Path(cfg["data_dir"])
        neg_csv_path = str(data_dir.parent / "outputs_split" / "training_negative_clear_peplen7_merged.csv")

    return FoldSplitConfig(
        data_dir=str(cfg["data_dir"]),
        neg_ratio=int(cfg["neg_ratio"]),
        n_folds=int(cfg["n_folds"]),
        random_seed=int(cfg["random_seed"]),
        neg_csv_path=str(neg_csv_path),
        pos_csv_path=str(cfg["pos_csv_path"]) if "pos_csv_path" in cfg and cfg["pos_csv_path"] is not None else None,
    )


def _resolve_model_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "fold_0" / "best_model.pt").exists() or (path / "config.json").exists():
        return path
    raise FileNotFoundError(f"Could not find model directory at {path}")


def _parse_folds_arg(folds: str, n_folds: int) -> List[int]:
    folds = folds.strip()
    if folds.lower() in {"all", "*"}:
        return list(range(n_folds))
    out: List[int] = []
    for part in folds.split(","):
        part = part.strip()
        if not part:
            continue
        f = int(part)
        if f < 0 or f >= n_folds:
            raise ValueError(f"Fold must be in range [0, {n_folds - 1}], got {f}")
        out.append(f)
    if not out:
        raise ValueError("No folds specified")
    return sorted(set(out))


def _make_splits(n_folds: int, random_seed: int, labels: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(labels, dtype=int)
    idx = np.arange(len(y))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in kfold.split(idx, y)]


def _import_training_dataset_module(module_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_stage1_logging_oldneg", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_path_maybe_relative_to_repo(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _default_cache_path(out_dir: Path, split_cfg: FoldSplitConfig) -> Path:
    mode = "strictpos" if split_cfg.pos_csv_path else "oldpos"
    tag = f"dataset_cache_seqonly_{mode}_neg{split_cfg.neg_ratio}_seed{split_cfg.random_seed}"
    return out_dir / f"{tag}.npz"


def _write_cache_atomic(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache_path)


def _build_or_load_cache(*, cache_path: Path, rebuild_cache: bool, split_cfg: FoldSplitConfig) -> dict:
    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    # Build sample list without loading ESM embeddings
    import pandas as pd
    from collections import defaultdict

    neg_csv_path = _resolve_path_maybe_relative_to_repo(split_cfg.neg_csv_path)
    neg_df = pd.read_csv(neg_csv_path)

    pos_csv_path = split_cfg.pos_csv_path or str(REPO_ROOT / "data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv")
    pos_df = pd.read_csv(pos_csv_path)
    pos_sample_ids = pos_df['id'].values

    # Build sample list (same logic as pHLATCRDataset._build_sample_list)
    samples = []
    np.random.seed(split_cfg.random_seed)

    neg_is_newneg = "id_hla" in neg_df.columns and "id_epitope" in neg_df.columns

    neg_id_to_csv_indices = defaultdict(list)
    if neg_is_newneg:
        for csv_idx, row_id in enumerate(neg_df['id_hla'].values):
            neg_id_to_csv_indices[row_id].append(csv_idx)
    else:
        for csv_idx, row_id in enumerate(neg_df['id'].values):
            neg_id_to_csv_indices[row_id].append(csv_idx)

    for pos_idx, pos_id in enumerate(pos_sample_ids):
        samples.append({'sample_id': pos_id, 'is_positive': True, 'pos_idx': pos_idx, 'label': 1})

        neg_csv_indices = neg_id_to_csv_indices.get(pos_id, [])
        if len(neg_csv_indices) > 0:
            if len(neg_csv_indices) >= split_cfg.neg_ratio:
                selected_csv_indices = np.random.choice(neg_csv_indices, size=split_cfg.neg_ratio, replace=False)
            else:
                selected_csv_indices = neg_csv_indices

            for csv_idx in selected_csv_indices:
                if neg_is_newneg:
                    neg_id = int(neg_df.iloc[csv_idx]['id_hla'])
                else:
                    neg_id = int(neg_df.iloc[csv_idx]['id'])
                neg_id_tcr = int(neg_df.iloc[csv_idx]['id_tcr'])
                samples.append({'sample_id': neg_id, 'is_positive': False, 'csv_idx': csv_idx, 'id_tcr': neg_id_tcr, 'label': 0})

    np.random.shuffle(samples)

    labels = [s['label'] for s in samples]
    splits = _make_splits(n_folds=split_cfg.n_folds, random_seed=split_cfg.random_seed, labels=labels)

    fold_id = np.full((len(samples),), fill_value=-1, dtype=np.int16)
    for f, (_, val_idx) in enumerate(splits):
        fold_id[np.asarray(val_idx, dtype=int)] = np.int16(f)

    if (fold_id < 0).any():
        raise RuntimeError(f"Cache build error: {int((fold_id < 0).sum())} samples not assigned to any fold")

    payload = {
        "version": np.asarray([1], dtype=np.int16),
        "n_folds": np.asarray([split_cfg.n_folds], dtype=np.int16),
        "random_seed": np.asarray([split_cfg.random_seed], dtype=np.int64),
        "neg_ratio": np.asarray([split_cfg.neg_ratio], dtype=np.int64),
        "sample_id": np.asarray([int(s["sample_id"]) for s in samples], dtype=np.int64),
        "id_tcr": np.asarray([int(s.get("id_tcr", s["sample_id"])) for s in samples], dtype=np.int64),
        "label": np.asarray([int(s["label"]) for s in samples], dtype=np.int8),
        "csv_idx": np.asarray([int(s.get("csv_idx", -1)) for s in samples], dtype=np.int64),
        "fold_id": fold_id,
    }
    _write_cache_atomic(cache_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold-wise validation prediction using predict_stage1_logging ESM generation")
    parser.add_argument("--model_root", type=str, required=True, help="Path containing fold models")
    parser.add_argument("--data_dir_override", type=str, default=None, help="Override data_dir from config")
    parser.add_argument("--pos_csv_path", type=str, default=None, help="Positive metadata CSV path")
    parser.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "analysis_code" / "fold_calibration_seqonly"))
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--prepare_cache_only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--folds", type=str, default="all", help="You can set 0/0,1,2")
    parser.add_argument("--on_missing_positive_metadata", type=str, default="fill_nan", choices=["error", "skip", "fill_nan"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--esm_model_name", type=str, default="esmc_600m")
    args = parser.parse_args()

    model_root = _resolve_model_root(Path(args.model_root))
    split_cfg = _load_fold_split_config(model_root)

    if args.data_dir_override:
        split_cfg = FoldSplitConfig(
            data_dir=args.data_dir_override,
            neg_ratio=split_cfg.neg_ratio,
            n_folds=split_cfg.n_folds,
            random_seed=split_cfg.random_seed,
            neg_csv_path=split_cfg.neg_csv_path,
            pos_csv_path=split_cfg.pos_csv_path,
        )

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        device = "cuda:0"

    folds = _parse_folds_arg(args.folds, split_cfg.n_folds)

    import pandas as pd
    pos_csv_arg = args.pos_csv_path or split_cfg.pos_csv_path or str(REPO_ROOT / "data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv")
    pos_df = pd.read_csv(pos_csv_arg)
    if pos_df["id"].duplicated().any():
        pos_df = pos_df.drop_duplicates(subset=["id"], keep="first").copy()
    pos_by_id: Dict[int, dict] = pos_df.set_index("id").to_dict(orient="index")

    neg_csv_path = _resolve_path_maybe_relative_to_repo(split_cfg.neg_csv_path)
    neg_df = pd.read_csv(neg_csv_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path) if args.cache_path else _default_cache_path(out_dir, split_cfg)
    cache = _build_or_load_cache(cache_path=cache_path, rebuild_cache=args.rebuild_cache, split_cfg=split_cfg)

    print(f"[OK] cache ready: {cache_path}")
    if args.prepare_cache_only:
        return 0

    # Import predictor
    from predict_stage1_logging import Stage1Predictor

    required_cols = ["id", "id_hla", "id_epitope", "hla_allele", "Epitope", "hla_alpha123_mature", "tcra_variable", "tcrb_variable", "label", "id_tcr"]

    for fold in folds:
        print(f"\n[Fold {fold}] Starting prediction...")

        # Load single fold model
        predictor = Stage1Predictor(model_dir=str(model_root), device=device, esm_model_name=args.esm_model_name)
        # Keep only the current fold model (not ensemble)
        predictor.models = [predictor.models[fold]]

        fold_id = cache["fold_id"].astype(int)
        val_indices = np.where(fold_id == int(fold))[0].astype(int)
        if args.max_samples > 0:
            val_indices = val_indices[:args.max_samples]

        sample_id_all = cache["sample_id"].astype(int)
        id_tcr_all = cache["id_tcr"].astype(int)
        label_all = cache["label"].astype(int)
        csv_idx_all = cache["csv_idx"].astype(int)

        # Collect sequences
        epitopes, hlas, tcras, tcrbs = [], [], [], []
        for idx in val_indices:
            sample_id = int(sample_id_all[idx])
            id_tcr = int(id_tcr_all[idx])
            csv_idx = int(csv_idx_all[idx])

            if csv_idx >= 0:
                row = neg_df.iloc[csv_idx]
                epitopes.append(str(row["Epitope"]))
                hlas.append(str(row["hla_alpha123_mature"]))
                tcras.append(str(row["tcra_variable"]))
                tcrbs.append(str(row["tcrb_variable"]))
            else:
                if sample_id not in pos_by_id:
                    epitopes.append("")
                    hlas.append("")
                    tcras.append("")
                    tcrbs.append("")
                else:
                    meta = pos_by_id[sample_id]
                    epitopes.append(str(meta["Epitope"]))
                    hlas.append(str(meta["hla_alpha123_mature"]))
                    tcras.append(str(meta["tcra_variable"]))
                    tcrbs.append(str(meta["tcrb_variable"]))

        # Batch predict with manual batching
        print(f"[Fold {fold}] Predicting {len(epitopes)} samples in batches of {args.batch_size}...")
        probs = []
        logits = []
        from tqdm import tqdm
        import torch

        for start in tqdm(range(0, len(epitopes), args.batch_size), disable=args.no_progress):
            end = min(start + args.batch_size, len(epitopes))

            # Prepare batch input
            batch_dict = predictor._prepare_batch_input(
                epitopes=epitopes[start:end],
                hlas=hlas[start:end],
                tcras=tcras[start:end],
                tcrbs=tcrbs[start:end]
            )

            # Get model output (logits)
            with torch.no_grad():
                output = predictor.models[0](
                    peptide_emb=batch_dict['peptide_emb'],
                    hla_emb=batch_dict['hla_emb'],
                    tcra_emb=batch_dict['tcra_emb'],
                    tcrb_emb=batch_dict['tcrb_emb'],
                    peptide_mask=batch_dict['peptide_mask'],
                    hla_mask=batch_dict['hla_mask'],
                    tcra_mask=batch_dict['tcra_mask'],
                    tcrb_mask=batch_dict['tcrb_mask'],
                    mode="seq_only",
                )
                batch_logits = output['logit'].squeeze(-1).cpu().numpy()
                batch_probs = torch.sigmoid(output['logit'].squeeze(-1)).cpu().numpy()

            logits.extend(batch_logits.tolist())
            probs.extend(batch_probs.tolist())

        # Detect newneg format
        neg_is_newneg = "id_hla" in neg_df.columns and "id_epitope" in neg_df.columns

        # Build output rows
        rows: List[dict] = []
        skipped_missing_meta = 0
        for i, (idx, prob, logit) in enumerate(zip(val_indices.tolist(), probs, logits)):
            csv_idx = int(csv_idx_all[idx])
            sample_id = int(sample_id_all[idx])
            id_tcr = int(id_tcr_all[idx])
            label = int(label_all[idx])

            if csv_idx >= 0:
                if csv_idx >= len(neg_df):
                    continue
                row = neg_df.iloc[csv_idx].to_dict()
                if neg_is_newneg:
                    base = {
                        "id": row.get("id_hla"),
                        "id_hla": row.get("id_hla"),
                        "id_epitope": row.get("id_epitope"),
                        "hla_allele": row.get("hla_allele"),
                        "Epitope": row.get("Epitope"),
                        "hla_alpha123_mature": row.get("hla_alpha123_mature"),
                        "tcra_variable": row.get("tcra_variable"),
                        "tcrb_variable": row.get("tcrb_variable"),
                        "label": label,
                        "id_tcr": row.get("id_tcr"),
                    }
                else:
                    base = {k: row.get(k) for k in required_cols}
                    base["label"] = label
            else:
                if sample_id not in pos_by_id:
                    if args.on_missing_positive_metadata == "skip":
                        skipped_missing_meta += 1
                        continue
                    if args.on_missing_positive_metadata == "error":
                        raise KeyError(f"Positive metadata missing for id={sample_id}")
                    meta = {}
                else:
                    meta = dict(pos_by_id[sample_id])
                base = {
                    "id": sample_id,
                    "id_hla": sample_id,
                    "id_epitope": sample_id,
                    "hla_allele": meta.get("hla_allele"),
                    "Epitope": meta.get("Epitope"),
                    "hla_alpha123_mature": meta.get("hla_alpha123_mature"),
                    "tcra_variable": meta.get("tcra_variable"),
                    "tcrb_variable": meta.get("tcrb_variable"),
                    "label": label,
                    "id_tcr": id_tcr,
                }

            base["logit"] = float(logit)
            base["prediction"] = float(prob)
            rows.append(base)

        out_df = pd.DataFrame(rows, columns=required_cols + ["logit", "prediction"])
        out_path = out_dir / f"fold_{fold}_val_predictions.csv"
        out_df.to_csv(out_path, index=False)
        extra = f", skipped_missing_pos_meta={skipped_missing_meta}" if skipped_missing_meta else ""
        print(f"[OK] fold {fold}: saved {len(out_df)} rows -> {out_path}{extra}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

