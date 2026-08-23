#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fold_validation_prediction.py

Goal
----
For each fold (0-4):
  1) Predict ONLY on that fold's validation/test indices.
  2) Save a per-fold CSV with:
       id,hla_allele,Epitope,hla_alpha123_mature,tcra_variable,tcrb_variable,label,id_tcr,prediction

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
    # `analysis_code/fold_validation_prediction.py` -> repo root is parent of analysis_code
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_training_dataset_module(module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_stage2_surfonly_logging_oldversion", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@dataclass(frozen=True)
class FoldSplitConfig:
    imfp_dir: str
    neg_ratio: int
    n_folds: int
    random_seed: int
    pmhc_P: int
    tcr_P: int
    neg_csv_path: str
    pos_csv_path: Optional[str]
    foldtype: str


def _load_fold_split_config(model_root: Path) -> FoldSplitConfig:
    cfg_path = model_root / "fold_0" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    required = ["imfp_dir", "neg_ratio", "n_folds", "random_seed", "pmhc_P", "tcr_P", "neg_csv_path", "foldtype"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"config.json missing keys {missing}: {cfg_path}")
    return FoldSplitConfig(
        imfp_dir=str(cfg["imfp_dir"]),
        neg_ratio=int(cfg["neg_ratio"]),
        n_folds=int(cfg["n_folds"]),
        random_seed=int(cfg["random_seed"]),
        pmhc_P=int(cfg["pmhc_P"]),
        tcr_P=int(cfg["tcr_P"]),
        neg_csv_path=str(cfg["neg_csv_path"]),
        pos_csv_path=str(cfg["pos_csv_path"]) if "pos_csv_path" in cfg and cfg["pos_csv_path"] is not None else None,
        foldtype=str(cfg.get("foldtype", "StratifiedKFold")),
    )


def _resolve_model_root(path: Path) -> Path:
    path = path.expanduser().resolve()

    def _has_fold0_best_weights(p: Path) -> bool:
        if (p / "fold_0" / "best_model.pt").exists():
            return True
        # Finetune layouts: fold_0/{stage}/best*.pt
        for stage in ("stage1", "stage2", "stage3"):
            if (p / "fold_0" / stage / "best_model.pt").exists():
                return True
            for metric in ("auroc", "auprc"):
                if (p / "fold_0" / stage / f"best_{metric}_model.pt").exists():
                    return True
        return False

    if _has_fold0_best_weights(path):
        return path

    # Common alternate layouts:
    candidates = [
        # "runs/stage2_surfnew/neg_ratio10/neg_ratio_10"
        path / "stage2_surfnew" / "neg_ratio10" / "neg_ratio_10",
        path / "stage2_surfnew" / "neg_ratio5" / "neg_ratio_5",
        path / "stage2_surfnew" / "neg_ratio1" / "neg_ratio_1",
        # "runs/stage2_surfnew/neg_ratio_10" (legacy)
        path / "stage2_surfnew" / "neg_ratio_10",
        path / "stage2_surfnew" / "neg_ratio_5",
        path / "stage2_surfnew" / "neg_ratio_1",
        # "runs/stage2_surfnew/neg_ratio10" (needs trailing neg_ratio_10)
        path / "neg_ratio10" / "neg_ratio_10",
        path / "neg_ratio5" / "neg_ratio_5",
        path / "neg_ratio1" / "neg_ratio_1",
        # "runs/.../neg_ratio_10"
        path / "neg_ratio_10",
        path / "neg_ratio_5",
        path / "neg_ratio_1",
    ]
    for cand in candidates:
        if _has_fold0_best_weights(cand):
            return cand

    raise FileNotFoundError(
        "Could not find fold model directory. Expected either:\n"
        f"  - {path}/fold_0/best_model.pt\n"
        f"  - {path}/fold_0/stageX/best_model.pt (finetune)\n"
        "or a common subpath like stage2_surfnew/neg_ratio10/neg_ratio_10/"
    )


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


def _make_splits(
    foldtype: str,
    n_folds: int,
    random_seed: int,
    labels: Sequence[int],
    groups: Sequence[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    foldtype = str(foldtype)
    y = np.asarray(labels, dtype=int)
    idx = np.arange(len(y))
    if foldtype == "StratifiedGroupKFold":
        try:
            from sklearn.model_selection import StratifiedGroupKFold  # type: ignore

            g = np.asarray(groups, dtype=int)
            kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            return [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in kfold.split(idx, y, g)]
        except ImportError:
            foldtype = "StratifiedKFold"

    if foldtype != "StratifiedKFold":
        raise ValueError(f"Unsupported foldtype={foldtype!r}; expected StratifiedKFold or StratifiedGroupKFold")

    from sklearn.model_selection import StratifiedKFold  # type: ignore

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return [(np.asarray(tr, dtype=int), np.asarray(te, dtype=int)) for tr, te in kfold.split(idx, y)]


def _load_pos_metadata(pos_csv_path: Path) -> "pd.DataFrame":
    import pandas as pd

    if not pos_csv_path.exists():
        raise FileNotFoundError(f"Positive CSV not found: {pos_csv_path}")
    df = pd.read_csv(pos_csv_path)
    if "id" not in df.columns:
        raise ValueError(f"Positive CSV must contain 'id' column: {pos_csv_path}")
    # De-duplicate by id if needed
    if df["id"].duplicated().any():
        df = df.drop_duplicates(subset=["id"], keep="first").copy()
    return df


def _load_pos_metadata_multi(pos_csv_paths: str) -> "pd.DataFrame":
    """
    Load one or multiple positive metadata CSVs.

    Accepts a single path or a comma-separated list. Rows are de-duplicated by `id`
    (first occurrence wins).
    """
    import pandas as pd

    paths = [p.strip() for p in str(pos_csv_paths).split(",") if p.strip()]
    if not paths:
        raise ValueError("Empty --pos_csv_path")

    dfs = []
    for p in paths:
        dfs.append(_load_pos_metadata(Path(p)))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    if df["id"].duplicated().any():
        df = df.drop_duplicates(subset=["id"], keep="first").copy()
    return df


def _load_single_fold_model(model_root: Path, fold: int, device: str,
                            finetune_stage: Optional[str] = None,
                            finetune_metric: str = "best"):
    import torch
    from Network_v3 import Network

    fold_dir = model_root / f"fold_{fold}"
    if finetune_stage is not None:
        stage_dir = fold_dir / finetune_stage
        model_path = stage_dir / ("best_model.pt" if finetune_metric == "best"
                                  else f"best_{finetune_metric}_model.pt")
        cfg_path = fold_dir / "config.json"
    else:
        model_path = fold_dir / "best_model.pt"
        cfg_path = fold_dir / "config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model = Network(
        hid=256,
        seq_nhead=8,
        seq_dropout=0.1,
        phla_seq_layers=1,
        tcr_seq_layers=1,
        n_thetas=int(cfg["n_thetas"]),
        n_rhos=int(cfg["n_rhos"]),
        n_rotations=int(cfg["n_rotations"]),
        max_rho_phla=float(cfg["max_rho_phla"]),
        max_rho_tcr=float(cfg["max_rho_tcr"]),
        joint_nhead=int(cfg["joint_nhead"]),
        joint_layers=int(cfg["joint_layers"]),
        joint_dropout=float(cfg["joint_dropout"]),
        ff_dim_scale=float(cfg["ff_dim_scale"]),
        num_feature_type=int(cfg["num_feature_type"]),
        ft_nhead=int(cfg["ft_nhead"]),
        ft_layers=int(cfg["ft_layers"]),
        ft_dropout=float(cfg["ft_dropout"]),
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(torch.device(device))
    return model


def _predict_probs_surfonly(
    *,
    model,
    pmhc_ids: Sequence[int],
    tcr_ids: Sequence[int],
    imfp_dir: Path,
    device: str,
    pmhc_P: int,
    tcr_P: int,
    batch_size: int,
    show_progress: bool,
) -> Tuple[List[float], List[float]]:
    import torch
    from tqdm import tqdm

    from predict_stage2_surfonly_logging import load_masif_data

    if len(pmhc_ids) != len(tcr_ids):
        raise ValueError("pmhc_ids and tcr_ids must have the same length")

    torch_device = torch.device(device)
    probs: List[float] = []
    logits: List[float] = []

    num_feature_type = 5  # fixed in load_masif_data/predictor

    iterable = range(0, len(pmhc_ids), batch_size)
    if show_progress:
        iterable = tqdm(iterable, desc="Predict batches", unit="batch")

    with torch.no_grad():
        for start in iterable:
            end = min(start + batch_size, len(pmhc_ids))
            batch_pmhc = pmhc_ids[start:end]
            batch_tcr = tcr_ids[start:end]

            phla_masifs_batch: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(num_feature_type)]
            tcr_masifs_batch: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(num_feature_type)]

            for i in range(len(batch_pmhc)):
                phla_masifs, tcr_masifs = load_masif_data(
                    imfp_dir=imfp_dir,
                    pmhc_id=int(batch_pmhc[i]),
                    tcr_id=int(batch_tcr[i]),
                    pmhc_P=pmhc_P,
                    tcr_P=tcr_P,
                    num_feature_type=num_feature_type,
                )
                for feat_idx in range(num_feature_type):
                    phla_masifs_batch[feat_idx].append(phla_masifs[feat_idx])
                    tcr_masifs_batch[feat_idx].append(tcr_masifs[feat_idx])

            phla_masifs_stacked: List[Dict[str, torch.Tensor]] = []
            tcr_masifs_stacked: List[Dict[str, torch.Tensor]] = []

            for feat_idx in range(num_feature_type):
                phla_masifs_stacked.append(
                    {
                        "input_feat": torch.stack([d["input_feat"] for d in phla_masifs_batch[feat_idx]], dim=0).to(
                            torch_device
                        ),
                        "rho_coords": torch.stack([d["rho_coords"] for d in phla_masifs_batch[feat_idx]], dim=0).to(
                            torch_device
                        ),
                        "theta_coords": torch.stack(
                            [d["theta_coords"] for d in phla_masifs_batch[feat_idx]], dim=0
                        ).to(torch_device),
                        "mask": torch.stack([d["mask"] for d in phla_masifs_batch[feat_idx]], dim=0).to(torch_device),
                    }
                )
                tcr_masifs_stacked.append(
                    {
                        "input_feat": torch.stack([d["input_feat"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(
                            torch_device
                        ),
                        "rho_coords": torch.stack([d["rho_coords"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(
                            torch_device
                        ),
                        "theta_coords": torch.stack([d["theta_coords"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(
                            torch_device
                        ),
                        "mask": torch.stack([d["mask"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(torch_device),
                    }
                )

            output = model(
                peptide_emb=None,
                hla_emb=None,
                tcra_emb=None,
                tcrb_emb=None,
                peptide_mask=None,
                hla_mask=None,
                tcra_mask=None,
                tcrb_mask=None,
                phla_masifs=phla_masifs_stacked,
                tcr_masifs=tcr_masifs_stacked,
                mode="surf_only",
            )
            logit = output["logit"].squeeze(-1)
            batch_logits = logit.detach().cpu().numpy().astype(float).tolist()
            batch_probs = torch.sigmoid(logit).detach().cpu().numpy().astype(float).tolist()
            logits.extend(batch_logits)
            probs.extend(batch_probs)

            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    return probs, logits


def _resolve_path_maybe_relative_to_repo(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _resolve_split_config_root(model_root: Path, finetune_stage: Optional[str]) -> Path:
    """
    Resolve which run directory's config.json should be used for dataset/split reconstruction.

    In finetune runs, fold weights typically live in `fold_X/{stage}/best*.pt` and the
    fold split/dataset config should match the *pretrain/base* run.
    """
    if finetune_stage is None:
        return model_root

    finetune_cfg_path = model_root / "fold_0" / "config.json"
    if finetune_cfg_path.exists():
        try:
            finetune_cfg = json.loads(finetune_cfg_path.read_text(encoding="utf-8"))
            pretrain_dir = finetune_cfg.get("pretrain_dir")
            if pretrain_dir:
                return _resolve_model_root(_resolve_path_maybe_relative_to_repo(str(pretrain_dir)))
        except Exception:
            pass

    # Fallback: default base run used by surfnew finetune experiments.
    fallback = REPO_ROOT / "runs" / "stage2_surfnew" / "neg_ratio10" / "neg_ratio_10"
    if (fallback / "fold_0" / "config.json").exists():
        return fallback

    return model_root


def _default_cache_path(out_dir: Path, split_cfg: FoldSplitConfig) -> Path:
    safe_ft = split_cfg.foldtype.replace("/", "_")
    mode = "strictpos" if split_cfg.pos_csv_path else "oldpos"
    tag = f"dataset_cache_surfonly_{mode}_neg{split_cfg.neg_ratio}_seed{split_cfg.random_seed}_{safe_ft}"
    if split_cfg.pos_csv_path:
        tag += f"_poscsv{Path(split_cfg.pos_csv_path).name}"
    return out_dir / f"{tag}.npz"


def _write_cache_atomic(cache_path: Path, payload: dict) -> None:
    """
    Atomic write for npz.

    Note: np.savez_compressed appends ".npz" when given a *path string* that does not
    end with ".npz". To avoid surprising renames like "*.npz.tmp.npz", we write to an
    explicit file handle.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up legacy temp file name produced by older implementation (if any).
    legacy_tmp = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")  # "*.npz.tmp.npz"
    try:
        if legacy_tmp.exists():
            legacy_tmp.unlink()
    except Exception:
        pass

    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, cache_path)


def _build_or_load_cache(
    *,
    cache_path: Path,
    rebuild_cache: bool,
    train_script_path: Path,
    split_cfg: FoldSplitConfig,
) -> dict:
    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    train_mod = _import_training_dataset_module(train_script_path)
    # Support both old and strict-pos training scripts.
    if hasattr(train_mod, "Stage2SurfDatasetStrictPos"):
        Stage2SurfDataset = getattr(train_mod, "Stage2SurfDatasetStrictPos")
    else:
        Stage2SurfDataset = getattr(train_mod, "Stage2SurfDataset")

    dataset = Stage2SurfDataset(
        imfp_dir=str(_resolve_path_maybe_relative_to_repo(split_cfg.imfp_dir)),
        neg_ratio=split_cfg.neg_ratio,
        random_seed=split_cfg.random_seed,
        pmhc_P=split_cfg.pmhc_P,
        tcr_P=split_cfg.tcr_P,
        neg_csv_path=str(_resolve_path_maybe_relative_to_repo(split_cfg.neg_csv_path)),
        **(
            {"pos_csv_path": str(_resolve_path_maybe_relative_to_repo(split_cfg.pos_csv_path))}
            if split_cfg.pos_csv_path is not None
            else {}
        ),
    )

    labels = dataset.get_labels()
    groups = dataset.get_group_ids()
    splits = _make_splits(
        foldtype=split_cfg.foldtype,
        n_folds=split_cfg.n_folds,
        random_seed=split_cfg.random_seed,
        labels=labels,
        groups=groups,
    )

    fold_id = np.full((len(dataset.samples),), fill_value=-1, dtype=np.int16)
    for f, (_, val_idx) in enumerate(splits):
        fold_id[np.asarray(val_idx, dtype=int)] = np.int16(f)

    if (fold_id < 0).any():
        missing = int((fold_id < 0).sum())
        raise RuntimeError(f"Cache build error: {missing} samples not assigned to any fold")

    samples = dataset.samples
    payload = {
        "version": np.asarray([1], dtype=np.int16),
        "n_folds": np.asarray([split_cfg.n_folds], dtype=np.int16),
        "random_seed": np.asarray([split_cfg.random_seed], dtype=np.int64),
        "neg_ratio": np.asarray([split_cfg.neg_ratio], dtype=np.int64),
        "pmhc_P": np.asarray([split_cfg.pmhc_P], dtype=np.int16),
        "tcr_P": np.asarray([split_cfg.tcr_P], dtype=np.int16),
        "pmhc_id": np.asarray([int(s["pmhc_id"]) for s in samples], dtype=np.int64),
        "tcr_id": np.asarray([int(s["tcr_id"]) for s in samples], dtype=np.int64),
        "label": np.asarray([int(s["label"]) for s in samples], dtype=np.int8),
        "csv_idx": np.asarray([int(s["csv_idx"]) for s in samples], dtype=np.int64),
        "fold_id": fold_id,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_cache_atomic(cache_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold-wise validation prediction (surf-only stage2)")
    parser.add_argument(
        "--model_root",
        type=str,
        default=str(REPO_ROOT / "runs" / "stage2_surfnew" / "neg_ratio10" / "neg_ratio_10"),
        help=(
            "Run directory containing fold_0..fold_4. "
            "For non-finetune runs: fold_X/best_model.pt; "
            "for finetune runs: fold_X/{stage}/best*.pt."
        ),
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default="auto",
        help=(
            "Training script path that defines the dataset/split logic. "
            "Use 'auto' to select based on model config (pos_csv_path => strict-pos, else oldversion)."
        ),
    )
    parser.add_argument(
        "--pos_csv_path",
        type=str,
        default=None,
        help=(
            "Positive metadata CSV path, or comma-separated list of paths. "
            "Must include: id,hla_allele,Epitope,hla_alpha123_mature,tcra_variable,tcrb_variable,label"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "analysis_code" / "fold_calibration_surfonly"),
        help="Output directory for per-fold CSVs",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to dataset/split cache .npz (default: <out_dir>/dataset_cache_*.npz)",
    )
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild cache even if it exists")
    parser.add_argument(
        "--prepare_cache_only",
        action="store_true",
        help="Only build/load cache then exit (recommended before parallel fold runs)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / etc (default: auto)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--folds", type=str, default="all", help="Comma list like '0,1,2,3,4' or 'all'")
    parser.add_argument(
        "--on_missing_positive_metadata",
        type=str,
        default="fill_nan",
        choices=["error", "skip", "fill_nan"],
        help="How to handle positive samples whose id is absent from provided positive metadata CSV(s)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, limit to first N val samples per fold (debug only)",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--finetune_stage", type=str, default=None,
                        choices=["stage1", "stage2", "stage3"],
                        help="For finetune dirs: load weights from fold_X/{stage}/")
    parser.add_argument("--finetune_metric", type=str, default="best",
                        choices=["best", "auroc", "auprc"],
                        help="For finetune dirs: which checkpoint (best/auroc/auprc)")
    args = parser.parse_args()
        
    model_root = _resolve_model_root(Path(args.model_root))
    split_cfg_root = _resolve_split_config_root(model_root, args.finetune_stage)
    if split_cfg_root != model_root:
        print(f"[INFO] finetune mode: using split/config from: {split_cfg_root}")
    split_cfg = _load_fold_split_config(split_cfg_root)

    device = args.device
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    # Normalize a few common device strings
    if device == "cuda":
        device = "cuda:0"

    # Pin CUDA device context for this process (useful when running folds in parallel).
    if torch is not None and isinstance(device, str) and device.startswith("cuda"):
        # Accept "cuda" (normalized above) or "cuda:N"
        try:
            if ":" in device:
                torch.cuda.set_device(int(device.split(":", 1)[1]))
            else:
                torch.cuda.set_device(0)
        except Exception:
            # If parsing fails, still proceed; torch.device(...) may handle it.
            pass

    folds = _parse_folds_arg(args.folds, split_cfg.n_folds)

    if str(args.train_script).strip().lower() == "auto":
        if split_cfg.pos_csv_path is not None:
            train_script_path = (REPO_ROOT / "train_stage2_surfonly_logging.py").resolve()
        else:
            train_script_path = (REPO_ROOT / "train_stage2_surfonly_logging-oldversion.py").resolve()
    else:
        train_script_path = Path(args.train_script).expanduser().resolve()

    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    # Load metadata tables
    import pandas as pd

    pos_csv_arg = args.pos_csv_path
    if pos_csv_arg is None:
        # Prefer the pos_csv_path recorded at training time (strict-pos script writes it).
        if split_cfg.pos_csv_path:
            pos_csv_arg = split_cfg.pos_csv_path
        else:
            pos_csv_arg = str(REPO_ROOT / "data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv")

    pos_df = _load_pos_metadata_multi(pos_csv_arg)
    pos_by_id: Dict[int, dict] = pos_df.set_index("id").to_dict(orient="index")
    neg_csv_path = _resolve_path_maybe_relative_to_repo(split_cfg.neg_csv_path)
    neg_df = pd.read_csv(neg_csv_path)

    required_cols = [
        "id",
        "hla_allele",
        "Epitope",
        "hla_alpha123_mature",
        "tcra_variable",
        "tcrb_variable",
        "label",
        "id_tcr",
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_path) if args.cache_path else _default_cache_path(out_dir, split_cfg)
    cache_path = cache_path.expanduser().resolve()
    cache = _build_or_load_cache(
        cache_path=cache_path,
        rebuild_cache=bool(args.rebuild_cache),
        train_script_path=train_script_path,
        split_cfg=split_cfg,
    )

    print(f"[OK] cache ready: {cache_path}")
    if args.prepare_cache_only:
        return 0

    imfp_dir = _resolve_path_maybe_relative_to_repo(split_cfg.imfp_dir)

    for fold in folds:
        fold_id = cache["fold_id"].astype(int)
        val_indices = np.where(fold_id == int(fold))[0].astype(int)
        if args.max_samples and args.max_samples > 0:
            val_indices = val_indices[: int(args.max_samples)]

        pmhc_all = cache["pmhc_id"].astype(int)
        tcr_all = cache["tcr_id"].astype(int)
        label_all = cache["label"].astype(int)
        csv_idx_all = cache["csv_idx"].astype(int)

        pmhc_ids = pmhc_all[val_indices].tolist()
        tcr_ids = tcr_all[val_indices].tolist()

        model = _load_single_fold_model(model_root, fold=fold, device=device,
                                        finetune_stage=args.finetune_stage,
                                        finetune_metric=args.finetune_metric)
        probs, logits = _predict_probs_surfonly(
            model=model,
            pmhc_ids=pmhc_ids,
            tcr_ids=tcr_ids,
            imfp_dir=imfp_dir,
            device=device,
            pmhc_P=split_cfg.pmhc_P,
            tcr_P=split_cfg.tcr_P,
            batch_size=int(args.batch_size),
            show_progress=not bool(args.no_progress),
        )

        rows: List[dict] = []
        skipped_missing_meta = 0
        for i, prob, logit in zip(val_indices.tolist(), probs, logits):
            csv_idx = int(csv_idx_all[i])
            pmhc_id = int(pmhc_all[i])
            tcr_id = int(tcr_all[i])
            label = int(label_all[i])

            if csv_idx >= 0:
                if csv_idx >= len(neg_df):
                    # Should not happen if CSV is consistent with the dataset cache,
                    # but keep this robust if files were edited after cache creation.
                    continue
                row = neg_df.iloc[csv_idx].to_dict()
                base = {k: row.get(k) for k in required_cols}
                base["label"] = label
            else:
                if pmhc_id not in pos_by_id:
                    if args.on_missing_positive_metadata == "skip":
                        skipped_missing_meta += 1
                        continue
                    if args.on_missing_positive_metadata == "error":
                        raise KeyError(
                            f"Positive metadata missing for id={pmhc_id}. "
                            f"Check --pos_csv_path ({args.pos_csv_path})"
                        )
                    meta = {}
                else:
                    meta = dict(pos_by_id[pmhc_id])
                base = {
                    "id": pmhc_id,
                    "hla_allele": meta.get("hla_allele"),
                    "Epitope": meta.get("Epitope"),
                    "hla_alpha123_mature": meta.get("hla_alpha123_mature"),
                    "tcra_variable": meta.get("tcra_variable"),
                    "tcrb_variable": meta.get("tcrb_variable"),
                    "label": label,
                    "id_tcr": tcr_id,  # positive samples use tcr_id==pmhc_id in this dataset
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
