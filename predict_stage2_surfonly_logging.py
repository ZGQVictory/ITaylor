# predict_stage2_surfonly_logging.py
# -*- coding: utf-8 -*-
"""
Stage 2 Surf-Only prediction module.

Features:
1. Provides Stage2SurfOnlyPredictor for model loading and inference.
2. Supports single-sample and batch prediction.
3. Uses a five-fold ensemble.
4. Uses structural MaSIF data only and requires no sequence input.

Example:
    from predict_stage2_surfonly_logging import Stage2SurfOnlyPredictor

    # Initialize the predictor
    predictor = Stage2SurfOnlyPredictor(
        model_dir="./runs/stage2_surf/neg_ratio_10",
        device="cuda:0"
    )

    # Predict one sample
    prob = predictor.predict_single(
        pmhc_id=1,  # pMHC ID for loading MaSIF data
        tcr_id=1    # TCR ID for loading MaSIF data
    )

    # Predict a batch
    probs = predictor.predict_batch(
        pmhc_ids=[1, 2],
        tcr_ids=[1, 2]
    )
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import the model
from Network_v3 import Network


# =========================
#      MaSIF data-loading utilities
# =========================
FEATURE_NAMES = ["charge", "ddc", "hbond", "hphob", "si"]  # Fixed order


def _safe_np_load(path: Path) -> np.ndarray:
    """Load a NumPy file with mmap for efficient access to large arrays."""
    return np.load(str(path), mmap_mode="r")


def _ensure_patch_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    Convert a mask to patch-level shape [P].

    A [P, V] mask is reduced to [P] with ``any(vertex_valid)``.
    """
    if mask_arr.ndim == 2:
        return mask_arr.astype(bool).any(axis=1)
    if mask_arr.ndim == 1:
        return mask_arr.astype(bool)
    raise ValueError(f"Unsupported mask shape {mask_arr.shape}; expected [P] or [P,V]")


def _pad_or_trunc_2d(arr: np.ndarray, target_P: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate a [P, V] array to [target_P, V].
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [P,V], got shape {arr.shape}")
    P, V = arr.shape
    if P == target_P:
        return arr
    if P > target_P:
        return arr[:target_P, :]
    # Pad missing rows
    pad_rows = target_P - P
    pad = np.full((pad_rows, V), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _pad_or_trunc_1d_mask(mask: np.ndarray, target_P: int) -> np.ndarray:
    """
    Pad or truncate a [P] mask to [target_P].
    """
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask [P], got shape {mask.shape}")
    P = mask.shape[0]
    if P == target_P:
        return mask
    if P > target_P:
        return mask[:target_P]
    pad = np.zeros((target_P - P,), dtype=bool)
    return np.concatenate([mask.astype(bool), pad], axis=0)


def load_masif_data(
    imfp_dir: Path,
    pmhc_id: int,
    tcr_id: int,
    pmhc_P: int = 13,
    tcr_P: int = 16,
    num_feature_type: int = 5,
    pmhc_override_dir: Optional[Path] = None,
) -> tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Load MaSIF data for one sample.

    Args:
        imfp_dir: Root MaSIF data directory (./data/Database_stage2/imfp).
        pmhc_id: pMHC ID
        tcr_id: TCR ID
        pmhc_P: Target number of pMHC patches after padding.
        tcr_P: Target number of TCR patches after padding.
        num_feature_type: Number of feature types; defaults to 5.
        pmhc_override_dir: Optional directory from which to load the pMHC
            surface directly for the nonered path.

    Returns:
        phla_masifs: List[Dict] with ``num_feature_type`` entries.
        tcr_masifs: List[Dict]
    """
    # Load pMHC data
    if pmhc_override_dir is not None:
        pmhc_folder = Path(pmhc_override_dir)
    else:
        pmhc_folder = imfp_dir / "train_pmhc" / f"pmhc_{pmhc_id:06d}"
    if not pmhc_folder.exists():
        raise FileNotFoundError(f"pMHC folder not found: {pmhc_folder}")

    pmhc_rho = _safe_np_load(pmhc_folder / "p1_rho_wrt_center.npy").astype(np.float32)
    pmhc_theta = _safe_np_load(pmhc_folder / "p1_theta_wrt_center.npy").astype(np.float32)
    pmhc_mask = _ensure_patch_mask(_safe_np_load(pmhc_folder / "p1_mask.npy"))

    pmhc_feats = {}
    for name in FEATURE_NAMES:
        pmhc_feats[name] = _safe_np_load(pmhc_folder / f"p1_input_feat_{name}.npy").astype(np.float32)

    # Load TCR data
    tcr_folder = imfp_dir / "train_tcr" / f"tcr_{tcr_id:06d}"
    if not tcr_folder.exists():
        raise FileNotFoundError(f"TCR folder not found: {tcr_folder}")

    tcr_rho = _safe_np_load(tcr_folder / "p2_rho_wrt_center.npy").astype(np.float32)
    tcr_theta = _safe_np_load(tcr_folder / "p2_theta_wrt_center.npy").astype(np.float32)
    tcr_mask = _ensure_patch_mask(_safe_np_load(tcr_folder / "p2_mask.npy"))

    tcr_feats = {}
    for name in FEATURE_NAMES:
        tcr_feats[name] = _safe_np_load(tcr_folder / f"p2_input_feat_{name}.npy").astype(np.float32)

    # Pad to the target lengths
    pmhc_rho_padded = _pad_or_trunc_2d(pmhc_rho, pmhc_P)
    pmhc_theta_padded = _pad_or_trunc_2d(pmhc_theta, pmhc_P)
    pmhc_mask_padded = _pad_or_trunc_1d_mask(pmhc_mask, pmhc_P)

    tcr_rho_padded = _pad_or_trunc_2d(tcr_rho, tcr_P)
    tcr_theta_padded = _pad_or_trunc_2d(tcr_theta, tcr_P)
    tcr_mask_padded = _pad_or_trunc_1d_mask(tcr_mask, tcr_P)

    # Build one phla_masifs and tcr_masifs dictionary per feature type
    phla_masifs = []
    tcr_masifs = []

    for name in FEATURE_NAMES:
        # pMHC
        pmhc_feat_padded = _pad_or_trunc_2d(pmhc_feats[name], pmhc_P)
        phla_masifs.append({
            "input_feat": torch.from_numpy(pmhc_feat_padded).float(),      # [P, V]
            "rho_coords": torch.from_numpy(pmhc_rho_padded).float(),       # [P, V]
            "theta_coords": torch.from_numpy(pmhc_theta_padded).float(),   # [P, V]
            "mask": torch.from_numpy(pmhc_mask_padded).bool(),             # [P]
        })

        # TCR
        tcr_feat_padded = _pad_or_trunc_2d(tcr_feats[name], tcr_P)
        tcr_masifs.append({
            "input_feat": torch.from_numpy(tcr_feat_padded).float(),       # [P, V]
            "rho_coords": torch.from_numpy(tcr_rho_padded).float(),        # [P, V]
            "theta_coords": torch.from_numpy(tcr_theta_padded).float(),    # [P, V]
            "mask": torch.from_numpy(tcr_mask_padded).bool(),              # [P]
        })

    return phla_masifs, tcr_masifs


# =========================
#      Logging utilities
# =========================
def setup_logging(name: str = "Stage2SurfOnlyPredictor") -> logging.Logger:
    """Configure and return a console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(sh)

    return logger


# =========================
#      Stage 2 Surf-Only predictor
# =========================
class Stage2SurfOnlyPredictor:
    """
    Load a five-fold ensemble and predict from structural information only.
    """

    def __init__(
        self,
        model_dir: str,
        imfp_dir: str = "./data/Database_stage2/imfp",
        device: str = "cuda:0",
        logger: Optional[logging.Logger] = None,
        finetune_stage: Optional[str] = None,
        finetune_metric: str = "best",
    ):
        """
        Initialize the predictor.

        Args:
            model_dir: Model directory containing fold_0 through fold_4.
            imfp_dir: Root MaSIF data directory.
            device: Inference device.
            logger: Optional logger.
        """
        self.model_dir = Path(model_dir)
        self.imfp_dir = Path(imfp_dir)
        self.device = torch.device(device)
        self.logger = logger if logger is not None else setup_logging()
        self.finetune_stage = finetune_stage    # e.g. "stage1"/"stage2"/"stage3" or None
        self.finetune_metric = finetune_metric  # "best"/"auroc"/"auprc"

        # Model configuration matching training
        # Sequence parameters are unused in surf_only mode but required by Network
        self.hid = 256
        self.seq_nhead = 8
        self.seq_dropout = 0.1
        self.phla_seq_layers = 1
        self.tcr_seq_layers = 1

        # MaSIF parameters
        self.num_feature_type = 5
        self.n_thetas = 16
        self.n_rhos = 5
        self.n_rotations = 16
        self.max_rho_phla = 12.0
        self.max_rho_tcr = 12.0

        # Joint encoder params
        self.joint_nhead = 10
        self.joint_layers = 1
        self.joint_dropout = 0.1
        self.ff_dim_scale = 4.0

        # Feature-type level params
        self.ft_nhead = 10
        self.ft_layers = 1
        self.ft_dropout = 0.1

        # MaSIF padding targets
        self.pmhc_P = 13
        self.tcr_P = 16

        # Load models
        self.logger.info("=" * 60)
        self.logger.info("Initializing Stage2SurfOnlyPredictor...")
        self.logger.info(f"Model directory: {model_dir}")
        self.logger.info(f"MaSIF data directory: {imfp_dir}")
        self.logger.info(f"Device: {device}")

        # Set the CUDA device context
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()

        # Load five fold models
        self.models = self._load_fold_models()
        self.logger.info("Stage2SurfOnlyPredictor initialized successfully")
        self.logger.info("=" * 60)

    def _load_fold_models(self) -> List[nn.Module]:
        """Load the five fold models."""
        models = []
        self.logger.info("Loading 5-fold models...")

        for fold in range(5):
            fold_dir = self.model_dir / f"fold_{fold}"

            # Resolve the model path
            if self.finetune_stage:
                # Fine-tuned layout: fold_X/stageY/{best_metric}_model.pt
                stage_dir = fold_dir / self.finetune_stage
                if self.finetune_metric == "best":
                    model_path = stage_dir / "best_model.pt"
                else:
                    model_path = stage_dir / f"best_{self.finetune_metric}_model.pt"
            else:
                # Standard layout: fold_X/best_model.pt
                model_path = fold_dir / "best_model.pt"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.logger.info(f"  Loading fold {fold} from: {model_path}")

            # Create the model
            model = Network(
                hid=self.hid, seq_nhead=self.seq_nhead, seq_dropout=self.seq_dropout,
                phla_seq_layers=self.phla_seq_layers, tcr_seq_layers=self.tcr_seq_layers,
                n_thetas=self.n_thetas, n_rhos=self.n_rhos, n_rotations=self.n_rotations,
                max_rho_phla=self.max_rho_phla, max_rho_tcr=self.max_rho_tcr,
                joint_nhead=self.joint_nhead, joint_layers=self.joint_layers,
                joint_dropout=self.joint_dropout, ff_dim_scale=self.ff_dim_scale,
                num_feature_type=self.num_feature_type,
                ft_nhead=self.ft_nhead, ft_layers=self.ft_layers, ft_dropout=self.ft_dropout,
            )

            # Load weights
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Log validation performance
            if 'val_auroc' in checkpoint:
                self.logger.info(f"    Epoch: {checkpoint['epoch']}, Val AUROC: {checkpoint['val_auroc']:.4f}")
            elif 'val_metrics' in checkpoint:
                self.logger.info(f"    Epoch: {checkpoint['epoch']}, Val AUROC: {checkpoint['val_metrics']['auroc']:.4f}")
            else:
                self.logger.info(f"    Epoch: {checkpoint['epoch']}")

            models.append(model)

        self.logger.info(f"  All 5 models loaded successfully")
        return models

    def _prepare_batch_input(
        self,
        pmhc_ids: List[int],
        tcr_ids: List[int]
    ) -> dict:
        """
        Prepare a structure-only batch.

        Args:
            pmhc_ids: List of pMHC IDs.
            tcr_ids: List of TCR IDs.

        Returns:
            batch_dict: Dictionary containing all model inputs.
        """
        batch_size = len(pmhc_ids)

        # Load MaSIF data
        self.logger.info(f"Loading MaSIF data for {batch_size} samples...")

        # Collect data for each feature type
        phla_masifs_batch = [[] for _ in range(self.num_feature_type)]
        tcr_masifs_batch = [[] for _ in range(self.num_feature_type)]

        for i in tqdm(range(batch_size), desc="Loading MaSIF data"):
            phla_masifs, tcr_masifs = load_masif_data(
                self.imfp_dir,
                pmhc_ids[i],
                tcr_ids[i],
                pmhc_P=self.pmhc_P,
                tcr_P=self.tcr_P,
                num_feature_type=self.num_feature_type
            )

            # Add data to the corresponding feature-type lists
            for feat_idx in range(self.num_feature_type):
                phla_masifs_batch[feat_idx].append(phla_masifs[feat_idx])
                tcr_masifs_batch[feat_idx].append(tcr_masifs[feat_idx])

        # Stack samples into a batch
        batch_dict = {}

        # Stack MaSIF data
        phla_masifs_stacked = []
        tcr_masifs_stacked = []

        for feat_idx in range(self.num_feature_type):
            # Stack pMHC
            phla_masifs_stacked.append({
                "input_feat": torch.stack([d["input_feat"] for d in phla_masifs_batch[feat_idx]], dim=0).to(self.device),
                "rho_coords": torch.stack([d["rho_coords"] for d in phla_masifs_batch[feat_idx]], dim=0).to(self.device),
                "theta_coords": torch.stack([d["theta_coords"] for d in phla_masifs_batch[feat_idx]], dim=0).to(self.device),
                "mask": torch.stack([d["mask"] for d in phla_masifs_batch[feat_idx]], dim=0).to(self.device),
            })

            # Stack TCR
            tcr_masifs_stacked.append({
                "input_feat": torch.stack([d["input_feat"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(self.device),
                "rho_coords": torch.stack([d["rho_coords"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(self.device),
                "theta_coords": torch.stack([d["theta_coords"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(self.device),
                "mask": torch.stack([d["mask"] for d in tcr_masifs_batch[feat_idx]], dim=0).to(self.device),
            })

        batch_dict['phla_masifs'] = phla_masifs_stacked
        batch_dict['tcr_masifs'] = tcr_masifs_stacked

        return batch_dict

    @torch.no_grad()
    def predict_batch(
        self,
        pmhc_ids: List[int],
        tcr_ids: List[int]
    ) -> List[float]:
        """
        Predict a batch.

        Args:
            pmhc_ids: List of pMHC IDs used to load MaSIF data.
            tcr_ids: List of TCR IDs used to load MaSIF data.

        Returns:
            predictions: List of predicted probabilities.
        """
        # Validate inputs
        assert len(pmhc_ids) == len(tcr_ids), \
            "pmhc_ids and tcr_ids must have the same length"

        batch_size = len(pmhc_ids)
        self.logger.info(f"Predicting {batch_size} samples...")

        # Prepare inputs
        start_time = time.time()
        batch_dict = self._prepare_batch_input(pmhc_ids, tcr_ids)
        self.logger.info(f"  Input preparation completed in {time.time() - start_time:.2f}s")

        # Run inference with each fold model
        start_time = time.time()
        all_logits = []  # [5, B]

        for fold_idx, model in enumerate(self.models):
            output = model(
                peptide_emb=None,      # surf_only mode requires no sequence input
                hla_emb=None,
                tcra_emb=None,
                tcrb_emb=None,
                peptide_mask=None,
                hla_mask=None,
                tcra_mask=None,
                tcrb_mask=None,
                phla_masifs=batch_dict['phla_masifs'],
                tcr_masifs=batch_dict['tcr_masifs'],
                mode="surf_only",  # Use surf_only mode
            )
            logit = output['logit'].squeeze(-1)  # [B, 1] -> [B]
            all_logits.append(logit.cpu())

        # Ensemble: average logits from five models, then apply sigmoid
        all_logits = torch.stack(all_logits, dim=0)  # [5, B]
        mean_logits = all_logits.mean(dim=0)  # [B]
        predictions = torch.sigmoid(mean_logits).numpy()  # [B]

        self.logger.info(f"  Prediction completed in {time.time() - start_time:.2f}s")

        # Clear the CUDA cache to avoid memory accumulation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return predictions.tolist()

    def predict_single(
        self,
        pmhc_id: int,
        tcr_id: int
    ) -> float:
        """
        Predict one sample.

        Args:
            pmhc_id: pMHC ID used to load MaSIF data.
            tcr_id: TCR ID used to load MaSIF data.

        Returns:
            prediction: Predicted probability.
        """
        # Reuse the batch interface with batch_size=1
        predictions = self.predict_batch(
            pmhc_ids=[pmhc_id],
            tcr_ids=[tcr_id]
        )

        return predictions[0]


# =========================
#      Example CLI
# =========================
if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2 Surf-Only Predictor Test')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory (containing fold_0 to fold_4)')
    parser.add_argument('--imfp_dir', type=str, default='./data/Database_stage2/imfp',
                        help='Path to MaSIF data directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0, cuda:1, cpu, etc.)')

    args = parser.parse_args()

    # Initialize the predictor
    predictor = Stage2SurfOnlyPredictor(
        model_dir=args.model_dir,
        imfp_dir=args.imfp_dir,
        device=args.device
    )

    # Test single-sample prediction
    print("\n" + "=" * 60)
    print("Testing single sample prediction...")

    # Example pMHC and TCR IDs loaded from the data directory
    pmhc_id = 1
    tcr_id = 1

    prob = predictor.predict_single(pmhc_id, tcr_id)
    print(f"Prediction probability: {prob:.6f}")

    # Test batch prediction
    print("\n" + "=" * 60)
    print("Testing batch prediction...")

    pmhc_ids = [1, 2, 3]
    tcr_ids = [1, 2, 3]

    probs = predictor.predict_batch(pmhc_ids, tcr_ids)
    print(f"Batch predictions: {probs}")

    print("=" * 60)
