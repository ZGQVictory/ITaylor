# -*- coding: utf-8 -*-
"""
Train_stage2_surfonly_logging.py

Stage 2 Surf-Only: Pure structure (MaSIF) training for pHLA-TCR binding.
Single fold training on specified GPU device.

Key design choices:
- Structure (MaSIF) inputs only:
  - Root: ./data/Database_stage2/imfp
  - pmhc folders: imfp/train_pmhc/pmhc_000001/, ... using p1_*.npy
  - tcr  folders: imfp/train_tcr/tcr_000001/,  ... using p2_*.npy
  - 5 feature types (default): charge, ddc, hbond, hphob, si
  - For each feature type, build dict: {input_feat, rho_coords, theta_coords, mask}
- Padding:
  - V is fixed (assumed)
  - P is padded/truncated to fixed:
      pmhc P -> 13
      tcr  P -> 16
- Model:
  - Use Network_v3(mode="surf_only") which requires phla_masifs and tcr_masifs
  - No sequence embeddings

Usage example:
  # Train fold 0 on GPU 0
  python Train_stage2_surfonly_logging.py \
    --imfp_dir ./data/Database_stage2/imfp \
    --fold 0 \
    --gpu 0 \
    --neg_ratio 10 \
    --output_dir ./runs/stage2_surf

  # Train fold 1 on GPU 1
  python Train_stage2_surfonly_logging.py \
    --imfp_dir ./data/Database_stage2/imfp \
    --fold 1 \
    --gpu 1 \
    --neg_ratio 10 \
    --output_dir ./runs/stage2_surf
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import logging
import platform
import resource
import traceback
import signal
import faulthandler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Network_v3 import Network


_BASE_DIR = Path(__file__).resolve().parent

# =========================
#      Config
# =========================
@dataclass
class Stage2SurfConfig:
    # Data
    imfp_dir: str      # ./data/Database_stage2/imfp
    neg_ratio: int = 10
    n_folds: int = 5
    random_seed: int = 42
    fold: int = 0      # Which fold to train
    gpu: int = 0       # Which GPU to use

    # MasIF padding targets
    pmhc_P: int = 13
    tcr_P: int = 16

    # Model hyperparams (sequence-related params removed, structure params kept)
    num_feature_type: int = 5
    n_thetas: int = 16
    n_rhos: int = 5
    n_rotations: int = 16
    max_rho_phla: float = 12.0
    max_rho_tcr: float = 12.0

    # Joint encoder params (for surf-only mode)
    joint_nhead: int = 10  # 修改为 10，能同时整除 joint_d=90 和 n_gauss=80
    joint_layers: int = 1
    joint_dropout: float = 0.1
    ff_dim_scale: float = 4.0

    # Feature-type level params
    ft_nhead: int = 10  # 修改为 10，能同时整除 joint_d=90 和 n_gauss=80
    ft_layers: int = 1
    ft_dropout: float = 0.1

    # Training
    batch_size: int = 64
    max_epochs: int = 30
    lr_initial: float = 5e-5
    lr_max: float = 3e-4
    lr_min: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 8
    min_delta: float = 1e-4

    # Loss
    loss_type: str = "focal"  # "bce" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Mixed precision
    use_amp: bool = True

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 2

    # Output
    output_dir: str = "./runs/stage2_surf"

    # Paths (optional override)
    neg_csv_path: Optional[str] = None

    foldtype: str = 'StratifiedKFold'

# =========================
#      Logging utils
# =========================
LOGGER = logging.getLogger("train_stage2_surf_singlefold")


def _get_cpu_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 10**8:  # macOS bytes
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0  # Linux KB
    except Exception:
        return float("nan")


def _get_gpu_mem_mb(device: str) -> Dict[str, float]:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {"alloc": 0.0, "reserved": 0.0, "max_alloc": 0.0}
    try:
        alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        max_alloc = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return {"alloc": float(alloc), "reserved": float(reserved), "max_alloc": float(max_alloc)}
    except Exception:
        return {"alloc": float("nan"), "reserved": float("nan"), "max_alloc": float("nan")}


def _log_mem(logger: logging.Logger, device: str, prefix: str = "") -> None:
    cpu_mb = _get_cpu_rss_mb()
    gpu = _get_gpu_mem_mb(device)
    logger.info(
        "%sMEM cpu_rss=%.1fMB | gpu_alloc=%.1fMB gpu_reserved=%.1fMB gpu_max_alloc=%.1fMB",
        (prefix + " " if prefix else ""),
        cpu_mb,
        gpu["alloc"],
        gpu["reserved"],
        gpu["max_alloc"],
    )


def setup_logging(output_dir: Path, fold: int) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_stage2_surf_fold_{fold}.log"

    logger = logging.getLogger("train_stage2_surf_singlefold")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logging initialized. Log file: %s", str(log_path))
    return logger


def log_environment(logger: logging.Logger, config: Stage2SurfConfig) -> None:
    logger.info("Python: %s", platform.python_version())
    logger.info("Platform: %s", platform.platform())
    logger.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            logger.info("GPU: %s | total_mem=%.1fGB", props.name, props.total_memory / (1024.0**3))
        except Exception:
            logger.info("GPU: (failed to query properties)")
    logger.info("multiprocessing start_method: %s", torch.multiprocessing.get_start_method(allow_none=True))
    _log_mem(logger, config.device, prefix="ENV")


# =========================
#      Losses & metrics
# =========================
class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification, operating on logits."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []
        self.preds = []
        self.losses = []

    def update(self, labels: torch.Tensor, preds: torch.Tensor, loss: float):
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        self.labels.extend(labels_np.tolist())
        self.preds.extend(preds_np.tolist())
        self.losses.append(float(loss))

    def compute(self) -> Dict[str, float]:
        labels = np.array(self.labels, dtype=np.float32)
        preds = np.array(self.preds, dtype=np.float32)

        if len(labels) == 0:
            LOGGER.warning("No samples available for metrics computation, returning zeros")
            return {
                "loss": float(np.mean(self.losses)) if self.losses else 0.0,
                "auroc": 0.0,
                "auprc": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "sensitivity": 0.0,
                "specificity": 0.0,
            }

        preds_bin = (preds > 0.5).astype(int)

        out = {
            "loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "auroc": roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
            "auprc": average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
            "accuracy": accuracy_score(labels, preds_bin) if len(labels) else 0.0,
            "f1": f1_score(labels, preds_bin, zero_division=0) if len(labels) else 0.0,
        }
        if len(labels) > 0 and len(np.unique(preds_bin)) > 0 and len(np.unique(labels)) > 0:
            tn, fp, fn, tp = confusion_matrix(labels, preds_bin).ravel()
            out.update({
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            })
        return out


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, lr_min, lr_max, lr_initial):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.lr_min = float(lr_min)
        self.lr_max = float(lr_max)
        self.lr_initial = float(lr_initial)
        self.current_epoch = 0

    def step(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            lr = self.lr_initial + (self.lr_max - self.lr_initial) * (self.current_epoch / max(1, self.warmup_epochs))
        else:
            denom = max(1, (self.max_epochs - self.warmup_epochs))
            progress = (self.current_epoch - self.warmup_epochs) / denom
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.current_epoch += 1
        return float(lr)


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode)
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = float(score)
            return False

        if self.mode == "min":
            improved = (self.best_score - score) > self.min_delta
        else:
            improved = (score - self.best_score) > self.min_delta

        if improved:
            self.best_score = float(score)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class LossPlotter:
    """Loss 曲线绘制器 (Surf-Only 版本)"""

    def __init__(self, save_dir: Path, fold: int, neg_ratio: int):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fold = fold
        self.neg_ratio = neg_ratio

        self.train_losses = []
        self.val_losses = []
        self.train_aurocs = []
        self.val_aurocs = []
        self.train_auprcs = []
        self.val_auprcs = []
        self.epochs = []
        self.lrs = []

    def update(self, epoch: int, train_loss: float, val_loss: float,
               train_auroc: float, val_auroc: float,
               train_auprc: float, val_auprc: float, lr: float):
        """每个 epoch 更新数据并重新绘图"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_aurocs.append(train_auroc)
        self.val_aurocs.append(val_auroc)
        self.train_auprcs.append(train_auprc)
        self.val_auprcs.append(val_auprc)
        self.lrs.append(lr)

        self._plot_curves()
        self._save_data()

    def _plot_curves(self):
        """绘制并保存 loss、AUROC 和 AUPRC 曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss 曲线
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'Stage 2 Surf-Only Fold {self.fold} Loss (Neg Ratio 1:{self.neg_ratio})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # AUROC 曲线
        axes[0, 1].plot(self.epochs, self.train_aurocs, 'b-', label='Train AUROC', linewidth=2)
        axes[0, 1].plot(self.epochs, self.val_aurocs, 'r-', label='Val AUROC', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUROC')
        axes[0, 1].set_title(f'Stage 2 Surf-Only Fold {self.fold} AUROC (Neg Ratio 1:{self.neg_ratio})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # AUPRC 曲线
        axes[1, 0].plot(self.epochs, self.train_auprcs, 'b-', label='Train AUPRC', linewidth=2)
        axes[1, 0].plot(self.epochs, self.val_auprcs, 'r-', label='Val AUPRC', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUPRC')
        axes[1, 0].set_title(f'Stage 2 Surf-Only Fold {self.fold} AUPRC (Neg Ratio 1:{self.neg_ratio})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])

        # 学习率曲线
        axes[1, 1].plot(self.epochs, self.lrs, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title(f'Stage 2 Surf-Only Fold {self.fold} Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'fold_{self.fold}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _save_data(self):
        """保存训练数据到 JSON 文件"""
        data = {
            'fold': self.fold,
            'neg_ratio': self.neg_ratio,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_aurocs': self.train_aurocs,
            'val_aurocs': self.val_aurocs,
            'train_auprcs': self.train_auprcs,
            'val_auprcs': self.val_auprcs,
            'learning_rates': self.lrs,
        }
        with open(self.save_dir / f'fold_{self.fold}_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
#      Dataset
# =========================
FEATURE_NAMES = ["charge", "ddc", "hbond", "hphob", "si"]


def _safe_np_load(path: Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")


def _ensure_patch_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 2:
        return mask_arr.astype(bool).any(axis=1)
    if mask_arr.ndim == 1:
        return mask_arr.astype(bool)
    raise ValueError(f"Unsupported mask shape {mask_arr.shape}; expected [P] or [P,V]")


class Stage2SurfDataset(Dataset):
    """
    Surf-only dataset: Returns only MaSIF structure data (no sequence embeddings).

    Negative construction follows the same CSV approach as stage2 (old-neg format):
      - id      : pmhc (peptide + hla) id
      - id_tcr  : tcr id
    """
    def __init__(
        self,
        imfp_dir: str,
        neg_ratio: int = 10,
        random_seed: int = 42,
        pmhc_P: int = 13,
        tcr_P: int = 16,
        neg_csv_path: Optional[str] = None,
        cache_size: int = 2048,
    ):
        super().__init__()
        self.imfp_dir = Path(imfp_dir)
        self.neg_ratio = int(neg_ratio)
        self.random_seed = int(random_seed)
        self.pmhc_P = int(pmhc_P)
        self.tcr_P = int(tcr_P)
        self.cache_size = int(cache_size)

        # Load positive sample IDs
        # For surf-only, we need to get sample IDs from folder names
        pmhc_folder = self.imfp_dir / "train_pmhc"
        tcr_folder = self.imfp_dir / "train_tcr"

        if not pmhc_folder.exists() or not tcr_folder.exists():
            raise FileNotFoundError(f"MaSIF folders not found: {pmhc_folder} or {tcr_folder}")

        # Extract positive IDs from folder names
        pmhc_ids = set()
        for folder in pmhc_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("pmhc_"):
                try:
                    pmhc_id = int(folder.name.split("_")[1])
                    pmhc_ids.add(pmhc_id)
                except (ValueError, IndexError):
                    continue

        tcr_ids = set()
        for folder in tcr_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("tcr_"):
                try:
                    tcr_id = int(folder.name.split("_")[1])
                    tcr_ids.add(tcr_id)
                except (ValueError, IndexError):
                    continue

        # Positive samples are those that have both pmhc and tcr folders with same ID
        self.pos_ids = sorted(list(pmhc_ids & tcr_ids))
        print(f"Found {len(self.pos_ids)} positive sample IDs")

        # Load old-neg CSV for negative sampling
        if neg_csv_path is None:
            # Try default location relative to imfp_dir
            neg_csv_path = str(self.imfp_dir.parent / "outputs_split" / "training_negative_clear_peplen7_v1.csv")
        self.neg_csv_path = Path(neg_csv_path)

        import pandas as pd
        self.neg_df = pd.read_csv(self.neg_csv_path)
        if "id" not in self.neg_df.columns or "id_tcr" not in self.neg_df.columns:
            raise ValueError(f"Negative CSV must contain columns ['id','id_tcr'], got {list(self.neg_df.columns)}")

        # Validate MaSIF folders
        print("Validating MaSIF folders...")
        self.valid_pmhc_ids, self.valid_tcr_ids = self._validate_masif_folders()
        print(f"Valid pMHC folders: {len(self.valid_pmhc_ids)}")
        print(f"Valid TCR folders: {len(self.valid_tcr_ids)}")

        # Build sample list
        self.samples = self._build_sample_list()

        # Shuffle
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(self.samples)

        # Simple LRU cache
        self._pmhc_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._tcr_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._pmhc_cache_order: List[int] = []
        self._tcr_cache_order: List[int] = []

    def _validate_masif_folders(self) -> Tuple[set, set]:
        """验证所有 MaSIF 文件夹是否存在且包含必需文件"""
        valid_pmhc_ids = set()
        valid_tcr_ids = set()

        # Validate pMHC folders
        for pmhc_id in self.pos_ids:
            folder = self.imfp_dir / "train_pmhc" / f"pmhc_{pmhc_id:06d}"
            if self._is_valid_masif_folder(folder, "p1"):
                valid_pmhc_ids.add(pmhc_id)
            else:
                print(f"  [WARNING] Invalid/empty pMHC folder: pmhc_{pmhc_id:06d}")

        # Collect all TCR IDs from positive + negative CSV
        all_tcr_ids = set(self.pos_ids)
        for tcr_id in self.neg_df["id_tcr"].values:
            all_tcr_ids.add(int(tcr_id))

        # Validate TCR folders
        for tcr_id in all_tcr_ids:
            folder = self.imfp_dir / "train_tcr" / f"tcr_{tcr_id:06d}"
            if self._is_valid_masif_folder(folder, "p2"):
                valid_tcr_ids.add(tcr_id)
            else:
                print(f"  [WARNING] Invalid/empty TCR folder: tcr_{tcr_id:06d}")

        return valid_pmhc_ids, valid_tcr_ids

    def _is_valid_masif_folder(self, folder: Path, prefix: str) -> bool:
        """检查 MaSIF 文件夹是否有效（包括 NaN 检查）"""
        if not folder.exists():
            return False

        required_files = [
            f"{prefix}_rho_wrt_center.npy",
            f"{prefix}_theta_wrt_center.npy",
            f"{prefix}_mask.npy",
        ]

        for feat_name in FEATURE_NAMES:
            required_files.append(f"{prefix}_input_feat_{feat_name}.npy")

        # 检查文件存在性和大小
        for filename in required_files:
            filepath = folder / filename
            if not filepath.exists():
                return False
            if filepath.stat().st_size < 128:
                return False

        # 检查是否包含 NaN 值
        try:
            rho = _safe_np_load(folder / f"{prefix}_rho_wrt_center.npy")
            theta = _safe_np_load(folder / f"{prefix}_theta_wrt_center.npy")

            if np.isnan(rho).any() or np.isinf(rho).any():
                print(f"  [WARNING] NaN/Inf in {folder.name}/{prefix}_rho_wrt_center.npy")
                return False

            if np.isnan(theta).any() or np.isinf(theta).any():
                print(f"  [WARNING] NaN/Inf in {folder.name}/{prefix}_theta_wrt_center.npy")
                return False

            # 检查所有特征文件
            for feat_name in FEATURE_NAMES:
                feat = _safe_np_load(folder / f"{prefix}_input_feat_{feat_name}.npy")
                if np.isnan(feat).any() or np.isinf(feat).any():
                    print(f"  [WARNING] NaN/Inf in {folder.name}/{prefix}_input_feat_{feat_name}.npy")
                    return False
        except Exception as e:
            print(f"  [WARNING] Error loading {folder.name}: {e}")
            return False

        return True

    def _build_sample_list(self) -> List[Dict]:
        samples: List[Dict] = []
        np.random.seed(self.random_seed)

        # mapping: pmhc id -> list of csv indices
        id_to_csv = defaultdict(list)
        for csv_idx, pmhc_id in enumerate(self.neg_df["id"].values):
            id_to_csv[int(pmhc_id)].append(int(csv_idx))

        skipped_pos = 0
        skipped_neg = 0

        for pos_id in self.pos_ids:
            # Check validity
            if pos_id not in self.valid_pmhc_ids or pos_id not in self.valid_tcr_ids:
                skipped_pos += 1
                continue

            # positive (pmhc_id=pos_id, tcr_id=pos_id)
            samples.append({
                "pmhc_id": int(pos_id),
                "tcr_id": int(pos_id),
                "is_positive": True,
                "label": 1,
                "csv_idx": -1
            })

            csv_list = id_to_csv.get(int(pos_id), [])
            if len(csv_list) == 0:
                continue
            if len(csv_list) >= self.neg_ratio:
                chosen = np.random.choice(csv_list, size=self.neg_ratio, replace=False)
            else:
                chosen = np.array(csv_list, dtype=int)

            for csv_idx in chosen.tolist():
                row = self.neg_df.iloc[int(csv_idx)]
                pmhc_id = int(row["id"])
                tcr_id = int(row["id_tcr"])

                if pmhc_id not in self.valid_pmhc_ids or tcr_id not in self.valid_tcr_ids:
                    skipped_neg += 1
                    continue

                samples.append({
                    "pmhc_id": pmhc_id,
                    "tcr_id": tcr_id,
                    "is_positive": False,
                    "label": 0,
                    "csv_idx": int(csv_idx)
                })

        if skipped_pos > 0 or skipped_neg > 0:
            print(f"  [INFO] Skipped samples due to invalid MaSIF folders:")
            print(f"    - Positive samples: {skipped_pos}")
            print(f"    - Negative samples: {skipped_neg}")
            print(f"    - Total valid samples: {len(samples)}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> List[int]:
        return [int(s["label"]) for s in self.samples]

    def get_group_ids(self) -> List[int]:
        return [int(s["pmhc_id"]) for s in self.samples]

    def _lru_put(self, cache: Dict[int, Dict[str, np.ndarray]], order: List[int], key: int, value: Dict[str, np.ndarray]):
        cache[key] = value
        order.append(key)
        if len(order) > self.cache_size:
            old = order.pop(0)
            cache.pop(old, None)

    def _load_pmhc_folder(self, pmhc_id: int) -> Dict[str, np.ndarray]:
        pmhc_id = int(pmhc_id)
        if pmhc_id in self._pmhc_cache:
            return self._pmhc_cache[pmhc_id]

        folder = self.imfp_dir / "train_pmhc" / f"pmhc_{pmhc_id:06d}"
        if not folder.exists():
            raise FileNotFoundError(f"Missing pmhc folder: {folder}")

        rho = _safe_np_load(folder / "p1_rho_wrt_center.npy").astype(np.float32)
        theta = _safe_np_load(folder / "p1_theta_wrt_center.npy").astype(np.float32)
        mask = _ensure_patch_mask(_safe_np_load(folder / "p1_mask.npy"))

        feats = {}
        for name in FEATURE_NAMES:
            feats[name] = _safe_np_load(folder / f"p1_input_feat_{name}.npy").astype(np.float32)

        pack = {"rho": rho, "theta": theta, "mask": mask, "feats": feats}
        self._lru_put(self._pmhc_cache, self._pmhc_cache_order, pmhc_id, pack)
        return pack

    def _load_tcr_folder(self, tcr_id: int) -> Dict[str, np.ndarray]:
        tcr_id = int(tcr_id)
        if tcr_id in self._tcr_cache:
            return self._tcr_cache[tcr_id]

        folder = self.imfp_dir / "train_tcr" / f"tcr_{tcr_id:06d}"
        if not folder.exists():
            raise FileNotFoundError(f"Missing tcr folder: {folder}")

        rho = _safe_np_load(folder / "p2_rho_wrt_center.npy").astype(np.float32)
        theta = _safe_np_load(folder / "p2_theta_wrt_center.npy").astype(np.float32)
        mask = _ensure_patch_mask(_safe_np_load(folder / "p2_mask.npy"))

        feats = {}
        for name in FEATURE_NAMES:
            feats[name] = _safe_np_load(folder / f"p2_input_feat_{name}.npy").astype(np.float32)

        pack = {"rho": rho, "theta": theta, "mask": mask, "feats": feats}
        self._lru_put(self._tcr_cache, self._tcr_cache_order, tcr_id, pack)
        return pack

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[int(idx)]
        pmhc_id = int(s["pmhc_id"])
        tcr_id = int(s["tcr_id"])

        # Load MaSIF data only (no sequence embeddings)
        pmhc_pack = self._load_pmhc_folder(pmhc_id)
        tcr_pack = self._load_tcr_folder(tcr_id)

        return {
            "pmhc_id": pmhc_id,
            "tcr_id": tcr_id,
            "label": torch.tensor(float(s["label"]), dtype=torch.float32),

            # MaSIF data
            "pmhc_rho": pmhc_pack["rho"],
            "pmhc_theta": pmhc_pack["theta"],
            "pmhc_mask": pmhc_pack["mask"],
            "pmhc_feats": pmhc_pack["feats"],

            "tcr_rho": tcr_pack["rho"],
            "tcr_theta": tcr_pack["theta"],
            "tcr_mask": tcr_pack["mask"],
            "tcr_feats": tcr_pack["feats"],
        }


# =========================
#      Collate (Surf-Only)
# =========================
def _pad_or_trunc_2d(arr: np.ndarray, target_P: int, pad_value: float = 0.0) -> np.ndarray:
    """arr: [P,V] -> [target_P,V]"""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [P,V], got shape {arr.shape}")
    P, V = arr.shape
    if P == target_P:
        return arr
    if P > target_P:
        return arr[:target_P, :]
    pad_rows = target_P - P
    pad = np.full((pad_rows, V), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _pad_or_trunc_1d_mask(mask: np.ndarray, target_P: int) -> np.ndarray:
    """mask: [P] bool -> [target_P] bool"""
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask [P], got shape {mask.shape}")
    P = mask.shape[0]
    if P == target_P:
        return mask
    if P > target_P:
        return mask[:target_P]
    pad = np.zeros((target_P - P,), dtype=bool)
    return np.concatenate([mask.astype(bool), pad], axis=0)


def collate_fn_surf(batch: List[Dict], pmhc_P: int = 13, tcr_P: int = 16, num_feature_type: int = 5) -> Dict:
    """
    Surf-only collate: Build phla_masifs/tcr_masifs for mode="surf_only".
    No sequence tensors.
    """
    B = len(batch)
    labels = []
    pmhc_ids = []
    tcr_ids = []

    for sample in batch:
        pmhc_ids.append(sample["pmhc_id"])
        tcr_ids.append(sample["tcr_id"])
        labels.append(sample["label"])

    # Build MaSIF lists
    phla_masifs: List[Dict[str, torch.Tensor]] = []
    tcr_masifs: List[Dict[str, torch.Tensor]] = []

    # Infer V dimension
    first_pmhc_feat = batch[0]["pmhc_feats"][FEATURE_NAMES[0]]
    first_tcr_feat = batch[0]["tcr_feats"][FEATURE_NAMES[0]]
    if first_pmhc_feat.ndim != 2 or first_tcr_feat.ndim != 2:
        raise ValueError("Expected input_feat_* to be 2D [P,V]")
    V_pmhc = first_pmhc_feat.shape[1]
    V_tcr = first_tcr_feat.shape[1]
    if V_pmhc != V_tcr:
        raise ValueError(f"V mismatch pmhc vs tcr: {V_pmhc} vs {V_tcr}")

    for ft_i in range(num_feature_type):
        name = FEATURE_NAMES[ft_i]

        # pmhc stack
        pmhc_input = np.zeros((B, pmhc_P, V_pmhc), dtype=np.float32)
        pmhc_rho = np.zeros((B, pmhc_P, V_pmhc), dtype=np.float32)
        pmhc_theta = np.zeros((B, pmhc_P, V_pmhc), dtype=np.float32)
        pmhc_mask = np.zeros((B, pmhc_P), dtype=bool)

        # tcr stack
        tcr_input = np.zeros((B, tcr_P, V_pmhc), dtype=np.float32)
        tcr_rho = np.zeros((B, tcr_P, V_pmhc), dtype=np.float32)
        tcr_theta = np.zeros((B, tcr_P, V_pmhc), dtype=np.float32)
        tcr_mask = np.zeros((B, tcr_P), dtype=bool)

        for b_idx, sample in enumerate(batch):
            # pmhc
            feat = sample["pmhc_feats"][name]
            rho = sample["pmhc_rho"]
            theta = sample["pmhc_theta"]
            m = sample["pmhc_mask"]
            if feat.shape[1] != V_pmhc or rho.shape[1] != V_pmhc or theta.shape[1] != V_pmhc:
                raise ValueError(f"V mismatch in pmhc sample {b_idx}")
            pmhc_input[b_idx] = _pad_or_trunc_2d(feat, pmhc_P, pad_value=0.0)
            pmhc_rho[b_idx] = _pad_or_trunc_2d(rho, pmhc_P, pad_value=0.0)
            pmhc_theta[b_idx] = _pad_or_trunc_2d(theta, pmhc_P, pad_value=0.0)
            pmhc_mask[b_idx] = _pad_or_trunc_1d_mask(m, pmhc_P)

            # tcr
            feat2 = sample["tcr_feats"][name]
            rho2 = sample["tcr_rho"]
            theta2 = sample["tcr_theta"]
            m2 = sample["tcr_mask"]
            if feat2.shape[1] != V_pmhc or rho2.shape[1] != V_pmhc or theta2.shape[1] != V_pmhc:
                raise ValueError(f"V mismatch in tcr sample {b_idx}")
            tcr_input[b_idx] = _pad_or_trunc_2d(feat2, tcr_P, pad_value=0.0)
            tcr_rho[b_idx] = _pad_or_trunc_2d(rho2, tcr_P, pad_value=0.0)
            tcr_theta[b_idx] = _pad_or_trunc_2d(theta2, tcr_P, pad_value=0.0)
            tcr_mask[b_idx] = _pad_or_trunc_1d_mask(m2, tcr_P)

        phla_masifs.append({
            "input_feat": torch.from_numpy(pmhc_input),
            "rho_coords": torch.from_numpy(pmhc_rho),
            "theta_coords": torch.from_numpy(pmhc_theta),
            "mask": torch.from_numpy(pmhc_mask),
        })
        tcr_masifs.append({
            "input_feat": torch.from_numpy(tcr_input),
            "rho_coords": torch.from_numpy(tcr_rho),
            "theta_coords": torch.from_numpy(tcr_theta),
            "mask": torch.from_numpy(tcr_mask),
        })

    return {
        "pmhc_id": pmhc_ids,
        "tcr_id": tcr_ids,
        "label": torch.stack(labels, dim=0),
        "phla_masifs": phla_masifs,
        "tcr_masifs": tcr_masifs,
    }


# =========================
#      Utility Functions
# =========================
def move_masifs_to_device(masifs: List[Dict[str, torch.Tensor]], device: str) -> List[Dict[str, torch.Tensor]]:
    """
    一次性将所有 MaSIF 数据搬到 GPU

    Args:
        masifs: List of MaSIF dictionaries containing input_feat, rho_coords, theta_coords, mask
        device: Target device (e.g., 'cuda')

    Returns:
        List of MaSIF dictionaries with all tensors moved to device
    """
    moved_masifs = []
    for masif_dict in masifs:
        moved_dict = {
            key: tensor.to(device, non_blocking=True) for key, tensor in masif_dict.items()
        }
        moved_masifs.append(moved_dict)
    return moved_masifs


# =========================
#      Train / Eval
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, config: Stage2SurfConfig, desc: str) -> Dict:
    model.eval()
    tracker = MetricsTracker()
    pbar = tqdm(dataloader, desc=desc, leave=False)
    device = config.device

    for batch in pbar:
        label = batch["label"].to(device)

        # 一次性将 MaSIF 数据搬到 GPU
        phla_masifs_gpu = move_masifs_to_device(batch["phla_masifs"], device)
        tcr_masifs_gpu = move_masifs_to_device(batch["tcr_masifs"], device)

        # surf_only 模式：不需要序列输入,直接传入 None
        output = model(
            peptide_emb=None,
            hla_emb=None,
            tcra_emb=None,
            tcrb_emb=None,
            peptide_mask=None,
            hla_mask=None,
            tcra_mask=None,
            tcrb_mask=None,
            phla_masifs=phla_masifs_gpu,
            tcr_masifs=tcr_masifs_gpu,
            mode="surf_only",
            return_attn=False,
        )
        logit = output["logit"].squeeze(-1)
        loss = criterion(logit, label)

        pred = torch.sigmoid(logit)
        tracker.update(label, pred, float(loss.item()))

    return tracker.compute()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    config: Stage2SurfConfig,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict:
    model.train()
    tracker = MetricsTracker()
    optimizer.zero_grad()

    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        try:
            label = batch["label"].to(config.device)

            # 一次性将 MaSIF 数据搬到 GPU
            phla_masifs_gpu = move_masifs_to_device(batch["phla_masifs"], config.device)
            tcr_masifs_gpu = move_masifs_to_device(batch["tcr_masifs"], config.device)

            if config.use_amp and scaler is not None:
                with torch.amp.autocast(config.device):
                    # surf_only 模式：不需要序列输入
                    output = model(
                        peptide_emb=None,
                        hla_emb=None,
                        tcra_emb=None,
                        tcrb_emb=None,
                        peptide_mask=None,
                        hla_mask=None,
                        tcra_mask=None,
                        tcrb_mask=None,
                        phla_masifs=phla_masifs_gpu,
                        tcr_masifs=tcr_masifs_gpu,
                        mode="surf_only",
                        return_attn=False,
                    )
                    logit = output["logit"].squeeze(-1)
                    loss = criterion(logit, label)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                # surf_only 模式：不需要序列输入
                output = model(
                    peptide_emb=None,
                    hla_emb=None,
                    tcra_emb=None,
                    tcrb_emb=None,
                    peptide_mask=None,
                    hla_mask=None,
                    tcra_mask=None,
                    tcrb_mask=None,
                    phla_masifs=phla_masifs_gpu,
                    tcr_masifs=tcr_masifs_gpu,
                    mode="surf_only",
                    return_attn=False,
                )
                logit = output["logit"].squeeze(-1)
                loss = criterion(logit, label)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            optimizer.zero_grad()

            # Update metrics
            with torch.no_grad():
                pred = torch.sigmoid(logit)
                tracker.update(label, pred, float(loss.item()))

            if step % config.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        except Exception as e:
            LOGGER.error("Exception in training step: %s", str(e))
            LOGGER.error(traceback.format_exc())
            continue

    return tracker.compute()


# =========================
#      Main training loop
# =========================
def train_fold(
    config: Stage2SurfConfig,
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    dataset: Stage2SurfDataset,
):
    """Train single fold"""
    fold_dir = Path(config.output_dir) / f"neg_ratio_{config.neg_ratio}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 80)
    LOGGER.info("Fold %d: %d train, %d val samples", fold, len(train_indices), len(val_indices))

    # Create dataloaders
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn_surf(batch, config.pmhc_P, config.tcr_P, config.num_feature_type),
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: collate_fn_surf(batch, config.pmhc_P, config.tcr_P, config.num_feature_type),
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    # Create model
    model = Network(
        # Sequence params (not used but required for Network_v3)
        hid=256,
        seq_nhead=8,
        seq_dropout=0.1,
        phla_seq_layers=1,
        tcr_seq_layers=1,

        # MaSIF params
        n_thetas=config.n_thetas,
        n_rhos=config.n_rhos,
        n_rotations=config.n_rotations,
        max_rho_phla=config.max_rho_phla,
        max_rho_tcr=config.max_rho_tcr,

        # Joint encoder params
        joint_nhead=config.joint_nhead,
        joint_layers=config.joint_layers,
        joint_dropout=config.joint_dropout,
        ff_dim_scale=config.ff_dim_scale,

        # Feature-type params
        num_feature_type=config.num_feature_type,
        ft_nhead=config.ft_nhead,
        ft_layers=config.ft_layers,
        ft_dropout=config.ft_dropout,
    ).to(config.device)

    LOGGER.info("Model summary:\n%s", model.check_logging(depth=2))

    # Loss
    if config.loss_type == "focal":
        criterion = BinaryFocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_initial, weight_decay=config.weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        lr_min=config.lr_min,
        lr_max=config.lr_max,
        lr_initial=config.lr_initial,
    )

    # Early stopping & plotting
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode="min")
    plotter = LossPlotter(fold_dir, fold, config.neg_ratio)

    # Mixed precision
    scaler = torch.amp.GradScaler(config.device) if config.use_amp else None

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        lr = scheduler.step()
        LOGGER.info("-" * 80)
        LOGGER.info("Fold %d | Epoch %d/%d | LR=%.2e", fold, epoch, config.max_epochs, lr)

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch, scaler)
        LOGGER.info("[Train] loss=%.4f auroc=%.4f auprc=%.4f acc=%.4f f1=%.4f",
                   train_metrics["loss"], train_metrics["auroc"], train_metrics["auprc"],
                   train_metrics["accuracy"], train_metrics["f1"])

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, config, desc=f"Epoch {epoch} [Val]")
        LOGGER.info("[Val]   loss=%.4f auroc=%.4f auprc=%.4f acc=%.4f f1=%.4f",
                   val_metrics["loss"], val_metrics["auroc"], val_metrics["auprc"],
                   val_metrics["accuracy"], val_metrics["f1"])

        # Update plotter
        plotter.update(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            train_auroc=train_metrics["auroc"],
            val_auroc=val_metrics["auroc"],
            train_auprc=train_metrics["auprc"],
            val_auprc=val_metrics["auprc"],
            lr=lr,
        )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "config": asdict(config),
            }, fold_dir / "best_model.pt")
            LOGGER.info("  -> Saved best model (val_loss=%.4f)", best_val_loss)

        # Periodic save
        if epoch % config.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": asdict(config),
            }, fold_dir / f"checkpoint_epoch_{epoch}.pt")

        # Early stopping
        if early_stopping(val_metrics["loss"]):
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            break

    LOGGER.info("Fold %d finished. Best epoch=%d val_loss=%.4f", fold, best_epoch, best_val_loss)
    return best_val_loss
# =========================
# Strict positive-ID merge
# =========================
# The helpers, strict dataset variant, and main entry point below preserve the
# current script's positive/negative CSV behavior. All training functionality
# above is now local to this file; oldversion is no longer imported at runtime.


def _normalize_columns(df):
    # Handle BOM and whitespace in CSV headers, e.g. "\ufeffid"
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _resolve_csv_path(user_path: Optional[str], candidates: list[Path]) -> Path:
    """
    Resolve CSV path:
    - If user_path is provided and exists -> use it
    - Else try candidates (first existing)
    - Else return user_path (even if missing) for clearer error
    """
    if user_path is not None:
        p = Path(user_path)
        if p.exists():
            return p
        p2 = (_BASE_DIR / p).resolve()
        if p2.exists():
            return p2
        return p
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _read_positive_ids(pos_csv_path: Path) -> Set[int]:
    import pandas as pd

    df = pd.read_csv(pos_csv_path)
    df = _normalize_columns(df)
    if "id" not in df.columns:
        raise ValueError(f"Positive CSV must contain column 'id', got {list(df.columns)}")
    ids = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int).tolist()
    return set(int(x) for x in ids)


class Stage2SurfDatasetStrictPos(Stage2SurfDataset):
    """
    与旧版 Stage2SurfDataset 相同，但正样本ID来源改为：
    - 从 positive CSV 的 id 列读取
    - 并与 MaSIF 文件夹存在性交集后得到 pos_ids
    """

    def __init__(
        self,
        imfp_dir: str,
        neg_ratio: int = 10,
        random_seed: int = 42,
        pmhc_P: int = 13,
        tcr_P: int = 16,
        neg_csv_path: Optional[str] = None,
        pos_csv_path: Optional[str] = None,
        cache_size: int = 2048,
        _prevalidated=None,  # (valid_pmhc_ids, valid_tcr_ids) to skip re-validation
    ):
        # Mirror old init layout but change how self.pos_ids is constructed
        super(Stage2SurfDataset, self).__init__()  # Skip base init; fields are initialized below

        self.imfp_dir = Path(imfp_dir)
        self.neg_ratio = int(neg_ratio)
        self.random_seed = int(random_seed)
        self.pmhc_P = int(pmhc_P)
        self.tcr_P = int(tcr_P)
        self.cache_size = int(cache_size)

        pmhc_folder = self.imfp_dir / "train_pmhc"
        tcr_folder = self.imfp_dir / "train_tcr"
        if not pmhc_folder.exists() or not tcr_folder.exists():
            raise FileNotFoundError(f"MaSIF folders not found: {pmhc_folder} or {tcr_folder}")

        # Extract IDs from folder names
        pmhc_ids = set()
        for folder in pmhc_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("pmhc_"):
                try:
                    pmhc_ids.add(int(folder.name.split("_")[1]))
                except (ValueError, IndexError):
                    continue

        tcr_ids = set()
        for folder in tcr_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("tcr_"):
                try:
                    tcr_ids.add(int(folder.name.split("_")[1]))
                except (ValueError, IndexError):
                    continue

        folder_pos_ids = set(pmhc_ids & tcr_ids)

        # Resolve and read positive CSV IDs, then constrain pos_ids
        pos_csv_candidates = [
            _BASE_DIR / "data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv",
            _BASE_DIR / "data/Database_stage1/analyze_outputs/outputs_split/training_positive_clear_peplen7.csv",
            self.imfp_dir.parent / "outputs_split" / "training_positive_clear_peplen7.csv",
        ]
        pos_csv_path_resolved = _resolve_csv_path(pos_csv_path, pos_csv_candidates)
        if not pos_csv_path_resolved.exists():
            raise FileNotFoundError(f"Missing positive CSV: {pos_csv_path_resolved}")

        pos_csv_ids = _read_positive_ids(pos_csv_path_resolved)
        self.pos_ids = sorted(list(folder_pos_ids & pos_csv_ids))

        print(
            f"Positive ID constraint: folder_pos_ids={len(folder_pos_ids)} | "
            f"pos_csv_ids={len(pos_csv_ids)} | final_pos_ids={len(self.pos_ids)}"
        )
        if len(self.pos_ids) == 0:
            raise RuntimeError(
                "No positive IDs left after applying positive CSV constraint + folder existence. "
                f"pos_csv={pos_csv_path_resolved}, imfp_dir={self.imfp_dir}"
            )

        # Load negative CSV (same as old, but with robust default resolution)
        neg_csv_candidates = [
            _BASE_DIR / "data/Database_stage1/outputs_split/training_negative_clear_peplen7_v1.csv",
            _BASE_DIR / "data/Database_stage1/analyze_outputs/outputs_split/training_negative_clear_peplen7_v1.csv",
            self.imfp_dir.parent / "outputs_split" / "training_negative_clear_peplen7_v1.csv",
        ]
        neg_csv_path_resolved = _resolve_csv_path(neg_csv_path, neg_csv_candidates)
        if not neg_csv_path_resolved.exists():
            raise FileNotFoundError(f"Missing negative CSV: {neg_csv_path_resolved}")

        self.neg_csv_path = Path(neg_csv_path_resolved)

        import pandas as pd

        self.neg_df = pd.read_csv(self.neg_csv_path)
        self.neg_df = _normalize_columns(self.neg_df)
        if "id" not in self.neg_df.columns or "id_tcr" not in self.neg_df.columns:
            raise ValueError(
                f"Negative CSV must contain columns ['id','id_tcr'], got {list(self.neg_df.columns)}"
            )

        # Validate MaSIF folders and build samples (reuse inherited methods)
        if _prevalidated is not None:
            self.valid_pmhc_ids, self.valid_tcr_ids = _prevalidated
            print(f"Reusing prevalidated: {len(self.valid_pmhc_ids)} pMHC, {len(self.valid_tcr_ids)} TCR")
        else:
            print("Validating MaSIF folders...")
            self.valid_pmhc_ids, self.valid_tcr_ids = self._validate_masif_folders()
            print(f"Valid pMHC folders: {len(self.valid_pmhc_ids)}")
            print(f"Valid TCR folders: {len(self.valid_tcr_ids)}")

        self.samples = self._build_sample_list()

        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(self.samples)

        # LRU cache (same fields expected by inherited _load_* methods)
        self._pmhc_cache = {}
        self._tcr_cache = {}
        self._pmhc_cache_order = []
        self._tcr_cache_order = []


def main():
    global LOGGER

    parser = argparse.ArgumentParser(description="Stage 2 Surf-Only Training - Single Fold (strict positive IDs)")
    parser.add_argument("--imfp_dir", type=str, required=True, help="Path to MaSIF data directory")
    parser.add_argument("--fold", type=int, required=True, help="Which fold to train (0-4)")
    parser.add_argument("--gpu", type=int, required=True, help="Which GPU to use (0, 1, 2, ...)")
    parser.add_argument("--neg_ratio", type=int, default=10, help="Negative sampling ratio")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=40, help="Maximum epochs")
    parser.add_argument("--lr_max", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--output_dir", type=str, default="./runs/stage2_surf", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--n_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pos_csv_path",
        type=str,
        default="data/Database_stage1/outputs_split/training_positive_clear_peplen7.csv",
        help="Path to positive samples CSV (must contain column 'id')",
    )
    parser.add_argument(
        "--neg_csv_path",
        type=str,
        default="data/Database_stage1/outputs_split/training_negative_clear_peplen7_v1.csv",
        help="Path to negative sampling CSV (must contain columns 'id','id_tcr')",
    )

    args = parser.parse_args()

    if args.fold < 0 or args.fold >= args.n_folds:
        raise ValueError(f"Fold must be in range [0, {args.n_folds - 1}], got {args.fold}")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    config = Stage2SurfConfig(
        imfp_dir=args.imfp_dir,
        neg_ratio=args.neg_ratio,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr_max=args.lr_max,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        neg_csv_path=args.neg_csv_path,
        fold=args.fold,
        gpu=args.gpu,
        device=device,
    )

    output_dir = Path(config.output_dir) / f"neg_ratio_{config.neg_ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the shared logger used by the local training functions
    LOGGER = setup_logging(output_dir, config.fold)
    LOGGER.info("=" * 80)
    LOGGER.info("Training Single Fold (strict pos ids): Fold %d on GPU %d", config.fold, config.gpu)
    LOGGER.info("=" * 80)
    log_environment(LOGGER, config)
    LOGGER.info("pos_csv_path: %s", str(args.pos_csv_path))

    # Save config (include pos_csv_path as extra field)
    config_path = output_dir / f"fold_{config.fold}" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["pos_csv_path"] = str(args.pos_csv_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Create dataset
    LOGGER.info("Creating dataset (strict positive IDs)...")
    dataset = Stage2SurfDatasetStrictPos(
        imfp_dir=config.imfp_dir,
        neg_ratio=config.neg_ratio,
        random_seed=config.random_seed,
        pmhc_P=config.pmhc_P,
        tcr_P=config.tcr_P,
        neg_csv_path=config.neg_csv_path,
        pos_csv_path=args.pos_csv_path,
    )
    LOGGER.info("Dataset size: %d samples", len(dataset))

    # K-fold split
    labels = dataset.get_labels()
    groups = dataset.get_group_ids()

    if config.foldtype == "StratifiedGroupKFold":
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            kfold = StratifiedGroupKFold(
                n_splits=config.n_folds, shuffle=True, random_state=config.random_seed
            )
            splits = list(kfold.split(np.arange(len(dataset)), labels, groups))
            LOGGER.info("Using StratifiedGroupKFold")
        except ImportError:
            LOGGER.warning("StratifiedGroupKFold not available, falling back to StratifiedKFold")
            from sklearn.model_selection import StratifiedKFold

            kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
            splits = list(kfold.split(np.arange(len(dataset)), labels))
    else:
        from sklearn.model_selection import StratifiedKFold

        kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
        splits = list(kfold.split(np.arange(len(dataset)), labels))
        LOGGER.info("Using StratifiedKFold")

    train_indices, val_indices = splits[config.fold]
    LOGGER.info(
        "Training fold %d with %d train samples and %d val samples",
        config.fold,
        len(train_indices),
        len(val_indices),
    )

    best_val_loss = train_fold(config, config.fold, train_indices, val_indices, dataset)

    LOGGER.info("=" * 80)
    LOGGER.info("Training completed!")
    LOGGER.info("Fold %d: best_val_loss=%.4f", config.fold, best_val_loss)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
