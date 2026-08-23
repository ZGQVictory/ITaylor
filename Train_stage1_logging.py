# -*- coding: utf-8 -*-
"""
Train_stage1_logging.py

Stage 1 Sequence-Only: Sequence foundation training for pHLA-TCR binding.
Uses sequence information only to learn the representation grammar of pHLA and TCR.

Key design choices:
- Sequence inputs only:
  - Positive samples use float32 ESM embeddings to preserve precision.
  - Negative ESM embeddings are not preloaded.
- Negative construction:
  - Read source IDs from training_negative_clear_peplen7_merged.csv.
  - id_epitope selects the peptide embedding.
  - id_hla selects the HLA mature-chain embedding.
  - id_tcr selects the TCR alpha- and beta-chain embeddings.
- Memory optimization:
  - Build negative samples dynamically from positive embeddings.
  - Use torch.load(..., mmap=True) when supported.

Usage examples:
  # Train with a 1:1 positive-to-negative ratio
  python Train_stage1_logging.py \
    --data_dir ./data/Database_stage1/esm-embedding \
    --neg_ratio 1 \
    --output_dir ./runs/stage1

  # Train with the default 1:10 positive-to-negative ratio
  python Train_stage1_logging.py \
    --data_dir ./data/Database_stage1/esm-embedding \
    --neg_ratio 10 \
    --output_dir ./runs/stage1

"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
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
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from Network_v3 import Network


# =========================
#      Config
# =========================
@dataclass
class Stage1Config:
    """Stage 1 training configuration."""
    # Data
    data_dir: str  # ESM embedding directory
    neg_ratio: int = 10  # Positive-to-negative ratio: 1:neg_ratio; typically 1, 5, or 10
    n_folds: int = 5
    random_seed: int = 42

    # Model hyperparameters; Stage 1 trains the sequence branch only
    hid: int = 256
    seq_nhead: int = 8
    seq_dropout: float = 0.1
    phla_seq_layers: int = 1  # Previously 1
    tcr_seq_layers: int = 1  # Previously 1
    num_feature_type: int = 5

    # Training hyperparameters
    batch_size: int = 128  # Supports batched training
    max_epochs: int = 50
    lr_initial: float = 1e-5
    lr_max: float = 1e-4
    lr_min: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # loss
    loss_type: str = "focal"   # "bce" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    save_every: int = 5
    log_every: int = 50

    # Output
    output_dir: str = "./runs/stage1"
    training_details_dir: str = "./Database_stage1/training_details"
    use_amp: bool = True

# =========================
#      Logging Utils
# =========================
LOGGER = logging.getLogger("train_stage1")


def _get_cpu_rss_mb() -> float:
    """Best-effort current process RSS in MB."""
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


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the same destination while representing the log path relative to the working directory.
    log_path = Path(os.path.relpath(output_dir / "train_stage1.log", start=Path.cwd()))

    logger = logging.getLogger("train_stage1")
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


def log_environment(logger: logging.Logger, config: Stage1Config) -> None:
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
#      Dataset
# =========================
class pHLATCRDataset(Dataset):
    """
    Sequence-only pHLA-TCR binding dataset for Stage 1.

    Version 3.0 reduces memory use by loading only positive embeddings with
    optional mmap support and constructing negatives dynamically from the CSV.
    """

    def __init__(self, data_dir: str, neg_ratio: int = 10, random_seed: int = 42, use_mmap: bool = True):
        """
        Args:
            data_dir: Directory containing ESM embedding files.
            neg_ratio: Positive-to-negative ratio, 1:neg_ratio, from 1 to 30.
            random_seed: Random seed used for negative sampling.
            use_mmap: Whether to load embeddings with mmap; recommended.
        """
        self.data_dir = Path(data_dir)
        self.neg_ratio = neg_ratio
        self.random_seed = random_seed
        self.use_mmap = use_mmap

        print(f"\n{'='*60}")
        print(f"Loading ESM embeddings from {data_dir}...")
        print(f"Negative sample ratio: 1:{neg_ratio}")
        print(f"Using mmap: {use_mmap}")
        print(f"Memory optimization: Only loading positive embeddings")
        print(f"{'='*60}\n")

        # Local helper for loading tensors with optional mmap support
        def _load_cpu_mmap(p):
            p = str(p)
            if use_mmap:
                try:
                    return torch.load(p, map_location="cpu", mmap=True)
                except TypeError:
                    # Fall back to a regular load when mmap is unsupported
                    print(f"  Warning: mmap not supported, falling back to normal load")
                    return torch.load(p, map_location="cpu")
            else:
                return torch.load(p, map_location="cpu")

        # Load positive embeddings at float32 precision
        pos_emb_path = self.data_dir / "training_positive_clear_peplen7_esm_embeddings_float32.pt"
        # pos_emb_path = self.data_dir / "training_positive_esm_embeddings_clear_peplen7.pt"

        pos_meta_path = self.data_dir / "training_positive_metadata_clear_peplen7.pt"

        print(f"  Loading positive embeddings (float32)...")
        self.pos_embeddings = _load_cpu_mmap(pos_emb_path)
        self.pos_metadata = torch.load(str(pos_meta_path), map_location="cpu")  # Metadata is small; mmap is unnecessary
        try:
            _log_mem(LOGGER, 'cpu', prefix='DATA pos_loaded')
        except Exception:
            pass

        # Read the negative-sample CSV containing the source-ID mappings
        # CSV structure:
        #   - id_epitope selects a peptide from the positive embeddings
        #   - id_hla selects an HLA mature chain from the positive embeddings
        #   - id_tcr selects both TCR chains from the positive embeddings
        # Negatives are constructed by mismatching these positive-sample components.
        neg_csv_path = self.data_dir.parent / "outputs_split" / "training_negative_clear_peplen7_merged.csv"
        print(f"  Loading negative sample info from CSV...")
        print(f"    Path: {neg_csv_path}")

        import pandas as pd
        self.neg_df = pd.read_csv(neg_csv_path)
        print(f"    Total negative samples in CSV: {len(self.neg_df)}")
        print(f"    Negative samples will be dynamically constructed using:")
        print(f"      - 'id_epitope' column for peptide embedding")
        print(f"      - 'id_hla' column for HLA embedding")
        print(f"      - 'id_tcr' column for TCR embeddings")
        print(f"      - anchor_id is inferred per-row (two of the three ids are the anchor positive id)")

        try:
            _log_mem(LOGGER, 'cpu', prefix='DATA neg_csv_loaded')
        except Exception:
            pass

        # Build the sample list
        self.samples = self._build_sample_list()

        # Shuffle sample order
        np.random.seed(random_seed)
        np.random.shuffle(self.samples)

        print(f"Total samples: {len(self.samples)} "
              f"(Positive: {sum(1 for s in self.samples if s['label'] == 1)}, "
              f"Negative: {sum(1 for s in self.samples if s['label'] == 0)})")

        try:
            _log_mem(LOGGER, 'cpu', prefix='DATA final')
        except Exception:
            pass

    
    def _build_sample_list(self) -> List[Dict]:
        """
        Build the sample list with up to ``neg_ratio`` negatives per positive.

        Version 3.0 (updated for new negatives):
        - Negative embeddings are no longer preloaded.
        - Three source IDs are read from the CSV:
            * id_epitope selects the peptide embedding.
            * id_hla selects the HLA mature-chain embedding.
            * id_tcr selects both TCR-chain embeddings.
        - An anchor_id is inferred for every negative row and used to group
          negative candidates by their corresponding positive sample. The
          negative-generation strategy guarantees that two of id_hla,
          id_epitope, and id_tcr are equal; that repeated ID is the anchor_id.
        """
        samples: List[Dict] = []
        np.random.seed(self.random_seed)

        # Get all positive-sample IDs
        pos_sample_ids = self.pos_metadata['sample_ids']

        # Infer each negative row's anchor_id from the two matching IDs
        def _infer_anchor_id(row) -> int:
            a = int(row['id_hla'])
            b = int(row['id_epitope'])
            c = int(row['id_tcr'])
            if a == b or a == c:
                return a
            if b == c:
                return b
            # This indicates that the CSV does not follow the negative-generation rules
            raise ValueError(f"Cannot infer anchor_id from ids: id_hla={a}, id_epitope={b}, id_tcr={c}")

        # Map each anchor_id to its negative-CSV row indices
        neg_anchor_to_csv_indices = defaultdict(list)
        for csv_idx, row in self.neg_df.iterrows():
            anchor_id = _infer_anchor_id(row)
            neg_anchor_to_csv_indices[anchor_id].append(csv_idx)

        # Add samples for each positive entry
        for pos_idx, pos_id in enumerate(pos_sample_ids):
            # Add the positive sample
            samples.append({
                'sample_id': int(pos_id),
                'is_positive': True,
                'pos_idx': pos_idx,
                'label': 1
            })

            # Select matching negatives grouped by anchor_id
            neg_csv_indices = neg_anchor_to_csv_indices.get(int(pos_id), [])
            if len(neg_csv_indices) > 0:
                # Randomly select up to neg_ratio candidates
                if len(neg_csv_indices) >= self.neg_ratio:
                    selected_csv_indices = np.random.choice(
                        neg_csv_indices, size=self.neg_ratio, replace=False
                    )
                else:
                    # Use every candidate when fewer than neg_ratio are available
                    selected_csv_indices = neg_csv_indices

                for csv_idx in selected_csv_indices:
                    # Read the three source IDs from the negative CSV
                    neg_row = self.neg_df.loc[csv_idx]
                    id_epitope = int(neg_row['id_epitope'])
                    id_hla = int(neg_row['id_hla'])
                    id_tcr = int(neg_row['id_tcr'])

                    samples.append({
                        # Anchor ID retained for compatibility with existing interfaces and logs
                        'sample_id': int(pos_id),
                        'is_positive': False,
                        'csv_idx': int(csv_idx),
                        'id_epitope': id_epitope,
                        'id_hla': id_hla,
                        'id_tcr': id_tcr,
                        'label': 0
                    })

        return samples



    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_id = sample_info['sample_id']

        if sample_info['is_positive']:
            # Positive sample: retrieve all components directly from positive embeddings
            peptide_emb = self.pos_embeddings['Epitope'][sample_id]
            hla_emb = self.pos_embeddings['hla_alpha123_mature'][sample_id]
            tcra_emb = self.pos_embeddings['tcra_variable'][sample_id]
            tcrb_emb = self.pos_embeddings['tcrb_variable'][sample_id]
        else:
            # Negative sample: construct components dynamically
            # - id_epitope selects the peptide
            # - id_hla selects the HLA mature chain
            # - id_tcr selects both TCR chains
            id_epitope = sample_info['id_epitope']
            id_hla = sample_info['id_hla']
            id_tcr = sample_info['id_tcr']

            peptide_emb = self.pos_embeddings['Epitope'][id_epitope]
            hla_emb = self.pos_embeddings['hla_alpha123_mature'][id_hla]
            tcra_emb = self.pos_embeddings['tcra_variable'][id_tcr]
            tcrb_emb = self.pos_embeddings['tcrb_variable'][id_tcr]

        return {
            'id': sample_id,
            'peptide_emb': peptide_emb,
            'hla_emb': hla_emb,
            'tcra_emb': tcra_emb,
            'tcrb_emb': tcrb_emb,
            'label': torch.tensor(sample_info['label'], dtype=torch.float32),
        }


    def get_labels(self):
        """Return all labels for StratifiedKFold."""
        return [sample['label'] for sample in self.samples]

    def get_original_ids(self):
        """Return original IDs for group-aware train/validation splitting."""
        return [sample['sample_id'] for sample in self.samples]


def collate_fn(batch):
    """
    Collate a batch by padding variable-length sequences and creating masks.
    """
    import torch.nn.functional as F

    # Fixed lengths including BOS/EOS tokens
    peptide_max_len = 15
    hla_len = 276
    tcra_max_len = 127
    tcrb_max_len = 130

    batch_size = len(batch)

    # Prepare batch containers
    peptide_embs = []
    peptide_masks = []
    hla_embs = []
    hla_masks = []
    tcra_embs = []
    tcra_masks = []
    tcrb_embs = []
    tcrb_masks = []
    labels = []
    ids = []

    for sample in batch:
        # Peptide
        pep_emb = sample['peptide_emb']  # [L_pep, 1152]
        L_pep = pep_emb.size(0)

        # Pad or truncate to the fixed length
        if L_pep < peptide_max_len:
            pep_emb = F.pad(pep_emb, (0, 0, 0, peptide_max_len - L_pep))
        else:
            pep_emb = pep_emb[:peptide_max_len]

        # Build the validity mask
        pep_mask = torch.zeros(peptide_max_len, dtype=torch.bool)
        pep_mask[:min(L_pep, peptide_max_len)] = True

        peptide_embs.append(pep_emb)
        peptide_masks.append(pep_mask)

        # HLA has a fixed length and requires no padding
        hla_emb = sample['hla_emb']  # [276, 1152]
        hla_mask = torch.ones(hla_len, dtype=torch.bool)
        hla_embs.append(hla_emb)
        hla_masks.append(hla_mask)

        # TCR alpha chain
        tcra_emb = sample['tcra_emb']  # [L_tcra, 1152]
        L_tcra = tcra_emb.size(0)

        if L_tcra < tcra_max_len:
            tcra_emb = F.pad(tcra_emb, (0, 0, 0, tcra_max_len - L_tcra))
        else:
            tcra_emb = tcra_emb[:tcra_max_len]

        tcra_mask = torch.zeros(tcra_max_len, dtype=torch.bool)
        tcra_mask[:min(L_tcra, tcra_max_len)] = True

        tcra_embs.append(tcra_emb)
        tcra_masks.append(tcra_mask)

        # TCR beta chain
        tcrb_emb = sample['tcrb_emb']  # [L_tcrb, 1152]
        L_tcrb = tcrb_emb.size(0)

        if L_tcrb < tcrb_max_len:
            tcrb_emb = F.pad(tcrb_emb, (0, 0, 0, tcrb_max_len - L_tcrb))
        else:
            tcrb_emb = tcrb_emb[:tcrb_max_len]

        tcrb_mask = torch.zeros(tcrb_max_len, dtype=torch.bool)
        tcrb_mask[:min(L_tcrb, tcrb_max_len)] = True

        tcrb_embs.append(tcrb_emb)
        tcrb_masks.append(tcrb_mask)

        # Label
        labels.append(sample['label'])
        ids.append(sample['id'])

    # Stack individual samples into a batch
    return {
        'id': ids,
        'peptide_emb': torch.stack(peptide_embs, dim=0),      # [B, 15, 1152]
        'peptide_mask': torch.stack(peptide_masks, dim=0),    # [B, 15]
        'hla_emb': torch.stack(hla_embs, dim=0),              # [B, 276, 1152]
        'hla_mask': torch.stack(hla_masks, dim=0),            # [B, 276]
        'tcra_emb': torch.stack(tcra_embs, dim=0),            # [B, 127, 1152]
        'tcra_mask': torch.stack(tcra_masks, dim=0),          # [B, 127]
        'tcrb_emb': torch.stack(tcrb_embs, dim=0),            # [B, 130, 1152]
        'tcrb_mask': torch.stack(tcrb_masks, dim=0),          # [B, 130]
        'label': torch.stack(labels, dim=0),                  # [B]
    }


# =========================
#      Training utilities
# =========================
class MetricsTracker:
    """Accumulate predictions and compute binary-classification metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []
        self.preds = []
        self.losses = []

    def update(self, labels, preds, loss):
        self.labels.extend(labels.cpu().numpy().tolist())
        self.preds.extend(preds.cpu().numpy().tolist())
        self.losses.append(loss)

    def compute(self) -> Dict:
        labels = np.array(self.labels)
        preds = np.array(self.preds)

        # Binary-classification metrics
        preds_binary = (preds > 0.5).astype(int)

        metrics = {
            'loss': np.mean(self.losses),
            'auroc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
            'auprc': average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0,
            'accuracy': accuracy_score(labels, preds_binary),
            'f1': f1_score(labels, preds_binary, zero_division=0),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        metrics.update({
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        })

        return metrics


class CosineWarmupScheduler:
    """Cosine-annealing learning-rate scheduler with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, lr_min, lr_max, lr_initial):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_initial = lr_initial
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: increase the learning rate linearly
            lr = self.lr_initial + (self.lr_max - self.lr_initial) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine-annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


class EarlyStopping:
    """Track validation progress and trigger early stopping."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = (self.best_score - score) > self.min_delta
        else:
            improved = (score - self.best_score) > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class LossPlotter:
    """
    Update and save training curves after every epoch.
    """

    def __init__(self, save_dir: Path, fold: int, neg_ratio: int):
        """
        Args:
            save_dir: Directory in which plots and metric data are saved.
            fold: Current fold index.
            neg_ratio: Positive-to-negative sample ratio.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fold = fold
        self.neg_ratio = neg_ratio

        # Store metric history
        self.train_losses = []
        self.val_losses = []
        self.train_aurocs = []
        self.val_aurocs = []
        self.epochs = []
        self.lrs = []

    def update(self, epoch: int, train_loss: float, val_loss: float,
               train_auroc: float, val_auroc: float, lr: float):
        """Append one epoch of data and regenerate the plots."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_aurocs.append(train_auroc)
        self.val_aurocs.append(val_auroc)
        self.lrs.append(lr)

        # Draw the curves
        self._plot_curves()

        # Save numeric data
        self._save_data()

    def _plot_curves(self):
        """Draw and save the loss, AUROC, and learning-rate curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curves
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Fold {self.fold + 1} Loss (Neg Ratio 1:{self.neg_ratio})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AUROC curves
        axes[1].plot(self.epochs, self.train_aurocs, 'b-', label='Train AUROC', linewidth=2)
        axes[1].plot(self.epochs, self.val_aurocs, 'r-', label='Val AUROC', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUROC')
        axes[1].set_title(f'Fold {self.fold + 1} AUROC (Neg Ratio 1:{self.neg_ratio})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        # Learning-rate curve
        axes[2].plot(self.epochs, self.lrs, 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'Fold {self.fold + 1} Learning Rate')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'fold_{self.fold}_training_curves.png', dpi=150)
        plt.close()

    def _save_data(self):
        """Save training data to a JSON file."""
        data = {
            'fold': self.fold,
            'neg_ratio': self.neg_ratio,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_aurocs': self.train_aurocs,
            'val_aurocs': self.val_aurocs,
            'learning_rates': self.lrs,
        }
        with open(self.save_dir / f'fold_{self.fold}_training_data.json', 'w') as f:
            json.dump(data, f, indent=2)


# =========================
#      Training and validation
# =========================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    config: Stage1Config,
    epoch: int,
    scaler=None,
) -> Dict:
    """Train one epoch in Stage 1 ``seq_only`` mode."""
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
        step_t0 = time.time()
        try:
            # Move input tensors to the configured device
            peptide_emb = batch['peptide_emb'].to(config.device)
            peptide_mask = batch['peptide_mask'].to(config.device)
            hla_emb = batch['hla_emb'].to(config.device)
            hla_mask = batch['hla_mask'].to(config.device)
            tcra_emb = batch['tcra_emb'].to(config.device)
            tcra_mask = batch['tcra_mask'].to(config.device)
            tcrb_emb = batch['tcrb_emb'].to(config.device)
            tcrb_mask = batch['tcrb_mask'].to(config.device)
            label = batch['label'].to(config.device)

            if step == 0:
                try:
                    LOGGER.info(
                        "Train epoch=%d first batch shapes: peptide=%s hla=%s tcra=%s tcrb=%s label=%s",
                        epoch,
                        tuple(peptide_emb.shape),
                        tuple(hla_emb.shape),
                        tuple(tcra_emb.shape),
                        tuple(tcrb_emb.shape),
                        tuple(label.shape),
                    )
                    _log_mem(LOGGER, config.device, prefix=f"TRAIN e{epoch} first_batch")
                except Exception:
                    pass

            # Forward pass in seq_only mode
            if config.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(
                        peptide_emb=peptide_emb,
                        hla_emb=hla_emb,
                        tcra_emb=tcra_emb,
                        tcrb_emb=tcrb_emb,
                        peptide_mask=peptide_mask,
                        hla_mask=hla_mask,
                        tcra_mask=tcra_mask,
                        tcrb_mask=tcrb_mask,
                        mode="seq_only",  # Stage 1 uses sequence inputs only
                    )
                    logit = output['logit'].squeeze(-1)  # [B, 1] -> [B]
                    loss = criterion(logit, label)
            else:
                output = model(
                    peptide_emb=peptide_emb,
                    hla_emb=hla_emb,
                    tcra_emb=tcra_emb,
                    tcrb_emb=tcrb_emb,
                    peptide_mask=peptide_mask,
                    hla_mask=hla_mask,
                    tcra_mask=tcra_mask,
                    tcrb_mask=tcrb_mask,
                    mode="seq_only",
                )
                logit = output['logit'].squeeze(-1)  # [B, 1] -> [B]
                loss = criterion(logit, label)

            # Backpropagation without gradient accumulation; update every step
            if config.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            optimizer.zero_grad()

            # Record metrics
            with torch.no_grad():
                pred = torch.sigmoid(logit)  # [B]
                tracker.update(label, pred, loss.item())

            # Update the progress bar and logs
            if step % config.log_every == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                try:
                    lr = optimizer.param_groups[0].get('lr', float('nan'))
                    dt = time.time() - step_t0
                    LOGGER.info(
                        "Train epoch=%d step=%d/%d loss=%.6f lr=%.6g grad_norm=%.4f step_time=%.3fs",
                        epoch, step, len(dataloader), float(loss.item()), float(lr), float(grad_norm), dt
                    )
                    _log_mem(LOGGER, config.device, prefix=f"TRAIN e{epoch} s{step}")
                except Exception:
                    pass

        except Exception as e:
            LOGGER.error("TRACE train_one_epoch exception at epoch=%d step=%d: %s", epoch, step, repr(e))
            LOGGER.error(traceback.format_exc())
            try:
                LOGGER.error(
                    "Batch keys=%s | shapes: peptide=%s hla=%s tcra=%s tcrb=%s label=%s",
                    list(batch.keys()),
                    tuple(batch['peptide_emb'].shape),
                    tuple(batch['hla_emb'].shape),
                    tuple(batch['tcra_emb'].shape),
                    tuple(batch['tcrb_emb'].shape),
                    tuple(batch['label'].shape),
                )
            except Exception:
                pass
            try:
                _log_mem(LOGGER, config.device, prefix=f"EXC e{epoch} s{step}")
            except Exception:
                pass
            raise

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Stage1Config,
    desc: str = "Validation"
) -> Dict:
    """Run validation in Stage 1 ``seq_only`` mode."""
    model.eval()
    tracker = MetricsTracker()

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        # Move input tensors to the configured device
        peptide_emb = batch['peptide_emb'].to(config.device)
        peptide_mask = batch['peptide_mask'].to(config.device)
        hla_emb = batch['hla_emb'].to(config.device)
        hla_mask = batch['hla_mask'].to(config.device)
        tcra_emb = batch['tcra_emb'].to(config.device)
        tcra_mask = batch['tcra_mask'].to(config.device)
        tcrb_emb = batch['tcrb_emb'].to(config.device)
        tcrb_mask = batch['tcrb_mask'].to(config.device)
        label = batch['label'].to(config.device)

        output = model(
            peptide_emb=peptide_emb,
            hla_emb=hla_emb,
            tcra_emb=tcra_emb,
            tcrb_emb=tcrb_emb,
            peptide_mask=peptide_mask,
            hla_mask=hla_mask,
            tcra_mask=tcra_mask,
            tcrb_mask=tcrb_mask,
            mode="seq_only",
        )
        logit = output['logit'].squeeze(-1)  # [B, 1] -> [B]
        loss = criterion(logit, label)

        pred = torch.sigmoid(logit)  # [B]
        tracker.update(label, pred, loss.item())

    return tracker.compute()

# =========================
#      Focal loss
# =========================
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification, working on logits directly.
    Paper: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B], targets: [B] in {0,1}
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

# =========================
#      Single-fold training
# =========================
def train_single_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset: Dataset,
    config: Stage1Config,
    fold_dir: Path,
) -> Dict:
    """Train one Stage 1 fold."""

    print("\n" + "=" * 80)
    print(f"Stage 1 - Fold {fold + 1}/{config.n_folds}")
    print("=" * 80)
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples:   {len(val_idx)}")

    # Create dataset subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.device.startswith("cuda") else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.device.startswith("cuda") else False
    )

    # Create the model; Stage 1 uses only its sequence branch
    model = Network(
        hid=config.hid,
        seq_nhead=config.seq_nhead,
        seq_dropout=config.seq_dropout,
        phla_seq_layers=config.phla_seq_layers,
        tcr_seq_layers=config.tcr_seq_layers,
        num_feature_type=config.num_feature_type,
    )
    model = model.to(config.device)

    # Optimize all parameters; the structure tower is unused in seq_only mode
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr_initial,
        weight_decay=config.weight_decay
    )

    # Loss function
    if config.loss_type == "focal":
        criterion = BinaryFocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Learning-rate scheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        lr_min=config.lr_min,
        lr_max=config.lr_max,
        lr_initial=config.lr_initial
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='min'
    )

    # Mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and str(config.device).startswith("cuda")) else None

    # Loss-curve plotter
    training_details_dir = Path(config.training_details_dir) / f"neg_ratio_{config.neg_ratio}"
    loss_plotter = LossPlotter(
        save_dir=training_details_dir,
        fold=fold,
        neg_ratio=config.neg_ratio
    )

    # Training history
    history = {
        'train_loss': [], 'train_auroc': [], 'train_auprc': [],
        'val_loss': [], 'val_auroc': [], 'val_auprc': [],
        'lr': []
    }

    best_val_auroc = 0.0
    best_epoch = 0

    fold_start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        LOGGER.info("Fold %d Epoch %d start", fold+1, epoch)
        _log_mem(LOGGER, config.device, prefix=f"F{fold+1} E{epoch} START")
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config, epoch, scaler
        )

        LOGGER.info("Fold %d Epoch %d train: loss=%.6f auroc=%.4f auprc=%.4f", fold+1, epoch, train_metrics['loss'], train_metrics['auroc'], train_metrics['auprc'])
        _log_mem(LOGGER, config.device, prefix=f"F{fold+1} E{epoch} AFTER_TRAIN")
        # Validate
        val_metrics = validate(model, val_loader, criterion, config, desc=f"Fold {fold+1} Epoch {epoch} [Val]")

        LOGGER.info("Fold %d Epoch %d val:   loss=%.6f auroc=%.4f auprc=%.4f", fold+1, epoch, val_metrics['loss'], val_metrics['auroc'], val_metrics['auprc'])
        _log_mem(LOGGER, config.device, prefix=f"F{fold+1} E{epoch} AFTER_VAL")
        # Update the learning rate
        current_lr = scheduler.step()
        LOGGER.info("Fold %d Epoch %d lr=%.6g", fold+1, epoch, current_lr)

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['train_auprc'].append(train_metrics['auprc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['lr'].append(current_lr)

        # Update loss curves
        loss_plotter.update(
            epoch=epoch,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'],
            train_auroc=train_metrics['auroc'],
            val_auroc=val_metrics['auroc'],
            lr=current_lr
        )

        # Print results
        print(f"Fold {fold+1} Epoch {epoch}/{config.max_epochs} - LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | AUROC: {train_metrics['auroc']:.4f} | "
              f"AUPRC: {train_metrics['auprc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.4f} | "
              f"AUPRC: {val_metrics['auprc']:.4f}")

        # Save the best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': asdict(config),
            }, fold_dir / "best_model.pt")
            print(f"  >>> Saved best model (AUROC: {best_val_auroc:.4f})")

        # Save periodic checkpoints
        if epoch % config.save_every == 0:
            LOGGER.info("Saved checkpoint: fold=%d epoch=%d", fold+1, epoch)
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history,
                'config': asdict(config),
            }, fold_dir / f"checkpoint_epoch_{epoch}.pt")

        # Check early stopping
        if early_stopping(val_metrics['loss']):
            print(f"  Early stopping triggered at epoch {epoch}")
            break

    fold_time = time.time() - fold_start_time
    print(f"\nFold {fold+1} completed in {fold_time / 3600:.2f} hours")
    print(f"Best validation AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")

    # Save training history
    torch.save(history, fold_dir / "training_history.pt")

    # Load the best model and run final validation
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_val_metrics = validate(model, val_loader, criterion, config, desc=f"Fold {fold+1} Final Val")

    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_val_auroc': best_val_auroc,
        'final_metrics': final_val_metrics,
        'history': history,
    }


# =========================
#      Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='Stage 1: Sequence Foundation Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to ESM embedding directory (e.g., ./data/Database_stage1/esm-embedding)')
    parser.add_argument('--neg_ratio', type=int, default=10,
                        help='Negative sample ratio (1:neg_ratio)')
    parser.add_argument('--output_dir', type=str, default='./runs/stage1', help='Output directory')
    parser.add_argument('--training_details_dir', type=str, default='./Database_stage1/training_details',
                        help='Directory to save training details (loss curves, etc.)')
    parser.add_argument('--fold', type=int, default=None, help='Train only a specific fold (0-4)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')

    args = parser.parse_args()

    # Create the configuration
    config = Stage1Config(
        data_dir=args.data_dir,
        neg_ratio=args.neg_ratio,
        output_dir=args.output_dir,
        training_details_dir=args.training_details_dir,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
    )

    if args.device:
        config.device = args.device
    if args.no_amp:
        config.use_amp = False

    # Create the output directory
    output_dir = Path(config.output_dir) / f"neg_ratio_{config.neg_ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the training-details directory
    training_details_dir = Path(config.training_details_dir) / f"neg_ratio_{config.neg_ratio}"
    training_details_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    logger = setup_logging(output_dir)
    global LOGGER
    LOGGER = logger
    faulthandler.enable(all_threads=True)

    def _signal_handler(signum, frame):
        try:
            logger.error("Received signal %s. Exiting...", signum)
            _log_mem(logger, config.device, prefix=f"SIGNAL {signum}")
        except Exception:
            pass
        raise SystemExit(1)

    for _sig in [signal.SIGTERM, signal.SIGINT]:
        try:
            signal.signal(_sig, _signal_handler)
        except Exception:
            pass


    # Save the configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)


    logger.info("Config: %s", json.dumps(asdict(config), ensure_ascii=False, indent=2))
    log_environment(logger, config)

    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    print("=" * 80)
    print("Stage 1: Sequence Foundation Training")
    print("=" * 80)
    print(f"Data directory: {config.data_dir}")
    print(f"Negative sample ratio: 1:{config.neg_ratio}")
    print(f"Device: {config.device}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Output directory: {output_dir}")
    print(f"Training details directory: {training_details_dir}")
    print("=" * 80)

    # Load the dataset
    dataset = pHLATCRDataset(
        data_dir=config.data_dir,
        neg_ratio=config.neg_ratio,
        random_seed=config.random_seed
    )
    labels = np.array(dataset.get_labels())

    print(f"\nDataset statistics:")
    logger.info("Dataset loaded. len=%d pos=%d neg=%d ratio=1:%.2f", len(dataset), int((labels==1).sum()), int((labels==0).sum()), float((labels==0).sum()/(labels==1).sum()))
    _log_mem(logger, config.device, prefix='AFTER_DATASET')
    print(f"  Total samples: {len(dataset)}")
    print(f"  Positive (label=1): {(labels == 1).sum()}")
    print(f"  Negative (label=0): {(labels == 0).sum()}")
    print(f"  Actual pos:neg ratio: 1:{(labels == 0).sum() / (labels == 1).sum():.1f}")

    # Create StratifiedKFold
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    # Store fold results
    fold_results = []

    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        if args.fold is not None and fold != args.fold:
            continue

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        # Save fold indices
        torch.save({
            'train_idx': train_idx,
            'val_idx': val_idx,
        }, fold_dir / "fold_indices.pt")

        # Train this fold
        fold_result = train_single_fold(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            dataset=dataset,
            config=config,
            fold_dir=fold_dir,
        )

        fold_results.append(fold_result)

    # Summarize results
    if args.fold is None and len(fold_results) > 0:
        print("\n" + "=" * 80)
        print(f"Stage 1 Training Summary (Neg Ratio 1:{config.neg_ratio})")
        print("=" * 80)

        aurocs = [r['final_metrics']['auroc'] for r in fold_results]
        auprcs = [r['final_metrics']['auprc'] for r in fold_results]

        for i, result in enumerate(fold_results):
            print(f"\nFold {i+1}:")
            print(f"  Best epoch: {result['best_epoch']}")
            print(f"  AUROC: {result['final_metrics']['auroc']:.4f}")
            print(f"  AUPRC: {result['final_metrics']['auprc']:.4f}")

        print("\n" + "-" * 80)
        print("Average Performance:")
        print(f"  AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
        print(f"  AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")

        # Save the summary
        summary = {
            'neg_ratio': config.neg_ratio,
            'fold_results': fold_results,
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'mean_auprc': float(np.mean(auprcs)),
            'std_auprc': float(np.std(auprcs)),
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Also save the summary in the training-details directory
        with open(training_details_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\nStage 1 training completed! Results saved to: {output_dir}")
    print(f"Training details saved to: {training_details_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
