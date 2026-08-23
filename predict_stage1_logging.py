# predict_stage1_logging.py
# -*- coding: utf-8 -*-
"""
Stage 1 prediction module.

Features:
1. Provides Stage1Predictor for model loading and inference.
2. Supports single-sample and batch prediction.
3. Uses a five-fold ensemble.

Example:
    from predict_stage1_logging import Stage1Predictor

    # Initialize the predictor
    predictor = Stage1Predictor(
        model_dir="./runs/stage1_ratio10_B128/neg_ratio_10",
        device="cuda:0"
    )

    # Predict one sample
    prob = predictor.predict_single(
        epitope="GILGFVFTL",
        hla="GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRW",
        tcra="GQQVMQIPQYQHVQEGEDFTTYCNSSTTLSNIQWYKQRPGGHPVFLIQLVKSGEVKKQKRLTFQFGEAKKNSSLHITATQTTDVGTYFCAGPTNAGKSTFGDGTTLTVKP",
        tcrb="DGGITQSPKYLFRKEGQNVTLSCEQNLNHDAMYWYRQDPGQGLRLIYYSQIVNDFQKGDIAEGYSVSREKKESFPLTVTSAQKNPTAFYLCASSGISTDTQYFGPGTRLTVLE"
    )

    # Predict a batch
    probs = predictor.predict_batch(
        epitopes=["GILGFVFTL", "NLVPMVATV"],
        hlas=[hla1, hla2],
        tcras=[tcra1, tcra2],
        tcrbs=[tcrb1, tcrb2]
    )
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ESM-C
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Import the model
from Network_v3 import Network


# =========================
#      Logging utilities
# =========================
def setup_logging(name: str = "Stage1Predictor") -> logging.Logger:
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
#      Stage 1 predictor
# =========================
class Stage1Predictor:
    """
    Load a five-fold ensemble and predict from input sequences.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        esm_model_name: str = "esmc_600m",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the predictor.

        Args:
            model_dir: Model directory containing fold_0 through fold_4.
            device: Inference device.
            esm_model_name: ESM model name.
            logger: Optional logger.
        """
        self.model_dir = Path(model_dir)
        # Normalize the device as torch.device for consistent device handling
        self.device = torch.device(device)
        self.esm_model_name = esm_model_name
        self.logger = logger if logger is not None else setup_logging()

        # Model configuration matching training
        self.hid = 256
        self.seq_nhead = 8
        self.seq_dropout = 0.1
        self.phla_seq_layers = 1
        self.tcr_seq_layers = 1
        self.num_feature_type = 5

        # Fixed lengths matching training
        self.peptide_max_len = 15
        self.hla_len = 276
        self.tcra_max_len = 127
        self.tcrb_max_len = 130

        # Load models
        self.logger.info("=" * 60)
        self.logger.info("Initializing Stage1Predictor...")
        self.logger.info(f"Model directory: {model_dir}")
        self.logger.info(f"Device: {device}")

        # Set the CUDA context so ESM does not create tensors on the default device
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            # Clear the selected device's cache
            torch.cuda.empty_cache()

        # Load the ESM-C model
        self.logger.info(f"Loading ESM-C model: {esm_model_name}")
        start_time = time.time()
        self.esm_client = ESMC.from_pretrained(esm_model_name).to(self.device)
        self.esm_client.eval()
        self.logger.info(f"  ESM-C model loaded in {time.time() - start_time:.2f}s")

        # Load five fold models
        self.models = self._load_fold_models()
        self.logger.info("Stage1Predictor initialized successfully")
        self.logger.info("=" * 60)

    def _load_fold_models(self) -> List[nn.Module]:
        """Load the five fold models."""
        models = []

        self.logger.info("Loading 5-fold models...")

        for fold in range(5):
            fold_dir = self.model_dir / f"fold_{fold}"
            model_path = fold_dir / "best_model.pt"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.logger.info(f"  Loading fold {fold} from: {model_path}")

            # Create the model
            model = Network(
                hid=self.hid,
                seq_nhead=self.seq_nhead,
                seq_dropout=self.seq_dropout,
                phla_seq_layers=self.phla_seq_layers,
                tcr_seq_layers=self.tcr_seq_layers,
                num_feature_type=self.num_feature_type,
            )

            # Load weights
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Move to the device and enable evaluation mode
            model = model.to(self.device)
            model.eval()

            self.logger.info(f"    Best epoch: {checkpoint['epoch']}, Val AUROC: {checkpoint['val_metrics']['auroc']:.4f}")

            models.append(model)

        self.logger.info(f"  All 5 models loaded successfully")

        return models

    def _generate_esm_embedding(self, sequence: str) -> torch.Tensor:
        """
        Generate an ESM embedding for one sequence.

        Args:
            sequence: Amino-acid sequence.

        Returns:
            embedding: [seq_len+2, 1152], including BOS/EOS, on the selected device.
        """
        # Run in the selected device context
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                with torch.no_grad():
                    protein = ESMProtein(sequence=sequence)
                    protein_tensor = self.esm_client.encode(protein)
                    logits_output = self.esm_client.logits(
                        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    # [1, seq_len+2, 1152] -> [seq_len+2, 1152]
                    embedding = logits_output.embeddings[0]
        else:
            with torch.no_grad():
                protein = ESMProtein(sequence=sequence)
                protein_tensor = self.esm_client.encode(protein)
                logits_output = self.esm_client.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                # [1, seq_len+2, 1152] -> [seq_len+2, 1152]
                embedding = logits_output.embeddings[0]

        # Return the embedding on the selected device without moving it to CPU
        return embedding

    def _pad_and_mask(
        self,
        embedding: torch.Tensor,
        max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad an embedding and create its validity mask.

        Args:
            embedding: [L, 1152]
            max_len: Target length.

        Returns:
            padded_emb: [max_len, 1152]
            mask: [max_len], where True is valid and False is padding.
        """
        L = embedding.size(0)

        # Padding
        if L < max_len:
            padded_emb = F.pad(embedding, (0, 0, 0, max_len - L))
        else:
            padded_emb = embedding[:max_len]

        # Mask
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:min(L, max_len)] = True

        return padded_emb, mask

    def _prepare_batch_input(
        self,
        epitopes: List[str],
        hlas: List[str],
        tcras: List[str],
        tcrbs: List[str]
    ) -> dict:
        """
        Prepare batched model inputs.

        Args:
            epitopes: List of epitope sequences.
            hlas: List of HLA sequences.
            tcras: List of TCR alpha-chain sequences.
            tcrbs: List of TCR beta-chain sequences.

        Returns:
            batch_dict: Dictionary containing all model inputs.
        """
        batch_size = len(epitopes)

        # Generate ESM embeddings
        self.logger.info(f"Generating ESM embeddings for {batch_size} samples...")

        peptide_embs = []
        peptide_masks = []
        hla_embs = []
        hla_masks = []
        tcra_embs = []
        tcra_masks = []
        tcrb_embs = []
        tcrb_masks = []

        for i in tqdm(range(batch_size), desc="Processing sequences"):
            # Peptide
            pep_emb = self._generate_esm_embedding(epitopes[i])
            pep_emb_padded, pep_mask = self._pad_and_mask(pep_emb, self.peptide_max_len)
            peptide_embs.append(pep_emb_padded)
            peptide_masks.append(pep_mask)

            # HLA
            hla_emb = self._generate_esm_embedding(hlas[i])
            hla_emb_padded, hla_mask = self._pad_and_mask(hla_emb, self.hla_len)
            hla_embs.append(hla_emb_padded)
            hla_masks.append(hla_mask)

            # TCR alpha chain
            tcra_emb = self._generate_esm_embedding(tcras[i])
            tcra_emb_padded, tcra_mask = self._pad_and_mask(tcra_emb, self.tcra_max_len)
            tcra_embs.append(tcra_emb_padded)
            tcra_masks.append(tcra_mask)

            # TCR beta chain
            tcrb_emb = self._generate_esm_embedding(tcrbs[i])
            tcrb_emb_padded, tcrb_mask = self._pad_and_mask(tcrb_emb, self.tcrb_max_len)
            tcrb_embs.append(tcrb_emb_padded)
            tcrb_masks.append(tcrb_mask)

        # Stack into a batch; embeddings are already on the correct device
        batch_dict = {
            'peptide_emb': torch.stack(peptide_embs, dim=0),      # [B, 15, 1152]
            'peptide_mask': torch.stack(peptide_masks, dim=0).to(self.device),    # [B, 15]
            'hla_emb': torch.stack(hla_embs, dim=0),              # [B, 276, 1152]
            'hla_mask': torch.stack(hla_masks, dim=0).to(self.device),            # [B, 276]
            'tcra_emb': torch.stack(tcra_embs, dim=0),            # [B, 127, 1152]
            'tcra_mask': torch.stack(tcra_masks, dim=0).to(self.device),          # [B, 127]
            'tcrb_emb': torch.stack(tcrb_embs, dim=0),            # [B, 130, 1152]
            'tcrb_mask': torch.stack(tcrb_masks, dim=0).to(self.device),          # [B, 130]
        }

        return batch_dict

    @torch.no_grad()
    def predict_batch(
        self,
        epitopes: List[str],
        hlas: List[str],
        tcras: List[str],
        tcrbs: List[str]
    ) -> List[float]:
        """
        Predict a batch.

        Args:
            epitopes: List of epitope sequences.
            hlas: List of HLA sequences.
            tcras: List of TCR alpha-chain sequences.
            tcrbs: List of TCR beta-chain sequences.

        Returns:
            predictions: List of predicted probabilities.
        """
        # Validate inputs
        assert len(epitopes) == len(hlas) == len(tcras) == len(tcrbs), \
            "All input lists must have the same length"

        batch_size = len(epitopes)
        self.logger.info(f"Predicting {batch_size} samples...")

        # Prepare inputs
        start_time = time.time()
        batch_dict = self._prepare_batch_input(epitopes, hlas, tcras, tcrbs)
        self.logger.info(f"  Input preparation completed in {time.time() - start_time:.2f}s")

        # Run inference with each fold model
        start_time = time.time()
        all_logits = []  # [5, B]

        for fold_idx, model in enumerate(self.models):
            output = model(
                peptide_emb=batch_dict['peptide_emb'],
                hla_emb=batch_dict['hla_emb'],
                tcra_emb=batch_dict['tcra_emb'],
                tcrb_emb=batch_dict['tcrb_emb'],
                peptide_mask=batch_dict['peptide_mask'],
                hla_mask=batch_dict['hla_mask'],
                tcra_mask=batch_dict['tcra_mask'],
                tcrb_mask=batch_dict['tcrb_mask'],
                mode="seq_only",  # Stage 1 uses seq_only mode
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
        epitope: str,
        hla: str,
        tcra: str,
        tcrb: str
    ) -> float:
        """
        Predict one sample.

        Args:
            epitope: Epitope sequence.
            hla: HLA sequence.
            tcra: TCR alpha-chain sequence.
            tcrb: TCR beta-chain sequence.

        Returns:
            prediction: Predicted probability.
        """
        # Reuse the batch interface with batch_size=1
        predictions = self.predict_batch(
            epitopes=[epitope],
            hlas=[hla],
            tcras=[tcra],
            tcrbs=[tcrb]
        )

        return predictions[0]


# =========================
#      Example CLI
# =========================
if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1 Predictor Test')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory (containing fold_0 to fold_4)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0, cuda:1, cpu, etc.)')

    args = parser.parse_args()

    # Initialize the predictor
    predictor = Stage1Predictor(
        model_dir=args.model_dir,
        device=args.device
    )

    # Test single-sample prediction
    print("\n" + "=" * 60)
    print("Testing single sample prediction...")

    epitope = "GILGFVFTL"
    hla = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRW"
    tcra = "GQQVMQIPQYQHVQEGEDFTTYCNSSTTLSNIQWYKQRPGGHPVFLIQLVKSGEVKKQKRLTFQFGEAKKNSSLHITATQTTDVGTYFCAGPTNAGKSTFGDGTTLTVKP"
    tcrb = "DGGITQSPKYLFRKEGQNVTLSCEQNLNHDAMYWYRQDPGQGLRLIYYSQIVNDFQKGDIAEGYSVSREKKESFPLTVTSAQKNPTAFYLCASSGISTDTQYFGPGTRLTVLE"

    prob = predictor.predict_single(epitope, hla, tcra, tcrb)
    print(f"Prediction probability: {prob:.6f}")

    # Test batch prediction
    print("\n" + "=" * 60)
    print("Testing batch prediction...")

    epitopes = ["GILGFVFTL", "NLVPMVATV", "GALGFVFTL"]
    hlas = [hla, hla, hla]
    tcras = [tcra, tcra, tcra]
    tcrbs = [tcrb, tcrb, tcrb]

    probs = predictor.predict_batch(epitopes, hlas, tcras, tcrbs)
    print(f"Batch predictions: {probs}")

    print("=" * 60)
