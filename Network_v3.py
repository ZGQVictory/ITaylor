# Network_v3.py extends Network_v2.py with surf_only mode.
# Main changes:
# 1. _extract_surf_embeddings directly returns five MaSIF feature streams without attention.
# 2. forward supports surf_only classification from structural information alone.
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- MaSIFCore implementation ----
from MaSIFCore_v2 import MaSIFCore  # v2 supports a true batch dimension and vectorized rotations


# -------------------------
# Utility for counting model parameters
# -------------------------
def _num_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

def summarize_model(module, name="Network", indent=0, depth=4, trainable_only=False):
    """
    Recursively summarize model parameters as multiple lines.

    - name: Root module name.
    - depth: Maximum recursion depth; large values produce lengthy output.
    """
    lines = []
    prefix = "  " * indent
    cls = module.__class__.__name__
    n = _num_params(module, trainable_only)
    lines.append(f"{prefix}{name} ({cls})  params={n:,}")

    if depth <= 0:
        return lines

    for child_name, child in module.named_children():
        child_n = _num_params(child, trainable_only)
        # Include a layer only when it has parameters or child modules
        has_kids = any(True for _ in child.children())
        if child_n > 0 or has_kids:
            lines.extend(summarize_model(
                child,
                name=child_name,
                indent=indent+1,
                depth=depth-1,
                trainable_only=trainable_only
            ))
    return lines

# -------------------------
# Transformer encoder layer that can return attention matrices
# -------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 4_096,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,               # [B, L, D]
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, L]; True marks masked positions
        attn_mask: Optional[torch.Tensor] = None,             # [L, L] or [B*num_heads, L, L]
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.norm1(src)
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=False  # [B, H, L, L]
        )
        src = src + self.dropout1(attn_out)

        y = self.norm2(src)
        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        src = src + self.dropout2(y)

        return src, attn_weights  # [B, H, L, L]


class TransformerEncoderWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 4_096,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model, nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, L]
        attn_mask: Optional[torch.Tensor] = None,
        collect_all_layers: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_all: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask, attn_mask, need_weights=True)
            if collect_all_layers:
                attn_all.append(attn)  # [B, H, L, L]
        x = self.final_norm(x)
        return x, attn_all


# -------------------------
# Parameter-free sinusoidal patch-query embeddings of variable length
# -------------------------
def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dim, device=device)
    position = torch.arange(length, device=device).unsqueeze(1)  # [L,1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# -------------------------
# Masked mean pooling
# -------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute mean pooling over valid masked positions.

    Args:
        x: Input tensor with shape [B, L, D] or [B, L].
        mask: Boolean [B, L] mask; True is valid and False is padding.
        dim: Dimension to average over; defaults to sequence dimension 1.
        keepdim: Whether to retain the reduced dimension.

    Returns:
        pooled: [B, D] or [B], depending on the input rank and keepdim.
    """
    mask = mask.bool()

    if x.dim() == 3 and mask.dim() == 2:
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    elif x.dim() == 2 and mask.dim() == 2:
        mask_expanded = mask
    else:
        mask_expanded = mask

    masked_x = x * mask_expanded
    valid_counts = mask.sum(dim=dim, keepdim=keepdim)  # [B] or [B, 1]
    valid_counts = valid_counts.clamp(min=1)

    if x.dim() == 3:
        sum_x = masked_x.sum(dim=dim, keepdim=keepdim)  # [B, D] or [B, 1, D]
        if keepdim:
            result = sum_x / valid_counts.unsqueeze(-1)
        else:
            result = sum_x / valid_counts.unsqueeze(-1)
    else:
        sum_x = masked_x.sum(dim=dim, keepdim=keepdim)  # [B] or [B, 1]
        result = sum_x / valid_counts

    return result


# -------------------------
# MaSIF parameter container registered as nn.Module for parameter tracking
# -------------------------
class _MasifParamModule(nn.Module):
    def __init__(self, coords: torch.Tensor, max_rho: float, n_gauss: int):
        super().__init__()
        self.mu_rho   = nn.Parameter(coords[:, 0][None, :].float())     # [1, G]
        self.mu_theta = nn.Parameter(coords[:, 1][None, :].float())     # [1, G]
        self.sigma_rho   = nn.Parameter(torch.ones_like(self.mu_rho) * (max_rho / 8.0))
        self.sigma_theta = nn.Parameter(torch.ones_like(self.mu_theta) * 1.0)
        self.W_conv = nn.Parameter(torch.empty(n_gauss, n_gauss))
        nn.init.xavier_normal_(self.W_conv)
        self.b_conv = nn.Parameter(torch.zeros(n_gauss))


# =========================
#         Main network
# =========================
class Network(nn.Module):
    """
    Multi-stream pHLA/TCR MaSIF and sequence-fusion pipeline.

    Version 3 adds surf_only mode, which classifies directly from MaSIF
    features without sequence information.
    """

    def __init__(
        self,
        # ESM and Transformer dimensions
        hid: int = 256,
        seq_nhead: int = 8,
        seq_dropout: float = 0.1,
        phla_seq_layers: int = 1,
        tcr_seq_layers: int = 1,

        # MaSIF hyperparameters
        n_thetas: int = 16,
        n_rhos: int = 5,
        n_rotations: int = 16,
        max_rho_phla: float = 12.0,
        max_rho_tcr: float = 12.0,

        # Joint patch-level encoder
        joint_nhead: int = 10,  # Divides both joint_d=90 and n_gauss=80
        joint_layers: int = 1,
        joint_dropout: float = 0.1,
        ff_dim_scale: float = 4.0,
        activation: str = "gelu",

        # Number of feature types
        num_feature_type: int = 5,

        # Feature-type aggregation encoder
        ft_nhead: int = 10,  # Divides both joint_d=90 and n_gauss=80
        ft_layers: int = 1,
        ft_dropout: float = 0.1,
    ):
        super().__init__()
        self.hid = hid
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos
        self.n_rotations = n_rotations
        self.n_gauss = n_thetas * n_rhos            # G
        self.num_feature_type = num_feature_type
        self.joint_d = self.n_gauss + 10            # D_joint

        # ---- ESM projection: 1152 -> hid ----
        self.esm_proj = nn.Linear(1152, hid)

        # ---- Independent sequence, patch, MaSIFCore, and joint modules per stream ----
        # Sequence encoders and 10D projections
        self.phla_seq_encoder = TransformerEncoderWithAttn(
                d_model=hid, nhead=seq_nhead, num_layers=phla_seq_layers,
                dim_feedforward=int(hid * 4), dropout=seq_dropout, activation=activation
            )

        self.tcra_seq_encoder = TransformerEncoderWithAttn(
                d_model=hid, nhead=seq_nhead, num_layers=tcr_seq_layers,
                dim_feedforward=int(hid * 4), dropout=seq_dropout, activation=activation
            )
        self.tcrb_seq_encoder = TransformerEncoderWithAttn(
                d_model=hid, nhead=seq_nhead, num_layers=tcr_seq_layers,
                dim_feedforward=int(hid * 4), dropout=seq_dropout, activation=activation
            )

        # ---- TCR joint encoder over concatenated alpha and beta chains ----
        tcr_joint_layer = nn.TransformerEncoderLayer(
            d_model=hid,
            nhead=seq_nhead,
            dim_feedforward=int(hid * 4),
            dropout=seq_dropout,
            activation=activation,
            batch_first=True
        )
        self.tcr_joint_encoder = nn.TransformerEncoder(
            encoder_layer=tcr_joint_layer,
            num_layers=tcr_seq_layers
        )

        # Distinguish alpha and beta chains
        self.tcr_chain_type_emb = nn.Embedding(2, hid)  # 0: alpha, 1: beta


        self.phla_patch_fcs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 10), nn.GELU())
            for _ in range(num_feature_type)
        ])
        self.tcr_patch_fcs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 10), nn.GELU())
            for _ in range(num_feature_type)
        ])

        # Independent pHLA and TCR MaSIFCore modules and parameters per stream
        self.core_phla_list = nn.ModuleList([
            MaSIFCore(max_rho=max_rho_phla, n_thetas=n_thetas, n_rhos=n_rhos, n_rotations=n_rotations)
            for _ in range(num_feature_type)
        ])
        self.core_tcr_list = nn.ModuleList([
            MaSIFCore(max_rho=max_rho_tcr, n_thetas=n_thetas, n_rhos=n_rhos, n_rotations=n_rotations)
            for _ in range(num_feature_type)
        ])
        self.masif_phla_params = nn.ModuleList()
        self.masif_tcr_params  = nn.ModuleList()
        for i in range(num_feature_type):
            coords_phla = torch.from_numpy(self.core_phla_list[i].compute_initial_coordinates())  # [G,2]
            coords_tcr  = torch.from_numpy(self.core_tcr_list[i].compute_initial_coordinates())   # [G,2]
            self.masif_phla_params.append(_MasifParamModule(coords_phla, max_rho_phla, self.n_gauss))
            self.masif_tcr_params.append(_MasifParamModule(coords_tcr,  max_rho_tcr,  self.n_gauss))

        # Per-stream pHLA/TCR patch joint encoders and output normalization for joint mode
        self.joint_encoders = nn.ModuleList([
            TransformerEncoderWithAttn(
                d_model=self.joint_d, nhead=joint_nhead, num_layers=joint_layers,
                dim_feedforward=int(self.joint_d * ff_dim_scale),
                dropout=joint_dropout, activation=activation
            ) for _ in range(num_feature_type)
        ])
        self.out_norms = nn.ModuleList([nn.LayerNorm(self.joint_d) for _ in range(num_feature_type)])

        # Feature-type aggregation encoder and classification head for joint mode
        self.ft_encoder = TransformerEncoderWithAttn(
            d_model=self.joint_d, nhead=ft_nhead, num_layers=ft_layers,
            dim_feedforward=int(self.joint_d * 4), dropout=ft_dropout, activation=activation
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.joint_d),
            nn.Linear(self.joint_d, self.joint_d // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.joint_d // 2, 1)  # Output logit
        )

        # ---- Stage 1 sequence-only classification head ----
        # Classification head independent of MaSIF
        # Attention module for fusing pHLA and TCR sequence representations
        self.seq_attn = nn.MultiheadAttention(
            embed_dim=hid,
            num_heads=seq_nhead,
            dropout=seq_dropout,
            batch_first=True
        )
        self.seq_attn_norm = nn.LayerNorm(hid)

        self.seq_cls_head = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid // 2, 1)  # Output logit
        )

        # ---- TCR alpha/beta cross-attention fusion ----
        # Alpha chain attends to the beta chain
        self.tcr_cross_attn = nn.MultiheadAttention(
            embed_dim=hid,
            num_heads=seq_nhead,
            dropout=seq_dropout,
            batch_first=True
        )
        self.tcr_cross_norm = nn.LayerNorm(hid)
        # FFN for post-attention fusion
        self.tcr_cross_ffn = nn.Sequential(
            nn.Linear(hid, hid * 4),
            nn.GELU(),
            nn.Dropout(seq_dropout),
            nn.Linear(hid * 4, hid)
        )
        self.tcr_cross_ffn_norm = nn.LayerNorm(hid)
        # Final fusion: concatenate alpha_global and beta_global into tcr_global
        self.tcr_fusion = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.GELU(),
            nn.Dropout(seq_dropout),
        )

        # ---- Stage 2: CLIP Projection Heads ----
        # Projection heads for sequence-structure alignment
        clip_dim = 128  # CLIP embedding dimension
        self.phla_seq_proj = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Linear(hid, 256),
            nn.GELU(),
            nn.Linear(256, clip_dim)
        )
        self.tcr_seq_proj = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Linear(hid, 256),
            nn.GELU(),
            nn.Linear(256, clip_dim)
        )
        self.phla_surf_proj = nn.Sequential(
            nn.LayerNorm(self.n_gauss),
            nn.Linear(self.n_gauss, 256),
            nn.GELU(),
            nn.Linear(256, clip_dim)
        )
        self.tcr_surf_proj = nn.Sequential(
            nn.LayerNorm(self.n_gauss),
            nn.Linear(self.n_gauss, 256),
            nn.GELU(),
            nn.Linear(256, clip_dim)
        )

        # ---- Modules dedicated to v3 surf_only mode ----
        # Per-stream pHLA/TCR patch joint encoders with dimension G and no seq10
        self.surf_joint_encoders = nn.ModuleList([
            TransformerEncoderWithAttn(
                d_model=self.n_gauss, nhead=joint_nhead, num_layers=joint_layers,
                dim_feedforward=int(self.n_gauss * ff_dim_scale),
                dropout=joint_dropout, activation=activation
            ) for _ in range(num_feature_type)
        ])
        self.surf_out_norms = nn.ModuleList([nn.LayerNorm(self.n_gauss) for _ in range(num_feature_type)])

        # Feature-type aggregation encoder with dimension G
        self.surf_ft_encoder = TransformerEncoderWithAttn(
            d_model=self.n_gauss, nhead=ft_nhead, num_layers=ft_layers,
            dim_feedforward=int(self.n_gauss * 4), dropout=ft_dropout, activation=activation
        )

        # Classification head dedicated to surf_only mode
        self.surf_cls_head = nn.Sequential(
            nn.LayerNorm(self.n_gauss),
            nn.Linear(self.n_gauss, self.n_gauss // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.n_gauss // 2, 1)
        )

    # ---------------- Batched MaSIFCore wrapper ----------------
    def _masif_batch_wrapper(
        self,
        core: nn.Module,
        params: nn.Module,
        input_feat: torch.Tensor,    # [B, P, V]
        rho_coords: torch.Tensor,    # [B, P, V]
        theta_coords: torch.Tensor,  # [B, P, V]
        patch_mask: torch.Tensor,    # [B, P] network-level patch mask
        mean_gauss_activation: bool = True,
    ) -> torch.Tensor:
        """
        Apply MaSIFCore to a batch without a Python loop.

        Args:
            core: MaSIFCore instance.
            params: _MasifParamModule instance.
            input_feat:  [B, P, V]
            rho_coords:  [B, P, V]
            theta_coords:[B, P, V]
            patch_mask: [B, P] patch mask; True is valid and False is padding.
            mean_gauss_activation: MaSIF normalization option.

        Returns:
            masif_feats: Batched patch-level MaSIF features with shape [B, P, G].
        """
        B, P, V = input_feat.shape

        # patch mask -> vertex mask: [B, P, V, 1]
        vertex_mask = patch_mask[:, :, None, None].expand(B, P, V, 1)

        # MaSIFCore v2 expects inputs with shape [B, P, V, 1]
        masif_feats_batch = core.masif_inf(
            input_feat=input_feat.unsqueeze(-1),        # [B, P, V, 1]
            rho_coords=rho_coords.unsqueeze(-1),        # [B, P, V, 1]
            theta_coords=theta_coords.unsqueeze(-1),    # [B, P, V, 1]
            mask=vertex_mask,                           # [B, P, V, 1]
            W_conv=params.W_conv,
            b_conv=params.b_conv,
            mu_rho=params.mu_rho,
            sigma_rho=params.sigma_rho,
            mu_theta=params.mu_theta,
            sigma_theta=params.sigma_theta,
            eps=1e-5,
            mean_gauss_activation=mean_gauss_activation,
            vectorized_rotations=True,
        )  # [B, P, G]

        return masif_feats_batch

    # ---------- Sequence-only embeddings computed once for joint/patch broadcasting ----------
    def _extract_seq_embeddings(
        self,
        peptide_emb: torch.Tensor,    # [B, 15, 1152]
        hla_emb: torch.Tensor,        # [B, 276, 1152]
        tcra_emb: torch.Tensor,       # [B, 127, 1152]
        tcrb_emb: torch.Tensor,       # [B, 130, 1152]
        peptide_mask: torch.Tensor,   # [B, 15]
        hla_mask: torch.Tensor,       # [B, 276]
        tcra_mask: torch.Tensor,      # [B, 127]
        tcrb_mask: torch.Tensor,      # [B, 130]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pHLA and TCR sequence embeddings.

        Returns:
            phla_seq_emb: [B, hid]
            tcr_seq_emb:  [B, hid]
        """
        device = next(self.parameters()).device

        # ---- pHLA sequence encoding: peptide plus HLA ----
        pep_tok = self.esm_proj(peptide_emb.to(device))  # [B, 15, hid]
        hla_tok = self.esm_proj(hla_emb.to(device))      # [B, 276, hid]
        phla_tokens = torch.cat([pep_tok, hla_tok], dim=1)  # [B, L_phla, hid]
        phla_mask_combined = torch.cat([peptide_mask, hla_mask], dim=1)  # [B, L_phla]
        phla_key_padding_mask = ~phla_mask_combined.bool()  # [B, L_phla]

        phla_enc, _ = self.phla_seq_encoder(
            phla_tokens,
            key_padding_mask=phla_key_padding_mask
        )  # [B, L_phla, hid]
        phla_seq_emb = masked_mean(phla_enc, phla_mask_combined, dim=1)  # [B, hid]

        # ---- Encode TCR alpha/beta chains separately, then fuse with the joint encoder ----
        tcra_tok = self.esm_proj(tcra_emb.to(device))    # [B, L_tcra, hid]
        tcrb_tok = self.esm_proj(tcrb_emb.to(device))    # [B, L_tcrb, hid]

        tcra_key_padding_mask = ~tcra_mask.bool()
        tcrb_key_padding_mask = ~tcrb_mask.bool()

        tcra_enc, _ = self.tcra_seq_encoder(
            tcra_tok,
            key_padding_mask=tcra_key_padding_mask
        )  # [B, L_tcra, hid]
        tcrb_enc, _ = self.tcrb_seq_encoder(
            tcrb_tok,
            key_padding_mask=tcrb_key_padding_mask
        )  # [B, L_tcrb, hid]

        tcr_tokens = torch.cat([tcra_enc, tcrb_enc], dim=1)  # [B, L_a+L_b, hid]
        tcr_key_padding_mask = torch.cat([tcra_key_padding_mask, tcrb_key_padding_mask], dim=1)  # [B, L_a+L_b]

        L_a = tcra_enc.size(1)
        L_b = tcrb_enc.size(1)
        chain_type_ids = torch.cat([
            torch.zeros((tcr_tokens.size(0), L_a), dtype=torch.long, device=tcr_tokens.device),
            torch.ones((tcr_tokens.size(0), L_b), dtype=torch.long, device=tcr_tokens.device),
        ], dim=1)  # [B, L_a+L_b]
        tcr_tokens = tcr_tokens + self.tcr_chain_type_emb(chain_type_ids)

        # PyTorch TransformerEncoder forward pass
        tcr_joint = self.tcr_joint_encoder(
            tcr_tokens,
            src_key_padding_mask=tcr_key_padding_mask
        )  # [B, L_a+L_b, hid]

        # Split the result back into alpha and beta segments
        tcra_joint = tcr_joint[:, :L_a, :]   # [B, L_a, hid]
        tcrb_joint = tcr_joint[:, L_a:, :]   # [B, L_b, hid]

        # Masked pooling
        tcra_global = masked_mean(tcra_joint, tcra_mask, dim=1)  # [B, hid]
        tcrb_global = masked_mean(tcrb_joint, tcrb_mask, dim=1)  # [B, hid]

        tcr_combined = torch.cat([tcra_global, tcrb_global], dim=-1)  # [B, 2*hid]
        tcr_seq_emb = self.tcr_fusion(tcr_combined)                   # [B, hid]

        return phla_seq_emb, tcr_seq_emb


    # ---------------- Single-stream processing with batching and masks ----------------
    def _separate_feature_one(
        self,
        idx: int,
        peptide_emb: torch.Tensor,    # [B, 15, 1152]
        hla_emb: torch.Tensor,        # [B, 276, 1152]
        tcra_emb: torch.Tensor,       # [B, 127, 1152]
        tcrb_emb: torch.Tensor,       # [B, 130, 1152]
        peptide_mask: torch.Tensor,   # [B, 15]
        hla_mask: torch.Tensor,       # [B, 276]
        tcra_mask: torch.Tensor,      # [B, 127]
        tcrb_mask: torch.Tensor,      # [B, 130]
        phla_masif: Dict[str, torch.Tensor],
        tcr_masif: Dict[str, torch.Tensor],
        mean_gauss_activation: bool = True,
        phla_seq_global: Optional[torch.Tensor] = None,  # [B, hid]
        tcr_seq_global: Optional[torch.Tensor] = None,   # [B, hid]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process feature stream ``idx`` with batched inputs and masks.

        Returns:
            phla_feat: [B, Pp_max, G+10]
            tcr_feat:  [B, Pt,     G+10]
            phla_patch_mask: [B, Pp_max]
            tcr_patch_mask:  [B, Pt]
        """
        device = next(self.parameters()).device

        # ---- Compute global sequence encodings once when not supplied ----
        if phla_seq_global is None or tcr_seq_global is None:
            phla_seq_global, tcr_seq_global = self._extract_seq_embeddings(
                peptide_emb, hla_emb, tcra_emb, tcrb_emb,
                peptide_mask, hla_mask, tcra_mask, tcrb_mask
            )  # [B, hid], [B, hid]

        phla_seq_global = phla_seq_global.to(device)
        tcr_seq_global = tcr_seq_global.to(device)
        B = phla_seq_global.size(0)

        # ---- pHLA patch seq10: sinusoidal positions, broadcasting, and FC ----
        Pp_max = phla_masif["input_feat"].size(1)
        phla_queries_base = sinusoidal_positions(Pp_max, self.hid, device)  # [Pp_max, hid]
        phla_queries = phla_queries_base.unsqueeze(0).expand(B, -1, -1) + phla_seq_global.unsqueeze(1)  # [B, Pp_max, hid]
        phla_seq10 = self.phla_patch_fcs[idx](phla_queries)  # [B, Pp_max, 10]

        # ---- TCR patch seq10: sinusoidal positions, broadcasting, and FC ----
        Pt = tcr_masif["input_feat"].size(1)
        tcr_queries_base = sinusoidal_positions(Pt, self.hid, device)  # [Pt, hid]
        tcr_queries = tcr_queries_base.unsqueeze(0).expand(B, -1, -1) + tcr_seq_global.unsqueeze(1)  # [B, Pt, hid]
        tcr_seq10 = self.tcr_patch_fcs[idx](tcr_queries)  # [B, Pt, 10]

        # ---- MaSIF through the batch wrapper ----
        phla_params = self.masif_phla_params[idx]
        tcr_params = self.masif_tcr_params[idx]

        phla_masif_feat = self._masif_batch_wrapper(
            core=self.core_phla_list[idx],
            params=phla_params,
            input_feat=phla_masif["input_feat"].to(device),
            rho_coords=phla_masif["rho_coords"].to(device),
            theta_coords=phla_masif["theta_coords"].to(device),
            patch_mask=phla_masif["mask"].to(device),
            mean_gauss_activation=mean_gauss_activation,
        )  # [B, Pp_max, G]

        tcr_masif_feat = self._masif_batch_wrapper(
            core=self.core_tcr_list[idx],
            params=tcr_params,
            input_feat=tcr_masif["input_feat"].to(device),
            rho_coords=tcr_masif["rho_coords"].to(device),
            theta_coords=tcr_masif["theta_coords"].to(device),
            patch_mask=tcr_masif["mask"].to(device),
            mean_gauss_activation=mean_gauss_activation,
        )  # [B, Pt, G]

        # ---- Concatenate MaSIF features with seq10 ----
        phla_feat = torch.cat([phla_masif_feat, phla_seq10], dim=-1)  # [B, Pp_max, G+10]
        tcr_feat = torch.cat([tcr_masif_feat, tcr_seq10], dim=-1)     # [B, Pt,     G+10]

        phla_patch_mask = phla_masif["mask"].to(device)  # [B, Pp_max]
        tcr_patch_mask = tcr_masif["mask"].to(device)    # [B, Pt]

        return phla_feat, tcr_feat, phla_patch_mask, tcr_patch_mask


    # ---------- v3 structure-only MaSIF embeddings without sequence or attention ----------
    def _extract_surf_embeddings(
        self,
        phla_masifs: List[Dict[str, torch.Tensor]],
        tcr_masifs: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract five structure-only MaSIF feature streams without attention or pooling.

        This resembles _separate_feature_one, except that it:
        - does not call _extract_seq_embeddings or generate seq10;
        - performs no patch-level attention pooling;
        - performs no type-level fusion attention; and
        - directly returns raw MaSIF features [B, P, G] and masks [B, P].

        Args:
            phla_masifs: List[5] of dictionaries containing input_feat,
                rho_coords, theta_coords, and mask.
            tcr_masifs: List[5] of dict

        Returns:
            phla_masif_feats: List[5] of [B, Pp, G]
            tcr_masif_feats: List[5] of [B, Pt, G]
            phla_patch_masks: List[5] of [B, Pp]
            tcr_patch_masks: List[5] of [B, Pt]
        """
        device = next(self.parameters()).device
        NumT = self.num_feature_type

        phla_masif_feats = []
        tcr_masif_feats = []
        phla_patch_masks = []
        tcr_patch_masks = []

        for i in range(NumT):
            phla_params = self.masif_phla_params[i]
            tcr_params = self.masif_tcr_params[i]

            # ---- pHLA MaSIF ----
            phla_masif_feat = self._masif_batch_wrapper(
                core=self.core_phla_list[i],
                params=phla_params,
                input_feat=phla_masifs[i]["input_feat"].to(device),
                rho_coords=phla_masifs[i]["rho_coords"].to(device),
                theta_coords=phla_masifs[i]["theta_coords"].to(device),
                patch_mask=phla_masifs[i]["mask"].to(device),
                mean_gauss_activation=True,
            )  # [B, Pp, G]

            # ---- TCR MaSIF ----
            tcr_masif_feat = self._masif_batch_wrapper(
                core=self.core_tcr_list[i],
                params=tcr_params,
                input_feat=tcr_masifs[i]["input_feat"].to(device),
                rho_coords=tcr_masifs[i]["rho_coords"].to(device),
                theta_coords=tcr_masifs[i]["theta_coords"].to(device),
                patch_mask=tcr_masifs[i]["mask"].to(device),
                mean_gauss_activation=True,
            )  # [B, Pt, G]

            phla_masif_feats.append(phla_masif_feat)
            tcr_masif_feats.append(tcr_masif_feat)
            phla_patch_masks.append(phla_masifs[i]["mask"].to(device))
            tcr_patch_masks.append(tcr_masifs[i]["mask"].to(device))

        return phla_masif_feats, tcr_masif_feats, phla_patch_masks, tcr_patch_masks

    # === Model-parameter summary ===
    def check_logging(self, depth: int = 3, trainable_only: bool = False) -> str:
        """
        Return a text report containing:
        - total, trainable, and frozen parameter counts; and
        - a recursive module hierarchy with parameter counts up to ``depth``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        size_mb = total * 4 / 1e6

        lines = []
        lines.append("=== Model Summary ===")
        lines.append(f"Total params: {total:,}")
        lines.append(f"Trainable:    {trainable:,}")
        lines.append(f"Frozen:       {frozen:,}")
        lines.append(f"~Model size (FP32): {size_mb:.2f} MB")
        lines.append("")
        lines.append("=== Architecture (truncated) ===")
        lines.extend(summarize_model(self, name=self.__class__.__name__, depth=depth, trainable_only=trainable_only))
        report = "\n".join(lines)
        return report


    # ============================ Forward: three modes with batched inputs ============================
    def forward(
        self,
        # ---- Batched sequence inputs; may be None in surf_only mode ----
        peptide_emb: Optional[torch.Tensor] = None,   # [B, 15, 1152]
        hla_emb: Optional[torch.Tensor] = None,       # [B, 276, 1152]
        tcra_emb: Optional[torch.Tensor] = None,      # [B, 127, 1152]
        tcrb_emb: Optional[torch.Tensor] = None,      # [B, 130, 1152]

        # ---- Batched masks; True is valid, False is padding, and surf_only may use None ----
        peptide_mask: Optional[torch.Tensor] = None,  # [B, 15]
        hla_mask: Optional[torch.Tensor] = None,      # [B, 276]
        tcra_mask: Optional[torch.Tensor] = None,     # [B, 127]
        tcrb_mask: Optional[torch.Tensor] = None,     # [B, 130]

        # ---- Multi-stream MaSIF inputs; may be None in seq_only mode ----
        phla_masifs: Optional[List[Dict[str, torch.Tensor]]] = None,
        tcr_masifs:  Optional[List[Dict[str, torch.Tensor]]] = None,

        # ---- Other parameters ----
        mean_gauss_activation: bool = True,
        return_attn: bool = True,
        mode: str = "joint",  # "seq_only" | "joint" | "surf_only"
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | Tuple[slice, slice]]:
        """
        Support three modes with batched inputs:

        1. seq_only: Stage 1 sequence-only training.
           - Input: sequence embeddings and masks.
           - Output: {"logit": [B, 1], "h_p": [B, hid], "h_t": [B, hid]}.

        2. joint: Full sequence-plus-structure mode.
           - Input: sequences, MaSIF structures, and masks.
           - Output: {"logit": [B, 1], "type_vecs": [B, NumT, D], ...}.

        3. surf_only: Structure-only mode added in v3.
           - Input: MaSIF structures and masks; sequence inputs are ignored.
           - Output: {"logit": [B, 1]}.
        """
        device = next(self.parameters()).device

        # ==================== Mode 1: seq_only ====================
        if mode == "seq_only":
            # Extract batched sequence embeddings
            phla_seq_emb, tcr_seq_emb = self._extract_seq_embeddings(
                peptide_emb, hla_emb, tcra_emb, tcrb_emb,
                peptide_mask, hla_mask, tcra_mask, tcrb_mask
            )  # [B, hid], [B, hid]

            # Fuse pHLA and TCR sequence representations with attention
            seq_tokens = torch.stack([phla_seq_emb, tcr_seq_emb], dim=1)  # [B, 2, hid]
            attn_out, _ = self.seq_attn(seq_tokens, seq_tokens, seq_tokens)  # [B, 2, hid]
            attn_out = self.seq_attn_norm(seq_tokens + attn_out)  # residual + norm

            # Aggregate into one representation
            combined = attn_out.mean(dim=1)  # [B, hid]

            # Classify
            logit = self.seq_cls_head(combined)  # [B, 1]

            return {"logit": logit, "h_p": phla_seq_emb, "h_t": tcr_seq_emb}


        # ==================== Mode 2: joint sequence-plus-structure mode ====================
        elif mode == "joint":
            assert phla_masifs is not None and tcr_masifs is not None, \
                "joint mode requires phla_masifs and tcr_masifs"

            NumT = self.num_feature_type
            assert len(phla_masifs) == NumT and len(tcr_masifs) == NumT, \
                "The input MaSIF list lengths must equal num_feature_type"

            B = peptide_emb.size(0)
            type_vecs: List[torch.Tensor] = []
            pair_attn_maps_all_types: List[List[torch.Tensor]] = []

            # ---- Extract global sequence embeddings once for all feature types ----
            phla_seq_global, tcr_seq_global = self._extract_seq_embeddings(
                peptide_emb, hla_emb, tcra_emb, tcrb_emb,
                peptide_mask, hla_mask, tcra_mask, tcrb_mask
            )  # [B, hid], [B, hid]

            # Process each feature stream
            for i in range(NumT):
                # _separate_feature_one supports batched inputs and masks
                phla_feat, tcr_feat, phla_patch_mask, tcr_patch_mask = self._separate_feature_one(
                    idx=i,
                    peptide_emb=peptide_emb, hla_emb=hla_emb,
                    tcra_emb=tcra_emb, tcrb_emb=tcrb_emb,
                    peptide_mask=peptide_mask, hla_mask=hla_mask,
                    tcra_mask=tcra_mask, tcrb_mask=tcrb_mask,
                    phla_masif=phla_masifs[i],
                    tcr_masif=tcr_masifs[i],
                    phla_seq_global=phla_seq_global,
                    tcr_seq_global=tcr_seq_global,
                    mean_gauss_activation=mean_gauss_activation
                )
                # phla_feat: [B, Pp_max, G+10]
                # tcr_feat: [B, Pt, G+10]

                # ---- Concatenate pHLA and TCR patches ----
                all_patches = torch.cat([phla_feat, tcr_feat], dim=1)  # [B, Pp_max+Pt, D_joint]
                all_patch_mask = torch.cat([phla_patch_mask, tcr_patch_mask], dim=1)  # [B, Pp_max+Pt]
                all_patch_key_padding_mask = ~all_patch_mask.bool()

                # ---- Joint encoder with patch mask ----
                joint_out, pair_attn_maps = self.joint_encoders[i](
                    all_patches,
                    key_padding_mask=all_patch_key_padding_mask
                )  # [B, Pp_max+Pt, D_joint]

                # Masked pooling
                final_vec_i = masked_mean(joint_out, all_patch_mask, dim=1)  # [B, D_joint]
                final_vec_i = self.out_norms[i](final_vec_i)

                type_vecs.append(final_vec_i)
                pair_attn_maps_all_types.append(pair_attn_maps)

            # Stack into a feature-type sequence with shape [B, NumT, D_joint]
            type_vecs_tensor = torch.stack(type_vecs, dim=1)

            # ---- Feature-type level encoder ----
            ft_out, feature_attn_maps = self.ft_encoder(type_vecs_tensor)  # [B, NumT, D_joint]
            ft_pooled = ft_out.mean(dim=1)  # [B, D_joint]

            # ---- Classification head ----
            logit = self.cls_head(ft_pooled)  # [B, 1]

            out: Dict[str, torch.Tensor | List[torch.Tensor]] = {
                "logit": logit,
                "type_vecs": type_vecs_tensor,
            }
            if return_attn:
                out["pair_attn_maps"] = pair_attn_maps_all_types
                out["feature_attn_maps"] = feature_attn_maps
            return out


        # ==================== Mode 3: v3 structure-only surf_only mode ====================
        elif mode == "surf_only":
            assert phla_masifs is not None and tcr_masifs is not None, \
                "surf_only mode requires phla_masifs and tcr_masifs"

            NumT = self.num_feature_type
            assert len(phla_masifs) == NumT and len(tcr_masifs) == NumT, \
                "The input MaSIF list lengths must equal num_feature_type"

            # ---- 1. Extract five MaSIF streams without sequence input or attention ----
            phla_masif_feats, tcr_masif_feats, phla_patch_masks, tcr_patch_masks = self._extract_surf_embeddings(
                phla_masifs, tcr_masifs
            )
            # phla_masif_feats: List[5] of [B, Pp, G]
            # tcr_masif_feats: List[5] of [B, Pt, G]

            type_vecs: List[torch.Tensor] = []
            pair_attn_maps_all_types: List[List[torch.Tensor]] = []

            # ---- 2. Merge pHLA and TCR within each stream, then apply self-attention ----
            for i in range(NumT):
                # Concatenate pHLA and TCR patches
                all_patches = torch.cat([phla_masif_feats[i], tcr_masif_feats[i]], dim=1)  # [B, Pp+Pt, G]
                all_patch_mask = torch.cat([phla_patch_masks[i], tcr_patch_masks[i]], dim=1)  # [B, Pp+Pt]
                all_patch_key_padding_mask = ~all_patch_mask.bool()

                # Self-attention through the surf_only encoder with dimension G
                joint_out, pair_attn_maps = self.surf_joint_encoders[i](
                    all_patches,
                    key_padding_mask=all_patch_key_padding_mask
                )  # [B, Pp+Pt, G]

                # Masked pooling
                final_vec_i = masked_mean(joint_out, all_patch_mask, dim=1)  # [B, G]
                final_vec_i = self.surf_out_norms[i](final_vec_i)

                type_vecs.append(final_vec_i)
                pair_attn_maps_all_types.append(pair_attn_maps)

            # ---- 3. Apply self-attention across the five feature types ----
            type_vecs_tensor = torch.stack(type_vecs, dim=1)  # [B, 5, G]

            ft_out, feature_attn_maps = self.surf_ft_encoder(type_vecs_tensor)  # [B, 5, G]
            ft_pooled = ft_out.mean(dim=1)  # [B, G]

            # ---- 4. Classify ----
            logit = self.surf_cls_head(ft_pooled)  # [B, 1]

            out: Dict[str, torch.Tensor | List[torch.Tensor]] = {
                "logit": logit,
                "type_vecs": type_vecs_tensor,  # [B, 5, G]
            }
            if return_attn:
                out["pair_attn_maps"] = pair_attn_maps_all_types
                out["feature_attn_maps"] = feature_attn_maps
            return out

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'seq_only', 'joint', or 'surf_only'.")
