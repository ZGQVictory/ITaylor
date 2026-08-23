import math
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaSIFCore(nn.Module):
    """
    PyTorch MaSIF-Core implementation for processing five feature types.

    Version 2 changes preserve the original logic and mathematical meaning:
    1. masif_inf accepts both 3D [N, V, 1] and 4D [B, P, V, 1] inputs.
       - In the legacy form, N is the number of patches P at the Network layer,
         rather than a true batch dimension.
       - The new form introduces B explicitly, flattens [B, P] to N=B*P for
         computation, and reshapes the result back to [B, P, *].
    2. The Python loop over n_rotations can be vectorized as matrix operations
       for more efficient GPU execution.
       - Only theta requires a per-rotation offset; rho weights are reusable.
       - Angle remainder, mean_gauss_activation normalization, W_conv/b_conv,
         max pooling over rotations, and ReLU match the original implementation.
    """

    def __init__(self, max_rho: float = 12.0, n_thetas: int = 16, n_rhos: int = 5, n_rotations: int = 16):
        super().__init__()
        self.max_rho = float(max_rho)
        self.n_thetas = int(n_thetas)
        self.n_rhos = int(n_rhos)
        self.n_rotations = int(n_rotations)

    # -------- Utility functions retained from the original implementation --------
    @staticmethod
    def frobenius_norm(t: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius norm. Retained for compatibility and review.
        """
        square_tensor = t * t
        tensor_sum = torch.sum(square_tensor)
        return torch.sqrt(tensor_sum)

    @staticmethod
    def build_sparse_matrix_softmax(
        idx_non_zero_values: np.ndarray,  # NumPy indices with shape (nnz, 2)
        X: torch.Tensor,                  # Values with shape (nnz,) or (nnz, 1)
        dense_shape_A: Tuple[int, int],   # Sparse-matrix shape
        dim: int = -1,                    # Softmax dimension
    ) -> torch.Tensor:
        """
        Build a sparse tensor and apply sparse softmax with PyTorch-equivalent semantics.

        This method remains unused in the main path, as in the original
        implementation, and is retained for interface compatibility and review.
        """
        if X.dim() > 1:
            X = X.squeeze(-1)
        indices = torch.from_numpy(idx_non_zero_values.T).long()
        A = torch.sparse_coo_tensor(indices, X, dense_shape_A).coalesce()
        try:
            A = torch.sparse.softmax(A, dim=dim if dim >= 0 else 1)
        except Exception:
            dense = A.to_dense()
            dense = F.softmax(dense, dim=dim if dim >= 0 else 1)
            A = dense.to_sparse_coo()
        return A

    def compute_initial_coordinates(self) -> np.ndarray:
        """
        Return polar coordinates (rho, theta) for n_rhos * n_thetas grid points.

        The result has shape (n_gauss, 2). As in the original logic, rho omits
        zero and theta omits the right endpoint at 2*pi.
        """
        range_rho = [0.0, self.max_rho]
        range_theta = [0.0, 2.0 * math.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)
        grid_rho_ = grid_rho_.T
        grid_theta_ = grid_theta_.T

        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0).T
        return coords.astype("float32")

    def masif_inf(
        self,
        input_feat: torch.Tensor,    # [N, V, 1] or [B, P, V, 1]
        rho_coords: torch.Tensor,    # Same shape as input_feat
        theta_coords: torch.Tensor,  # Same shape as input_feat
        mask: torch.Tensor,          # [N, V, 1], [B, P, V, 1], [N, V], or [B, P, V]
        W_conv: torch.Tensor,        # [G, G]
        b_conv: torch.Tensor,        # [G]
        mu_rho: torch.Tensor,        # [1, G]
        sigma_rho: torch.Tensor,     # [1, G]
        mu_theta: torch.Tensor,      # [1, G]
        sigma_theta: torch.Tensor,   # [1, G]
        eps: float = 1e-5,
        mean_gauss_activation: bool = True,
        vectorized_rotations: bool = False,
    ) -> torch.Tensor:
        """
        PyTorch equivalent of the original TensorFlow inference semantics:
          1. Apply n_rotations angular rotations.
          2. Compute soft-grid Gaussian weights over rho and theta.
          3. Sum weighted vertices into a G-dimensional descriptor.
          4. Apply the linear operation ``@ W_conv + b_conv``.
          5. Max-pool over rotations and apply ReLU.

        Returns:
          - [N, G] for input shape [N, V, 1].
          - [B, P, G] for input shape [B, P, V, 1].
        """
        # ---- For 4D input, flatten [B, P] to N=B*P and reshape afterward ----
        reshape_back = None
        if input_feat.dim() == 4:
            B, P, V, C = input_feat.shape
            reshape_back = (B, P)
            input_feat = input_feat.reshape(B * P, V, C)
            rho_coords = rho_coords.reshape(B * P, V, 1)
            theta_coords = theta_coords.reshape(B * P, V, 1)
            if mask.dim() == 4:
                mask = mask.reshape(B * P, V, mask.shape[-1])
            elif mask.dim() == 3:
                mask = mask.reshape(B * P, V, 1)
            elif mask.dim() == 2:
                mask = mask.reshape(B * P, V)
        elif input_feat.dim() != 3:
            raise ValueError(f"Unexpected input_feat dim={input_feat.dim()}, expected 3 or 4.")

        # Read N and V, matching the original n_samples and n_vertices
        n_samples = rho_coords.shape[0]
        n_vertices = rho_coords.shape[1]
        n_gauss = self.n_thetas * self.n_rhos

        # ---- Normalize the mask to [N, V, 1] ----
        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(-1)  # [N, V, 1]
        else:
            mask_expanded = mask  # [N, V, 1] or higher rank

        # ---- Flatten the vertex dimension to [N*V, 1] for Gaussian weights ----
        rho_flat = rho_coords.reshape(-1, 1)
        theta_flat = theta_coords.reshape(-1, 1)

        # rho weights are independent of rotation k and can be reused
        rho_weights = torch.exp(-((rho_flat - mu_rho) ** 2) / (sigma_rho ** 2 + eps))  # [N*V, G]

        if not vectorized_rotations:
            # Retain the original loop as a debugging and out-of-memory fallback
            all_conv_feat = []
            for k in range(self.n_rotations):
                thetas_k = theta_flat + (k * 2.0 * math.pi / self.n_rotations)
                thetas_k = torch.remainder(thetas_k, 2.0 * math.pi)
                theta_weights = torch.exp(-((thetas_k - mu_theta) ** 2) / (sigma_theta ** 2 + eps))
                gauss_activations = rho_weights * theta_weights  # [N*V, G]
                gauss_activations = gauss_activations.view(n_samples, n_vertices, -1)  # [N, V, G]
                gauss_activations = gauss_activations * mask_expanded  # [N, V, G]

                if mean_gauss_activation:
                    denom = torch.sum(gauss_activations, dim=1, keepdim=True) + eps  # [N, 1, G]
                    gauss_activations = gauss_activations / denom

                # Sum over the vertex dimension to obtain [N, G]
                gauss_desc = (gauss_activations * input_feat).sum(dim=1)  # [N, 1, G] or [N, G]
                gauss_desc = gauss_desc.view(n_samples, n_gauss)

                conv_feat = gauss_desc @ W_conv + b_conv  # [N, G]
                all_conv_feat.append(conv_feat)

            all_conv_feat = torch.stack(all_conv_feat, dim=0)     # [K, N, G]
            conv_feat = torch.max(all_conv_feat, dim=0).values    # [N, G]
            conv_feat = F.relu(conv_feat)

        else:
            # ---- Vectorized rotations processed in parallel over k ----
            K = self.n_rotations
            # [K, 1, 1] offsets avoid repeated Python loops and kernel launches
            rot_offsets = (torch.arange(K, device=theta_flat.device, dtype=theta_flat.dtype)
                           * (2.0 * math.pi / K)).view(K, 1, 1)
            theta_k = theta_flat.view(1, -1, 1) + rot_offsets  # [K, N*V, 1]
            theta_k = torch.remainder(theta_k, 2.0 * math.pi)  # [K, N*V, 1]

            theta_weights = torch.exp(-((theta_k - mu_theta) ** 2) / (sigma_theta ** 2 + eps))  # [K, N*V, G]
            gauss_activations = theta_weights * rho_weights.view(1, -1, n_gauss)                # [K, N*V, G]
            gauss_activations = gauss_activations.view(K, n_samples, n_vertices, n_gauss)       # [K, N, V, G]

            # Broadcast mask from [1, N, V, 1] to [K, N, V, G]
            gauss_activations = gauss_activations * mask_expanded.view(1, n_samples, n_vertices, 1)

            if mean_gauss_activation:
                denom = torch.sum(gauss_activations, dim=2, keepdim=True) + eps  # [K, N, 1, G]
                gauss_activations = gauss_activations / denom

            # Broadcast input_feat from [N, V, 1] through [1, N, V, 1] to [K, N, V, G]
            gauss_desc = (gauss_activations * input_feat.view(1, n_samples, n_vertices, 1)).sum(dim=2)  # [K, N, G]

            conv_feat_k = gauss_desc @ W_conv + b_conv  # [K, N, G]
            conv_feat = torch.max(conv_feat_k, dim=0).values  # [N, G]
            conv_feat = F.relu(conv_feat)

        # ---- Reshape back to [B, P, G] for 4D input ----
        if reshape_back is not None:
            B, P = reshape_back
            conv_feat = conv_feat.view(B, P, n_gauss)
        return conv_feat
