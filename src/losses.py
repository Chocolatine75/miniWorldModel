"""Loss functions for the mini world model.

All losses operate on batches of embeddings and return scalar tensors.
Reference: VICReg (Bardes et al., 2022) for var_loss and cov_loss.
"""

import torch
import torch.nn.functional as F


def pred_loss(z_hat: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    """JEPA prediction loss: MSE between predicted and target embedding.

    z_hat:  (B, emb_dim) — predicted next embedding from GRUPredictor
    z_next: (B, emb_dim) — actual next embedding from Encoder(next_obs)

    The stop-gradient on z_next is applied here with .detach().
    Returns scalar MSE loss.
    """
    return F.mse_loss(z_hat, z_next.detach())


def var_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Variance regularization: each embedding dimension should have std >= 1.

    z: (B, emb_dim) — batch of embeddings

    For each dimension d, penalizes max(0, 1 - std(z[:, d])).
    Returns scalar loss. Zero when all dimensions have std >= 1.
    Loss > 0 means some dimensions have collapsed to a constant.
    """
    std = torch.sqrt(z.var(dim=0) + eps)             # (emb_dim,) — per-dim std
    return torch.mean(torch.clamp(1.0 - std, min=0.0))


def cov_loss(z: torch.Tensor) -> torch.Tensor:
    """Covariance regularization: decorrelate embedding dimensions.

    z: (B, emb_dim) — batch of embeddings

    Penalizes squared off-diagonal entries of the covariance matrix.
    Returns scalar loss. Zero when all dimensions are uncorrelated.
    Loss > 0 means embedding dimensions are redundant (encode the same info).
    """
    B, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)    # center the batch
    cov = (z_centered.T @ z_centered) / (B - 1)     # (D, D) covariance matrix
    # Mask out diagonal (we only penalize off-diagonal entries)
    mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    off_diag = cov[mask]                             # (D*D - D,)
    return (off_diag ** 2).sum() / D


def idm_loss(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Inverse dynamics loss: predict which action was taken from (z_t, z_{t+1}).

    logits: (B, n_actions) — raw scores from IDMHead
    action: (B,) int64    — ground truth action indices

    Returns scalar cross-entropy loss.
    A low IDM loss means the encoder captures action-relevant information.
    """
    return F.cross_entropy(logits, action)
