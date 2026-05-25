"""World model components: encoders, predictor, IDM head, and factory functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


class MLPEncoder(nn.Module):
    """Encode CartPole state observations into embeddings.

    obs: (B, obs_dim) → z: (B, emb_dim)
    Two hidden layers with ReLU activations.
    """

    def __init__(self, obs_dim: int, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → z: (B, emb_dim)"""
        assert obs.dim() == 2, f"Expected (B, obs_dim), got {obs.shape}"
        return self.net(obs)


class CNNEncoder(nn.Module):
    """Encode MiniGrid image observations into embeddings.

    obs: (B, 3, H, W) float32 in [0, 1] → z: (B, emb_dim)
    Two conv layers → AdaptiveAvgPool(4,4) → two linear layers.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # output: (B, 64, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, H, W) → z: (B, emb_dim)"""
        assert obs.dim() == 4, f"Expected (B, 3, H, W), got {obs.shape}"
        features = self.conv(obs)        # (B, 64, 4, 4)
        features = features.flatten(1)   # (B, 1024)
        return self.fc(features)         # (B, emb_dim)


class GRUPredictor(nn.Module):
    """Predict the next embedding given current embedding and action.

    Inputs: z (B, emb_dim), a_onehot (B, n_actions)
    Output: z_next (B, emb_dim)
    """

    def __init__(self, emb_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.input_proj  = nn.Linear(emb_dim + n_actions, hidden_dim)
        self.gru         = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, z: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        """z: (B, emb_dim), a_onehot: (B, n_actions) → z_next: (B, emb_dim)"""
        assert z.dim() == 2,        f"Expected (B, emb_dim), got {z.shape}"
        assert a_onehot.dim() == 2, f"Expected (B, n_actions), got {a_onehot.shape}"
        x = torch.cat([z, a_onehot], dim=1)    # (B, emb_dim + n_actions)
        x = self.input_proj(x).unsqueeze(1)    # (B, 1, hidden_dim)
        h, _ = self.gru(x)                     # (B, 1, hidden_dim)
        return self.output_proj(h.squeeze(1))  # (B, emb_dim)


class IDMHead(nn.Module):
    """Inverse Dynamics Model: predict action from (z_t, z_{t+1}).

    Inputs: z_t (B, emb_dim), z_next (B, emb_dim)
    Output: logits (B, n_actions) — raw scores, apply softmax for probabilities
    Used ONLY during training, not at planning time.
    """

    def __init__(self, emb_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """z_t: (B, emb_dim), z_next: (B, emb_dim) → logits: (B, n_actions)"""
        assert z_t.shape == z_next.shape, f"Shape mismatch: {z_t.shape} vs {z_next.shape}"
        x = torch.cat([z_t, z_next], dim=1)  # (B, 2*emb_dim)
        return self.net(x)                    # (B, n_actions)


class WorldModel(nn.Module):
    """Container for encoder, predictor, and IDM head.

    Call forward() to get all intermediate tensors needed by the loss functions.
    """

    def __init__(self, encoder: nn.Module, predictor: GRUPredictor, idm_head: IDMHead):
        super().__init__()
        self.encoder   = encoder
        self.predictor = predictor
        self.idm_head  = idm_head

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                next_obs: torch.Tensor, n_actions: int) -> dict:
        """Full forward pass for training.

        obs:      (B, *obs_shape) — current observation
        action:   (B,) int64     — action taken
        next_obs: (B, *obs_shape) — next observation
        n_actions: int            — number of discrete actions (for one-hot)

        Returns dict:
          z_t:       (B, emb_dim) — embedding of obs
          z_t1:      (B, emb_dim) — embedding of next_obs (used as target)
          z_hat:     (B, emb_dim) — predicted next embedding
          idm_logits:(B, n_actions) — action logits from IDM head
        """
        z_t        = self.encoder(obs)                            # (B, emb_dim)
        z_t1       = self.encoder(next_obs)                       # (B, emb_dim)
        a_onehot   = F.one_hot(action, n_actions).float()        # (B, n_actions)
        z_hat      = self.predictor(z_t, a_onehot)               # (B, emb_dim)
        idm_logits = self.idm_head(z_t, z_t1)                    # (B, n_actions)
        return {"z_t": z_t, "z_t1": z_t1, "z_hat": z_hat, "idm_logits": idm_logits}


def build_encoder(cfg: SimpleNamespace) -> nn.Module:
    """Return MLPEncoder for CartPole or CNNEncoder for MiniGrid.

    cfg: SimpleNamespace with env_id, obs_dim, emb_dim, hidden_dim
    """
    if "CartPole" in cfg.env_id:
        return MLPEncoder(cfg.obs_dim, cfg.emb_dim, cfg.hidden_dim)
    return CNNEncoder(cfg.emb_dim, cfg.hidden_dim)


def build_world_model(cfg: SimpleNamespace) -> WorldModel:
    """Build a complete WorldModel from config.

    cfg: SimpleNamespace with env_id, obs_dim, emb_dim, hidden_dim, n_actions
    """
    encoder   = build_encoder(cfg)
    predictor = GRUPredictor(cfg.emb_dim, cfg.n_actions, cfg.hidden_dim)
    idm_head  = IDMHead(cfg.emb_dim, cfg.n_actions, cfg.hidden_dim)
    return WorldModel(encoder, predictor, idm_head)
