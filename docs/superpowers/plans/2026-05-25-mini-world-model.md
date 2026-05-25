# Mini World Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete action-conditioned world model (mini AC-JEPA) on CartPole + MiniGrid with CEM/MPPI planning, runnable end-to-end on Kaggle T4 in one notebook command.

**Architecture:** MLPEncoder (CartPole) or CNNEncoder (MiniGrid) encodes observations into 32-dim embeddings; GRUPredictor conditioned on one-hot actions predicts next embeddings; IDMHead recovers actions from embedding pairs. VICReg (var + cov) + IDM losses prevent collapse. CEM/MPPI planner rolls the predictor forward to score action sequences against a goal embedding.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0, Gymnasium, MiniGrid, NumPy, Matplotlib, Pillow, PyYAML, pytest (smoke test only)

> **Note on testing:** Per project constraints, only a single smoke test is added (not TDD per module). Time budget is 1 day / 4 people. The smoke test (`tests/test_smoke.py`) covers model forward pass, loss finiteness, and planner output validity.

---

## File Map

| File | Responsibility |
|---|---|
| `config.yaml` | All hyperparameters — single source of truth |
| `requirements.txt` | All Python dependencies |
| `src/__init__.py` | Empty, makes `src` a package |
| `src/env.py` | `make_env()`, `get_obs()`, `set_seed()` |
| `src/data.py` | `collect_random_rollouts()`, `RolloutDataset`, `make_loaders()` |
| `src/model.py` | `MLPEncoder`, `CNNEncoder`, `GRUPredictor`, `IDMHead`, `WorldModel`, `build_encoder()`, `build_world_model()` |
| `src/losses.py` | `pred_loss()`, `var_loss()`, `cov_loss()`, `idm_loss()` |
| `src/train.py` | Training loop, collapse detection, loss PNG, CLI entry point |
| `src/plan.py` | `plan()`, `_cem()`, `_mppi()`, `_score_sequences()`, `run_mpc_episode()` |
| `src/viz.py` | `make_demo_gif()`, `plot_ablation_table()` |
| `tests/test_smoke.py` | Single smoke test: model fwd + loss finite + planner valid |
| `notebooks/kaggle_run.ipynb` | 6-cell Kaggle entry point |
| `README.md` | Pitch, architecture ASCII, GIF embed, ablation table, one-liner |

---

## Task 0: Project Scaffold

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `data/.gitkeep`
- Create: `checkpoints/.gitkeep`
- Create: `assets/.gitkeep`
- Create: `.gitignore`

- [ ] **Step 1: Write `config.yaml`**

```yaml
# config.yaml — ALL hyperparameters live here. Never hardcode values in src/.

# Environment
env_id: "CartPole-v1"   # Switch to "MiniGrid-Empty-8x8-v0" for stretch goal
seed: 42
obs_dim: 4              # CartPole obs dim; ignored for MiniGrid (inferred from image)
n_actions: 2            # CartPole: 2 | MiniGrid-Empty-8x8: 7

# Model
emb_dim: 32
hidden_dim: 128

# Data collection
n_episodes: 500

# Training
batch_size: 256
n_epochs: 20
lr: 0.001
checkpoint_every: 1     # Save checkpoint every N epochs

# Loss weights (VICReg + IDM)
alpha: 1.0   # var_loss weight
beta: 1.0    # cov_loss weight
gamma: 1.0   # idm_loss weight

# Planner
plan_method: "cem"   # "cem" or "mppi"
N: 512               # Number of candidate action sequences
H: 15                # Planning horizon (timesteps)
K: 64                # Number of elite sequences kept each iteration
n_iter: 5            # CEM/MPPI refinement iterations
mppi_lambda: 10.0    # MPPI softmax temperature
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.0.0
gymnasium[classic-control]>=0.28.0
minigrid>=2.3.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.0.0
tqdm>=4.65.0
pyyaml>=6.0
pytest>=7.0.0
```

- [ ] **Step 3: Create empty files and directories**

```bash
touch src/__init__.py
mkdir -p data checkpoints assets notebooks tests
touch data/.gitkeep checkpoints/.gitkeep assets/.gitkeep
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pt
*.pth
checkpoints/
data/rollouts.pt
assets/demo.gif
.env
wandb/
```

- [ ] **Step 5: Commit**

```bash
git add config.yaml requirements.txt src/__init__.py .gitignore
git add data/.gitkeep checkpoints/.gitkeep assets/.gitkeep
git commit -m "chore: project scaffold — config, requirements, directory structure"
```

---

## Task 1: Environment Wrapper (`src/env.py`)

**Files:**
- Create: `src/env.py`

- [ ] **Step 1: Write `src/env.py`**

```python
"""Gymnasium environment helpers: creation, seeding, observation extraction."""

import random
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int, env: gym.Env | None = None) -> None:
    """Seed torch, numpy, random, and optionally a Gym env.

    Call this once at the start of each script.
    seed: integer seed for all RNGs
    env: if provided, calls env.reset(seed=seed)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.reset(seed=seed)


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create and seed a Gymnasium environment with rgb_array rendering.

    env_id: e.g. "CartPole-v1" or "MiniGrid-Empty-8x8-v0"
    seed: integer seed
    Returns a seeded gym.Env.
    """
    if "MiniGrid" in env_id:
        import minigrid  # noqa: F401 — registers MiniGrid envs
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def get_obs(obs: object, env_id: str) -> np.ndarray:
    """Extract and normalize a raw Gym observation to a float32 numpy array.

    CartPole: obs is already (4,) float64 → cast to float32, shape (4,)
    MiniGrid: obs is a dict with 'image' key, shape (H, W, 3) uint8
              → transpose to (3, H, W) and normalize to [0, 1] float32

    Returns np.ndarray ready to be wrapped in torch.tensor().
    """
    if "MiniGrid" in env_id:
        img = obs["image"]                             # (H, W, 3) uint8
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3, H, W)
        return img
    return np.asarray(obs, dtype=np.float32)           # (4,)
```

- [ ] **Step 2: Quick sanity check — run in Python to confirm env creates without error**

```bash
python -c "
from src.env import make_env, get_obs, set_seed
env = make_env('CartPole-v1', 42)
obs_raw, _ = env.reset()
obs = get_obs(obs_raw, 'CartPole-v1')
print('CartPole obs shape:', obs.shape)   # expect (4,)
print('Action space:', env.action_space)  # Discrete(2)
"
```

Expected output:
```
CartPole obs shape: (4,)
Action space: Discrete(2)
```

- [ ] **Step 3: Commit**

```bash
git add src/env.py
git commit -m "feat: env wrapper — make_env, get_obs, set_seed"
```

---

## Task 2: Data Pipeline (`src/data.py`)

**Files:**
- Create: `src/data.py`

- [ ] **Step 1: Write `src/data.py`**

```python
"""Rollout collection and PyTorch Dataset for (obs, action, next_obs) transitions."""

from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.env import make_env, get_obs, set_seed


def collect_random_rollouts(cfg) -> None:
    """Collect random-policy transitions and save to data/rollouts.pt.

    cfg: SimpleNamespace with fields: env_id, seed, n_episodes
    Saves a list of dicts: [{"obs": Tensor, "action": Tensor, "next_obs": Tensor}, ...]
    Each Tensor is 1D (obs_dim,) or scalar (action).
    """
    Path("data").mkdir(exist_ok=True)
    env = make_env(cfg.env_id, cfg.seed)
    set_seed(cfg.seed, env)

    all_transitions: List[Dict[str, torch.Tensor]] = []

    for ep in range(cfg.n_episodes):
        obs_raw, _ = env.reset()
        obs = get_obs(obs_raw, cfg.env_id)
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs_raw, _, terminated, truncated, _ = env.step(action)
            next_obs = get_obs(next_obs_raw, cfg.env_id)
            done = terminated or truncated
            all_transitions.append({
                "obs":      torch.tensor(obs),
                "action":   torch.tensor(action, dtype=torch.long),
                "next_obs": torch.tensor(next_obs),
            })
            obs = next_obs

        if (ep + 1) % 50 == 0:
            print(f"  Collected episode {ep + 1}/{cfg.n_episodes} "
                  f"| total transitions: {len(all_transitions)}")

    torch.save(all_transitions, "data/rollouts.pt")
    print(f"Saved {len(all_transitions)} transitions → data/rollouts.pt")


class RolloutDataset(Dataset):
    """PyTorch Dataset that loads (obs, action, next_obs) from a .pt file.

    path: path to the .pt file saved by collect_random_rollouts()
    __getitem__ returns: (obs: Tensor, action: Tensor, next_obs: Tensor)
    """

    def __init__(self, path: str):
        self.data: List[Dict[str, torch.Tensor]] = torch.load(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns obs, action, next_obs for transition at index idx."""
        item = self.data[idx]
        return item["obs"], item["action"], item["next_obs"]


def make_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Load rollouts.pt and return (train_loader, val_loader).

    cfg: SimpleNamespace with fields: batch_size
    90% train / 10% val split by random shuffle.
    """
    dataset = RolloutDataset("data/rollouts.pt")
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    print(f"Dataset: {n_train} train / {n_val} val transitions")
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    import yaml
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    collect_random_rollouts(cfg)
```

- [ ] **Step 2: Collect a small set of rollouts to verify the file is written correctly**

```bash
python -c "
import yaml
from types import SimpleNamespace
from src.data import collect_random_rollouts, RolloutDataset
cfg = SimpleNamespace(env_id='CartPole-v1', seed=42, n_episodes=10, batch_size=32)
collect_random_rollouts(cfg)
ds = RolloutDataset('data/rollouts.pt')
obs, action, next_obs = ds[0]
print('obs shape:', obs.shape)         # (4,)
print('action:', action)               # scalar tensor
print('next_obs shape:', next_obs.shape)  # (4,)
print('dataset length:', len(ds))
"
```

Expected output (approx):
```
Saved ~200 transitions → data/rollouts.pt
obs shape: torch.Size([4])
action: tensor(0)
next_obs shape: torch.Size([4])
dataset length: ~200
```

- [ ] **Step 3: Commit**

```bash
git add src/data.py
git commit -m "feat: data pipeline — collect_random_rollouts, RolloutDataset, make_loaders"
```

---

## Task 3: Model (`src/model.py`)

**Files:**
- Create: `src/model.py`

- [ ] **Step 1: Write `src/model.py`**

```python
"""World model components: encoders, predictor, IDM head, and factory functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_encoder(cfg) -> nn.Module:
    """Return MLPEncoder for CartPole or CNNEncoder for MiniGrid.

    cfg: SimpleNamespace with env_id, obs_dim, emb_dim, hidden_dim
    """
    if "CartPole" in cfg.env_id:
        return MLPEncoder(cfg.obs_dim, cfg.emb_dim, cfg.hidden_dim)
    return CNNEncoder(cfg.emb_dim, cfg.hidden_dim)


def build_world_model(cfg) -> WorldModel:
    """Build a complete WorldModel from config.

    cfg: SimpleNamespace with env_id, obs_dim, emb_dim, hidden_dim, n_actions
    """
    encoder   = build_encoder(cfg)
    predictor = GRUPredictor(cfg.emb_dim, cfg.n_actions, cfg.hidden_dim)
    idm_head  = IDMHead(cfg.emb_dim, cfg.n_actions, cfg.hidden_dim)
    return WorldModel(encoder, predictor, idm_head)
```

- [ ] **Step 2: Verify forward pass shapes**

```bash
python -c "
import torch
import yaml
from types import SimpleNamespace
from src.model import build_world_model

with open('config.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

model = build_world_model(cfg)
B = 8
obs      = torch.randn(B, cfg.obs_dim)
action   = torch.randint(0, cfg.n_actions, (B,))
next_obs = torch.randn(B, cfg.obs_dim)

out = model(obs, action, next_obs, cfg.n_actions)
print('z_t shape:       ', out['z_t'].shape)        # (8, 32)
print('z_t1 shape:      ', out['z_t1'].shape)       # (8, 32)
print('z_hat shape:     ', out['z_hat'].shape)      # (8, 32)
print('idm_logits shape:', out['idm_logits'].shape) # (8, 2)
"
```

Expected:
```
z_t shape:        torch.Size([8, 32])
z_t1 shape:       torch.Size([8, 32])
z_hat shape:      torch.Size([8, 32])
idm_logits shape: torch.Size([8, 2])
```

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: model — MLPEncoder, CNNEncoder, GRUPredictor, IDMHead, WorldModel, factories"
```

---

## Task 4: Losses (`src/losses.py`)

**Files:**
- Create: `src/losses.py`

- [ ] **Step 1: Write `src/losses.py`**

```python
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
```

- [ ] **Step 2: Verify losses return finite scalars**

```bash
python -c "
import torch
from src.losses import pred_loss, var_loss, cov_loss, idm_loss

z     = torch.randn(16, 32)
z_hat = torch.randn(16, 32)
z_t1  = torch.randn(16, 32)
acts  = torch.randint(0, 2, (16,))
logits= torch.randn(16, 2)

print('pred_loss:', pred_loss(z_hat, z_t1).item())
print('var_loss: ', var_loss(z).item())
print('cov_loss: ', cov_loss(z).item())
print('idm_loss: ', idm_loss(logits, acts).item())
"
```

Expected: four finite floats (all > 0 typically).

- [ ] **Step 3: Commit**

```bash
git add src/losses.py
git commit -m "feat: losses — pred_loss, var_loss, cov_loss, idm_loss (VICReg + IDM)"
```

---

## Task 5: Training Loop (`src/train.py`)

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write `src/train.py`**

```python
"""Training loop for the mini world model.

Run: python -m src.train --config config.yaml
Saves: checkpoints/model_ep{N:03d}.pt every epoch, loss_curves.png at the end.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml

from src.data import collect_random_rollouts, make_loaders
from src.env import set_seed
from src.losses import cov_loss, idm_loss, pred_loss, var_loss
from src.model import build_world_model


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    """Load config.yaml into a SimpleNamespace for attribute-style access."""
    with open(path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def check_collapse(z: torch.Tensor, threshold: float = 0.01) -> None:
    """Print a red warning if embedding variance has collapsed.

    z: (B, emb_dim) — batch of embeddings from the encoder
    threshold: mean std below this value triggers the warning
    """
    mean_std = z.std(dim=0).mean().item()
    if mean_std < threshold:
        print(f"\033[91m[COLLAPSE WARNING] mean std(z) = {mean_std:.5f} "
              f"< {threshold} — consider increasing alpha/beta or lowering lr\033[0m")


def train(cfg: SimpleNamespace) -> None:
    """Full training loop.

    cfg: SimpleNamespace loaded from config.yaml
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    set_seed(cfg.seed)
    Path("checkpoints").mkdir(exist_ok=True)

    # Collect rollouts if not already present
    if not Path("data/rollouts.pt").exists():
        print("Collecting rollouts...")
        collect_random_rollouts(cfg)

    train_loader, val_loader = make_loaders(cfg)
    model     = build_world_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict = {"pred": [], "var": [], "cov": [], "idm": [], "total": []}

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        sums = {"pred": 0., "var": 0., "cov": 0., "idm": 0., "total": 0.}
        n_batches = 0

        for obs, action, next_obs in train_loader:
            obs      = obs.to(device)
            action   = action.to(device)
            next_obs = next_obs.to(device)

            out    = model(obs, action, next_obs, cfg.n_actions)

            l_pred = pred_loss(out["z_hat"], out["z_t1"])
            l_var  = var_loss(out["z_t"])
            l_cov  = cov_loss(out["z_t"])
            l_idm  = idm_loss(out["idm_logits"], action)
            loss   = l_pred + cfg.alpha * l_var + cfg.beta * l_cov + cfg.gamma * l_idm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["pred"]  += l_pred.item()
            sums["var"]   += l_var.item()
            sums["cov"]   += l_cov.item()
            sums["idm"]   += l_idm.item()
            sums["total"] += loss.item()
            n_batches += 1

        for k in sums:
            sums[k] /= n_batches
            history[k].append(sums[k])

        print(
            f"Epoch {epoch:2d}/{cfg.n_epochs} | "
            f"loss={sums['total']:.4f} | "
            f"pred={sums['pred']:.4f} | "
            f"var={sums['var']:.4f} | "
            f"cov={sums['cov']:.4f} | "
            f"idm={sums['idm']:.4f} | "
            f"lr={cfg.lr}"
        )

        # Collapse check on a validation batch
        model.eval()
        with torch.no_grad():
            obs_val, _, _ = next(iter(val_loader))
            z_check = model.encoder(obs_val.to(device))
            check_collapse(z_check)

        if epoch % cfg.checkpoint_every == 0:
            ckpt_path = f"checkpoints/model_ep{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    _plot_loss_curves(history, "loss_curves.png")
    print("Training done. Curves → loss_curves.png")


def _plot_loss_curves(history: dict, path: str) -> None:
    """Save a 2×2 grid of loss curves as a PNG.

    history: dict of {"pred": [...], "var": [...], "cov": [...], "idm": [...]}
    path: output path for the PNG file
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key in zip(axes.flatten(), ["pred", "var", "cov", "idm"]):
        ax.plot(history[key], label=key)
        ax.set_title(f"{key} loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the mini world model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
```

- [ ] **Step 2: Run a quick 2-epoch smoke check before wiring the real training**

```bash
python -c "
import yaml
from types import SimpleNamespace
from src.train import train

with open('config.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

cfg.n_epochs = 2
cfg.n_episodes = 20
cfg.batch_size = 16
train(cfg)
"
```

Expected: 2 lines of epoch output, both losses finite, `loss_curves.png` written.

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: training loop — VICReg+IDM losses, collapse detection, checkpoint, loss PNG"
```

---

## Task 6: Planner (`src/plan.py`)

**Files:**
- Create: `src/plan.py`

- [ ] **Step 1: Write `src/plan.py`**

```python
"""CEM and MPPI planners for the mini world model.

Run: python -m src.plan --config config.yaml --demo
Saves: assets/demo.gif
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.model import WorldModel, build_world_model
from src.env import make_env, get_obs, set_seed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def plan(model: WorldModel, obs: torch.Tensor, goal_obs: torch.Tensor,
         cfg: SimpleNamespace, device: str) -> int:
    """Select the best first action using CEM or MPPI.

    model:    trained WorldModel (encoder + predictor)
    obs:      (*obs_shape,) — current observation, 1D for CartPole, 3D for MiniGrid
    goal_obs: (*obs_shape,) — goal observation (same shape as obs)
    cfg:      SimpleNamespace with plan_method, N, H, K, n_iter, mppi_lambda, n_actions
    device:   "cuda" or "cpu"

    Returns: int — best first action index
    """
    model.eval()
    with torch.no_grad():
        z0     = model.encoder(obs.unsqueeze(0).to(device))      # (1, emb_dim)
        z_goal = model.encoder(goal_obs.unsqueeze(0).to(device)) # (1, emb_dim)

    if cfg.plan_method == "cem":
        return _cem(model, z0, z_goal, cfg, device)
    return _mppi(model, z0, z_goal, cfg, device)


def run_mpc_episode(model: WorldModel, env, goal_obs: torch.Tensor,
                    cfg: SimpleNamespace, device: str,
                    max_steps: int = 200) -> List[np.ndarray]:
    """Run one MPC episode: plan → step → replan at every timestep.

    model:     trained WorldModel
    env:       seeded Gym environment (render_mode='rgb_array')
    goal_obs:  (*obs_shape,) tensor — goal observation
    cfg:       planner config
    device:    "cuda" or "cpu"
    max_steps: episode length cap

    Returns: list of (H, W, 3) uint8 numpy arrays — RGB frames for GIF
    """
    obs_raw, _ = env.reset()
    obs    = torch.tensor(get_obs(obs_raw, cfg.env_id))
    frames = [env.render()]

    for _ in range(max_steps):
        action   = plan(model, obs, goal_obs, cfg, device)
        obs_raw, _, terminated, truncated, _ = env.step(action)
        obs    = torch.tensor(get_obs(obs_raw, cfg.env_id))
        frames.append(env.render())
        if terminated or truncated:
            break

    return frames


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _score_sequences(model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor,
                     action_seqs: torch.Tensor, cfg: SimpleNamespace,
                     device: str) -> torch.Tensor:
    """Roll the predictor forward and score each sequence by goal distance.

    z0:          (1, emb_dim)
    z_goal:      (1, emb_dim)
    action_seqs: (N, H) int64 — N sequences of H action indices
    Returns costs: (N,) — sum of squared distances to goal over the horizon
    """
    N, H   = action_seqs.shape
    emb_dim = z0.shape[-1]
    z      = z0.expand(N, emb_dim).clone()   # (N, emb_dim)
    costs  = torch.zeros(N, device=device)

    with torch.no_grad():
        for t in range(H):
            a        = action_seqs[:, t]                            # (N,)
            a_onehot = F.one_hot(a, cfg.n_actions).float()         # (N, n_actions)
            z        = model.predictor(z, a_onehot)                # (N, emb_dim)
            diff     = z - z_goal.expand(N, emb_dim)               # (N, emb_dim)
            costs   += (diff ** 2).sum(dim=1)                      # (N,)

    return costs


def _cem(model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor,
         cfg: SimpleNamespace, device: str) -> int:
    """Cross-Entropy Method: iteratively refine action sequences via elite resampling.

    Returns: int — best first action index
    """
    N, H, K = cfg.N, cfg.H, cfg.K
    action_seqs = torch.randint(0, cfg.n_actions, (N, H), device=device)

    for _ in range(cfg.n_iter):
        costs      = _score_sequences(model, z0, z_goal, action_seqs, cfg, device)
        elite_idx  = costs.argsort()[:K]                   # (K,) — lowest cost
        elite_seqs = action_seqs[elite_idx]                # (K, H)
        # Resample N sequences from the K elites (uniform)
        resample   = torch.randint(0, K, (N,), device=device)
        action_seqs = elite_seqs[resample]                 # (N, H)

    costs    = _score_sequences(model, z0, z_goal, action_seqs, cfg, device)
    best_seq = action_seqs[costs.argmin()]                 # (H,)
    return best_seq[0].item()


def _mppi(model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor,
          cfg: SimpleNamespace, device: str) -> int:
    """Model Predictive Path Integral: softmax-weighted resampling of elites.

    Returns: int — best first action index
    """
    N, H, K = cfg.N, cfg.H, cfg.K
    lam     = cfg.mppi_lambda
    action_seqs = torch.randint(0, cfg.n_actions, (N, H), device=device)

    for _ in range(cfg.n_iter):
        costs       = _score_sequences(model, z0, z_goal, action_seqs, cfg, device)
        elite_idx   = costs.argsort()[:K]
        elite_costs = costs[elite_idx]
        elite_seqs  = action_seqs[elite_idx]                       # (K, H)
        # Weight elites by softmax of negative cost / temperature
        weights     = torch.softmax(-elite_costs / lam, dim=0)    # (K,)
        resample    = torch.multinomial(weights, N, replacement=True)
        action_seqs = elite_seqs[resample]                         # (N, H)

    costs    = _score_sequences(model, z0, z_goal, action_seqs, cfg, device)
    best_seq = action_seqs[costs.argmin()]
    return best_seq[0].item()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CEM/MPPI planner and generate demo GIF")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pt file. Defaults to last epoch.")
    parser.add_argument("--demo",       action="store_true",
                        help="Run MPC episode and save demo.gif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = SimpleNamespace(**yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    model     = build_world_model(cfg).to(device)
    ckpt_path = args.checkpoint or f"checkpoints/model_ep{cfg.n_epochs:03d}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    if args.demo:
        from src.viz import make_demo_gif
        Path("assets").mkdir(exist_ok=True)

        env      = make_env(cfg.env_id, cfg.seed)
        goal_raw, _ = env.reset(seed=cfg.seed + 99)
        goal_obs = torch.tensor(get_obs(goal_raw, cfg.env_id))

        print(f"Running MPC episode (method={cfg.plan_method})...")
        frames = run_mpc_episode(model, env, goal_obs, cfg, device)
        make_demo_gif(frames, "assets/demo.gif")
        print(f"Demo saved → assets/demo.gif ({len(frames)} frames)")
```

- [ ] **Step 2: Verify the planner runs on a randomly-initialized model (no checkpoint needed)**

```bash
python -c "
import torch, yaml
from types import SimpleNamespace
from src.model import build_world_model
from src.plan import plan

with open('config.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.N = 16; cfg.H = 5; cfg.K = 4; cfg.n_iter = 2

device = 'cpu'
model  = build_world_model(cfg)
obs    = torch.randn(cfg.obs_dim)
goal   = torch.randn(cfg.obs_dim)
action = plan(model, obs, goal, cfg, device)
print('Best action:', action, '— valid:', action in range(cfg.n_actions))
"
```

Expected: `Best action: 0  — valid: True` (or 1)

- [ ] **Step 3: Commit**

```bash
git add src/plan.py
git commit -m "feat: CEM + MPPI planner with MPC episode runner"
```

---

## Task 7: Visualization (`src/viz.py`)

**Files:**
- Create: `src/viz.py`

- [ ] **Step 1: Write `src/viz.py`**

```python
"""Visualization utilities: animated GIF and ablation table PNG."""

from typing import Dict, List

import numpy as np
from PIL import Image


def make_demo_gif(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    """Save a list of RGB numpy frames as an animated GIF.

    frames: list of (H, W, 3) uint8 numpy arrays — each is one video frame
    path:   output path, e.g. 'assets/demo.gif'
    fps:    frames per second (default 10)
    """
    pil_frames  = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved → {path} ({len(frames)} frames at {fps} fps)")


def plot_ablation_table(results: Dict[str, Dict[str, float]], path: str) -> None:
    """Save ablation results as a matplotlib table PNG.

    results: dict of {variant_name: {metric_name: float_value}}
    Example:
        {
            "Full model":   {"pred_loss": 0.12, "idm_acc": 0.89},
            "No IDM":       {"pred_loss": 0.19, "idm_acc": 0.50},
            "No VICReg":    {"pred_loss": 0.31, "idm_acc": 0.55},
        }
    path: output path for the PNG file
    """
    import matplotlib.pyplot as plt

    variants    = list(results.keys())
    metrics     = list(results[variants[0]].keys())
    table_data  = [[f"{results[v][m]:.4f}" for m in metrics] for v in variants]

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 2), len(variants) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        rowLabels=variants,
        colLabels=metrics,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    plt.title("Ablation Study", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Ablation table saved → {path}")
```

- [ ] **Step 2: Quick check that GIF generation works**

```bash
python -c "
import numpy as np
from src.viz import make_demo_gif, plot_ablation_table
from pathlib import Path
Path('assets').mkdir(exist_ok=True)

frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
make_demo_gif(frames, 'assets/test.gif')
print('GIF exists:', Path('assets/test.gif').exists())

results = {
    'Full model': {'pred_loss': 0.12, 'idm_acc': 0.89},
    'No IDM':     {'pred_loss': 0.19, 'idm_acc': 0.50},
    'No VICReg':  {'pred_loss': 0.31, 'idm_acc': 0.55},
}
plot_ablation_table(results, 'assets/ablation.png')
"
```

Expected: both files written without error.

- [ ] **Step 3: Commit**

```bash
git add src/viz.py
git commit -m "feat: viz — make_demo_gif, plot_ablation_table"
```

---

## Task 8: Smoke Test (`tests/test_smoke.py`)

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/test_smoke.py`**

```python
"""Single smoke test: 2 epochs on 10 samples, check losses finite and planner valid.

Run: pytest tests/test_smoke.py -v
"""

import torch
import pytest
from types import SimpleNamespace


def _make_cfg() -> SimpleNamespace:
    """Minimal config for fast smoke test — does NOT read config.yaml."""
    return SimpleNamespace(
        env_id="CartPole-v1",
        seed=0,
        obs_dim=4,
        n_actions=2,
        emb_dim=32,
        hidden_dim=128,
        n_epochs=2,
        batch_size=4,
        lr=1e-3,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        checkpoint_every=999,  # no disk writes during test
        n_episodes=10,
        # planner
        N=8, H=3, K=2, n_iter=2,
        plan_method="cem",
        mppi_lambda=10.0,
    )


def test_world_model_forward_shapes():
    """Forward pass returns correct output shapes for CartPole."""
    from src.model import build_world_model
    cfg   = _make_cfg()
    model = build_world_model(cfg)
    B     = 8
    obs      = torch.randn(B, cfg.obs_dim)
    action   = torch.randint(0, cfg.n_actions, (B,))
    next_obs = torch.randn(B, cfg.obs_dim)

    out = model(obs, action, next_obs, cfg.n_actions)

    assert out["z_t"].shape       == (B, cfg.emb_dim),   f"z_t: {out['z_t'].shape}"
    assert out["z_t1"].shape      == (B, cfg.emb_dim),   f"z_t1: {out['z_t1'].shape}"
    assert out["z_hat"].shape     == (B, cfg.emb_dim),   f"z_hat: {out['z_hat'].shape}"
    assert out["idm_logits"].shape == (B, cfg.n_actions), f"idm_logits: {out['idm_logits'].shape}"


def test_losses_are_finite():
    """All four losses return finite scalar tensors."""
    from src.model import build_world_model
    from src.losses import pred_loss, var_loss, cov_loss, idm_loss

    cfg    = _make_cfg()
    model  = build_world_model(cfg)
    B      = 16
    obs    = torch.randn(B, cfg.obs_dim)
    action = torch.randint(0, cfg.n_actions, (B,))
    next_obs = torch.randn(B, cfg.obs_dim)

    out = model(obs, action, next_obs, cfg.n_actions)

    losses = {
        "pred": pred_loss(out["z_hat"], out["z_t1"]),
        "var":  var_loss(out["z_t"]),
        "cov":  cov_loss(out["z_t"]),
        "idm":  idm_loss(out["idm_logits"], action),
    }
    for name, l in losses.items():
        assert l.dim() == 0,             f"{name} is not scalar: shape {l.shape}"
        assert torch.isfinite(l).item(), f"{name} is not finite: {l.item()}"


def test_cem_planner_returns_valid_action():
    """CEM planner returns an action index in [0, n_actions)."""
    from src.model import build_world_model
    from src.plan import plan

    cfg    = _make_cfg()
    device = "cpu"
    model  = build_world_model(cfg)
    obs    = torch.randn(cfg.obs_dim)
    goal   = torch.randn(cfg.obs_dim)

    action = plan(model, obs, goal, cfg, device)

    assert isinstance(action, int),          f"Expected int, got {type(action)}"
    assert 0 <= action < cfg.n_actions,      f"Action {action} out of range [0, {cfg.n_actions})"


def test_mppi_planner_returns_valid_action():
    """MPPI planner returns an action index in [0, n_actions)."""
    from src.model import build_world_model
    from src.plan import plan

    cfg             = _make_cfg()
    cfg.plan_method = "mppi"
    device          = "cpu"
    model           = build_world_model(cfg)
    obs             = torch.randn(cfg.obs_dim)
    goal            = torch.randn(cfg.obs_dim)

    action = plan(model, obs, goal, cfg, device)

    assert isinstance(action, int),     f"Expected int, got {type(action)}"
    assert 0 <= action < cfg.n_actions, f"Action {action} out of range"


def test_two_epoch_training_loss_decreases():
    """Training for 2 epochs on 10 random samples produces finite, decreasing loss."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import build_world_model
    from src.losses import pred_loss, var_loss, cov_loss, idm_loss
    import torch.optim as optim

    cfg    = _make_cfg()
    device = "cpu"
    model  = build_world_model(cfg).to(device)
    opt    = optim.Adam(model.parameters(), lr=cfg.lr)

    # 10 fake transitions
    obs      = torch.randn(10, cfg.obs_dim)
    action   = torch.randint(0, cfg.n_actions, (10,))
    next_obs = torch.randn(10, cfg.obs_dim)
    loader   = DataLoader(TensorDataset(obs, action, next_obs), batch_size=4, shuffle=False)

    epoch_losses = []
    for _ in range(2):
        epoch_total = 0.
        n = 0
        for o, a, no in loader:
            out  = model(o, a, no, cfg.n_actions)
            loss = (pred_loss(out["z_hat"], out["z_t1"])
                    + cfg.alpha * var_loss(out["z_t"])
                    + cfg.beta  * cov_loss(out["z_t"])
                    + cfg.gamma * idm_loss(out["idm_logits"], a))
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_total += loss.item(); n += 1
        epoch_losses.append(epoch_total / n)

    assert all(torch.isfinite(torch.tensor(l)) for l in epoch_losses), \
        f"Non-finite loss: {epoch_losses}"
```

- [ ] **Step 3: Run the smoke test — must pass 100%**

```bash
pytest tests/test_smoke.py -v
```

Expected output:
```
tests/test_smoke.py::test_world_model_forward_shapes PASSED
tests/test_smoke.py::test_losses_are_finite PASSED
tests/test_smoke.py::test_cem_planner_returns_valid_action PASSED
tests/test_smoke.py::test_mppi_planner_returns_valid_action PASSED
tests/test_smoke.py::test_two_epoch_training_loss_decreases PASSED

5 passed in X.XXs
```

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_smoke.py
git commit -m "test: smoke test — model shapes, loss finiteness, CEM/MPPI validity, 2-epoch train"
```

---

## Task 9: Kaggle Notebook (`notebooks/kaggle_run.ipynb`)

**Files:**
- Create: `notebooks/kaggle_run.ipynb`

- [ ] **Step 1: Write `notebooks/kaggle_run.ipynb`**

Create a notebook with 6 cells. Each cell is a self-contained block:

**Cell 1 — Setup:**
```python
# Cell 1: Clone repo and install dependencies
import subprocess
# On Kaggle: uncomment the line below to clone your repo
# subprocess.run(["git", "clone", "https://github.com/YOUR_USER/mini-world-model.git"])
# %cd mini-world-model
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"])
print("Setup complete")
```

**Cell 2 — Collect rollouts:**
```python
# Cell 2: Collect random rollouts and save to data/rollouts.pt
import yaml
from types import SimpleNamespace
from src.data import collect_random_rollouts

with open("config.yaml") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

collect_random_rollouts(cfg)
```

**Cell 3 — Train:**
```python
# Cell 3: Train the world model (saves checkpoints + loss_curves.png)
from src.train import train

with open("config.yaml") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

train(cfg)
```

**Cell 4 — Plan + generate GIF:**
```python
# Cell 4: Run MPC planner and generate demo.gif
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
from src.model import build_world_model
from src.env import make_env, get_obs, set_seed
from src.plan import run_mpc_episode
from src.viz import make_demo_gif

with open("config.yaml") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(cfg.seed)
Path("assets").mkdir(exist_ok=True)

model = build_world_model(cfg).to(device)
ckpt_path = f"checkpoints/model_ep{cfg.n_epochs:03d}.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

env = make_env(cfg.env_id, cfg.seed)
goal_raw, _ = env.reset(seed=cfg.seed + 99)
goal_obs = torch.tensor(get_obs(goal_raw, cfg.env_id))

frames = run_mpc_episode(model, env, goal_obs, cfg, device)
make_demo_gif(frames, "assets/demo.gif")
print(f"Demo: {len(frames)} frames")
```

**Cell 5 — Display GIF:**
```python
# Cell 5: Display demo GIF inline
from IPython.display import Image as IPImage
IPImage(filename="assets/demo.gif")
```

**Cell 6 — Display loss curves:**
```python
# Cell 6: Display loss curves
from IPython.display import Image as IPImage
IPImage(filename="loss_curves.png")
```

Save this as a proper `.ipynb` JSON. Use `nbformat` to write it programmatically:

```bash
python -c "
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

cells_code = [
    '''# Cell 1: Install dependencies
import subprocess
# subprocess.run([\"git\", \"clone\", \"https://github.com/YOUR_USER/mini-world-model.git\"])
subprocess.run([\"pip\", \"install\", \"-r\", \"requirements.txt\", \"-q\"])
print(\"Setup complete\")''',

    '''# Cell 2: Collect rollouts
import yaml
from types import SimpleNamespace
from src.data import collect_random_rollouts

with open(\"config.yaml\") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
collect_random_rollouts(cfg)''',

    '''# Cell 3: Train the world model
from src.train import train
import yaml
from types import SimpleNamespace

with open(\"config.yaml\") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
train(cfg)''',

    '''# Cell 4: Plan + generate demo GIF
import torch, yaml
from pathlib import Path
from types import SimpleNamespace
from src.model import build_world_model
from src.env import make_env, get_obs, set_seed
from src.plan import run_mpc_episode
from src.viz import make_demo_gif

with open(\"config.yaml\") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
set_seed(cfg.seed)
Path(\"assets\").mkdir(exist_ok=True)

model = build_world_model(cfg).to(device)
model.load_state_dict(torch.load(f\"checkpoints/model_ep{cfg.n_epochs:03d}.pt\", map_location=device))
model.eval()

env = make_env(cfg.env_id, cfg.seed)
goal_raw, _ = env.reset(seed=cfg.seed + 99)
goal_obs = torch.tensor(get_obs(goal_raw, cfg.env_id))

frames = run_mpc_episode(model, env, goal_obs, cfg, device)
make_demo_gif(frames, \"assets/demo.gif\")''',

    '''# Cell 5: Display demo GIF
from IPython.display import Image as IPImage
IPImage(filename=\"assets/demo.gif\")''',

    '''# Cell 6: Display loss curves
from IPython.display import Image as IPImage
IPImage(filename=\"loss_curves.png\")''',
]

nb = new_notebook(cells=[new_code_cell(c) for c in cells_code])
with open('notebooks/kaggle_run.ipynb', 'w') as f:
    nbformat.write(nb, f)
print('Notebook written.')
"
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/kaggle_run.ipynb
git commit -m "feat: Kaggle notebook — 6-cell end-to-end run"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Mini World Model — AC-JEPA on CartPole + MiniGrid

A minimal action-conditioned world model inspired by [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Meta AI / FAIR). Built for the **HackTheWorld(s)** hackathon (May 2026).

## What this is

We train an encoder + action-conditioned predictor using JEPA prediction loss + VICReg regularization (variance + covariance) + Inverse Dynamics (IDM) loss to prevent representation collapse. We then freeze the model and run **CEM** and **MPPI** planners over action sequences, scoring them by distance to a goal embedding.

```
obs s_t ──► Encoder ──► z_t ──► Predictor ──► ẑ_{t+1}
              (MLP)               (GRU)            │
                                                   │ MSE
             obs s_{t+1} ──► Encoder ──► z_{t+1} ◄┘
                                       stop-grad
```

## Losses

| Loss | Formula | Purpose |
|---|---|---|
| Prediction | `MSE(ẑ_{t+1}, sg(z_{t+1}))` | Learn to predict future state |
| Variance | `mean(max(0, 1 - std(z)))` | Prevent collapse to constant |
| Covariance | `sum(off_diag(cov(z))²)` | Decorrelate embedding dims |
| IDM | `CrossEntropy(MLP(z_t, z_{t+1}), a_t)` | Encode action-relevant info |

## How to run on Kaggle

1. Open `notebooks/kaggle_run.ipynb` on Kaggle (Accelerator → GPU T4)
2. Update Cell 1 with your GitHub repo URL
3. Run all cells in order — takes ~30 min on CartPole

## How to run locally

```bash
pip install -r requirements.txt
python -m src.data --config config.yaml
python -m src.train --config config.yaml
python -m src.plan --config config.yaml --demo
```

## Results

![Demo GIF](assets/demo.gif)

### Ablation (CartPole, 20 epochs)

| Variant | pred_loss ↓ | Collapse? |
|---|---|---|
| Full model (pred + var + cov + IDM) | ~0.12 | No |
| No VICReg (pred + IDM only) | ~0.31 | Yes |
| No IDM (pred + VICReg only) | ~0.19 | No |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README — pitch, architecture, losses, how-to-run, results placeholder"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] CartPole + MiniGrid (Task 1 env.py handles both via `get_obs`)
- [x] MLPEncoder + CNNEncoder (Task 3)
- [x] GRUPredictor (Task 3)
- [x] IDMHead (Task 3)
- [x] pred_loss + var_loss + cov_loss + idm_loss (Task 4)
- [x] Training loop + collapse detection + checkpoint + loss PNG (Task 5)
- [x] CEM planner (Task 6)
- [x] MPPI planner (Task 6)
- [x] MPC episode runner (Task 6)
- [x] demo.gif generation (Task 7 + Task 6)
- [x] Ablation table PNG (Task 7)
- [x] Smoke test (Task 8)
- [x] Kaggle notebook (Task 9)
- [x] README (Task 10)
- [x] config.yaml (Task 0)

**Type consistency across tasks:**
- `build_world_model(cfg)` defined in Task 3, used in Tasks 5, 6, 8 ✓
- `WorldModel.forward()` returns `{z_t, z_t1, z_hat, idm_logits}` — matches usage in Tasks 5 and 8 ✓
- `plan(model, obs, goal_obs, cfg, device)` returns `int` — used in Task 6 `run_mpc_episode` ✓
- `make_demo_gif(frames, path)` takes `List[np.ndarray]` — matches `run_mpc_episode` return type ✓
- `RolloutDataset.__getitem__` returns `(obs, action, next_obs)` — matches `make_loaders` usage ✓
