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
    from src.env import get_obs

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
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
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
