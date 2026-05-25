"""Gymnasium environment helpers: creation, seeding, observation extraction."""

import random
from typing import Union, Optional
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int, env: Optional[gym.Env] = None) -> None:
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
