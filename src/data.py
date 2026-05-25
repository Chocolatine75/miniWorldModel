"""Rollout collection and PyTorch Dataset for (obs, action, next_obs) transitions."""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.env import make_env, get_obs, set_seed


def collect_random_rollouts(cfg: SimpleNamespace) -> None:
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

    torch.save(all_transitions, cfg.data_path)
    print(f"Saved {len(all_transitions)} transitions → {cfg.data_path}")


class RolloutDataset(Dataset):
    """PyTorch Dataset that loads (obs, action, next_obs) from a .pt file.

    path: path to the .pt file saved by collect_random_rollouts()
    __getitem__ returns: (obs: Tensor, action: Tensor, next_obs: Tensor)
    """

    def __init__(self, path: str):
        self.data: List[Dict[str, torch.Tensor]] = torch.load(path, weights_only=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns obs, action, next_obs for transition at index idx."""
        item = self.data[idx]
        return item["obs"], item["action"], item["next_obs"]


def make_loaders(cfg: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Load rollouts.pt and return (train_loader, val_loader).

    cfg: SimpleNamespace with fields: batch_size
    90% train / 10% val split by random shuffle.
    """
    dataset = RolloutDataset(cfg.data_path)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)

    collect_random_rollouts(cfg)
