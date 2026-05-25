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
    if not Path(cfg.data_path).exists():
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
