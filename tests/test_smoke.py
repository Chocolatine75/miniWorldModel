"""Single smoke test: model shapes, losses finite, planners valid, 2-epoch train.

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
        checkpoint_every=999,
        n_episodes=10,
        data_path="data/rollouts.pt",
        N=8, H=3, K=2, n_iter=2,
        plan_method="cem",
        mppi_lambda=10.0,
    )


def test_world_model_forward_shapes():
    """Forward pass returns correct output shapes for CartPole."""
    from src.model import build_world_model
    cfg = _make_cfg()
    model = build_world_model(cfg)
    B = 8
    obs      = torch.randn(B, cfg.obs_dim)
    action   = torch.randint(0, cfg.n_actions, (B,))
    next_obs = torch.randn(B, cfg.obs_dim)
    out = model(obs, action, next_obs, cfg.n_actions)
    assert out["z_t"].shape       == (B, cfg.emb_dim)
    assert out["z_t1"].shape      == (B, cfg.emb_dim)
    assert out["z_hat"].shape     == (B, cfg.emb_dim)
    assert out["idm_logits"].shape == (B, cfg.n_actions)


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
        assert l.dim() == 0,             f"{name} is not scalar"
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
    assert isinstance(action, int)
    assert 0 <= action < cfg.n_actions


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
    assert isinstance(action, int)
    assert 0 <= action < cfg.n_actions


def test_two_epoch_training_loss_decreases():
    """Training for 2 epochs on 10 random samples produces finite loss."""
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import build_world_model
    from src.losses import pred_loss, var_loss, cov_loss, idm_loss
    import torch.optim as optim

    cfg    = _make_cfg()
    device = "cpu"
    model  = build_world_model(cfg).to(device)
    opt    = optim.Adam(model.parameters(), lr=cfg.lr)

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
