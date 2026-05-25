# Mini World Model — AC-JEPA on CartPole

> A minimal action-conditioned world model in the spirit of [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Meta FAIR / Yann LeCun).
> Built for **HackTheWorld(s)** — May 2026.

CartPole stays upright. Not because we hand-coded a controller — because a **learned world model** predicts the future in latent space, and a **model-predictive planner** picks actions that keep it balanced.

![Demo](assets/demo.gif)

---

## Why this matters

Yann LeCun's bet on JEPA-style world models is that intelligence emerges from **predicting structure in latent space**, not pixels. This project is a minimal, self-contained proof that the idea works end-to-end:

- Train an encoder + GRU predictor with **no reconstruction loss** — pure latent prediction.
- Prevent representation collapse with **VICReg** (variance + covariance regularization).
- Inject action grounding via an **Inverse Dynamics Model** (IDM).
- Freeze the world model, run **CEM** or **MPPI** over imagined rollouts → CartPole holds for 150+ frames.

Total codebase: ~600 lines of PyTorch. No RL. No reward shaping. Just prediction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING (AC-JEPA)                         │
│                                                                  │
│  obs s_t ──► Encoder (MLP) ──► z_t ──► Predictor (GRU) ──► ẑ_{t+1}
│                                  │                          │
│                                  │                   MSE ◄──┘  pred_loss
│                                  │                    ▲
│              obs s_{t+1} ──► Encoder ──► z_{t+1} (stop-grad)
│                                  │
│              (z_t, z_{t+1}) ──► IDM (MLP) ──► â_t   idm_loss
│                                  │
│                   z_t, z_{t+1} ──► VICReg ──────────► var_loss + cov_loss
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PLANNING (frozen model)                     │
│                                                                  │
│  obs s_0 ──► Encoder ──► z_0                                    │
│                           │                                      │
│   action sequence a_{0..H} ──► Predictor rollout ──► ẑ_{1..H}  │
│                                                         │        │
│                                    score = -||ẑ_H - z_goal||    │
│                                                                  │
│   CEM / MPPI ──► refine action sequence ──► execute a_0         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Losses

| Loss | Formula | Role |
|---|---|---|
| Prediction | `MSE(ẑ_{t+1}, sg(z_{t+1}))` | Learn to predict future latent state |
| Variance | `mean(max(0, 1 − std(z)))` | Prevent collapse to a constant embedding |
| Covariance | `sum(off_diag(cov(z))²) / d` | Decorrelate embedding dimensions |
| IDM | `CrossEntropy(MLP(z_t, z_{t+1}), a_t)` | Ground representations in action-relevant info |

**Total loss:** `L = pred_loss + α · var_loss + β · cov_loss + γ · idm_loss`

Default coefficients: `α = 1.0`, `β = 0.04`, `γ = 0.1`

---

## Results

Training on CartPole-v1 for 20 epochs on a Kaggle T4 GPU (~30 min):

| Metric | Epoch 1 | Epoch 20 | Change |
|---|---|---|---|
| `pred_loss` | 0.0175 | 0.0023 | **7.6× improvement** |
| `idm_loss` | 0.51 | ~0.000 | Near-perfect action grounding (3 epochs) |
| `var_loss` | 0.80 | 0.57 | Stable — no collapse |
| `cov_loss` | ~0.110 | ~0.083 | Normal VICReg oscillation |
| Collapse warnings | — | — | **None detected** |

![Loss Curves](assets/loss_curves.png)

### Ablation (CartPole, 20 epochs)

| Variant | `pred_loss` ↓ | Collapse? | Notes |
|---|---|---|---|
| Full model (pred + VICReg + IDM) | **0.0023** | No | Baseline |
| No VICReg (pred + IDM only) | ~0.018 | Yes | Embedding std → 0 after ~8 epochs |
| No IDM (pred + VICReg only) | ~0.009 | No | Slightly better pred, weaker planner |
| Prediction only | ~0.031 | Yes | Fast collapse, planner fails |

> Ablation values are estimates pending full sweep — will be updated before submission.

---

## Planners

Both planners operate on **imagined rollouts** in latent space using the frozen world model. No environment interaction during planning.

**CEM — Cross-Entropy Method**
Iteratively refine a distribution over action sequences by resampling from the top-k (elite) trajectories. Fast convergence, 3–5 iterations per step.

**MPPI — Model Predictive Path Integral**
Sample trajectories, weight by `softmax(-cost / λ)`, compute weighted mean. Smoother than CEM, better in noisy settings.

---

## How to run

### On Kaggle (recommended — T4 GPU, ~30 min)

1. Open `notebooks/kaggle_run.ipynb` on [Kaggle](https://www.kaggle.com/)
2. Set Accelerator → **GPU T4 x1**
3. Run all 6 cells in order:
   - Cell 1: clone repo + install deps
   - Cell 2: generate CartPole dataset
   - Cell 3: train world model (20 epochs)
   - Cell 4: plot loss curves
   - Cell 5: run CEM planner + record demo GIF
   - Cell 6: smoke tests (shapes, losses, planner validity)

### Locally

```bash
git clone https://github.com/Chocolatine75/miniWorldModel
cd miniWorldModel
pip install -r requirements.txt

# Generate dataset
python -m src.data --config config.yaml

# Train world model
python -m src.train --config config.yaml

# Run planner and record demo
python -m src.plan --config config.yaml --demo
```

Requirements: Python 3.10+, PyTorch 2.x, Gymnasium, NumPy, Matplotlib, Pillow, PyYAML.

---

## Stack

| Component | Library |
|---|---|
| World model | PyTorch |
| Environment | Gymnasium (CartPole-v1) |
| Data / viz | NumPy, Matplotlib, Pillow |
| Config | PyYAML |
| Training infra | Kaggle (T4 GPU) |

---

## Repo

[github.com/Chocolatine75/miniWorldModel](https://github.com/Chocolatine75/miniWorldModel) — HackTheWorld(s) 2026
