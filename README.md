# Mini World Model — AC-JEPA on CartPole

A minimal action-conditioned world model in the spirit of [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Meta FAIR / Yann LeCun) — built for HackTheWorld(s), May 2026.

<video src="assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>

**Left:** trained world model — CEM planner imagines 20 steps ahead, pole stays balanced.  
**Right:** untrained model, same planner — latent space is noise, pole falls in under 10 steps.

No RL. No reward shaping. ~600 lines of PyTorch.

---

## How it works

An **encoder** maps CartPole observations to a compact latent space. A **GRU predictor** learns to imagine the next latent state given an action — no pixel reconstruction, VICReg prevents collapse. At planning time, **CEM** samples 1024 action sequences, rolls them forward 20 steps in imagination, and picks the one that keeps the pole upright.

<details>
<summary>Architecture — Encoder · GRU Predictor · IDM · VICReg</summary>

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

**Encoder** — two-layer MLP, obs → 32-dim embedding. No reconstruction loss.

**GRU Predictor** — takes (z_t, action_onehot), returns ẑ_{t+1}. Trained with MSE against the stop-gradient target z_{t+1}.

**VICReg** — variance + covariance regularization on the embeddings. Prevents the encoder from collapsing to a constant output.

**IDM (Inverse Dynamics Model)** — MLP that predicts the action taken from (z_t, z_{t+1}). Grounds representations in action-relevant information. Used only at training time.

</details>

---

## Results

20 epochs on CartPole-v1, Kaggle T4 GPU (~30 min).

| Metric | Epoch 1 | Epoch 20 | Change |
|---|---|---|---|
| `pred_loss` | 0.0175 | 0.0023 | 7.6× |
| `idm_loss` | 0.51 | ~0.000 | near-perfect after 3 epochs |
| `var_loss` | 0.80 | 0.57 | stable — no collapse |
| `cov_loss` | ~0.110 | ~0.083 | normal VICReg oscillation |

![Loss Curves](assets/loss_curves.png)

---

## Run the visualization

```bash
pip install -r requirements.txt

# Generate both trajectories (trained + untrained) — checkpoint included
python -m src.plan --export-viz

# Open split-screen Three.js visualization
python -m http.server --directory viz
# → open http://localhost:8000
```

The visualization shows both sides simultaneously: trained model (pole stays up) vs untrained model (pole falls in under 10 steps). Checkpoint included — no training needed. To retrain from scratch: run `notebooks/kaggle_run.ipynb` on Kaggle (T4 GPU, ~30 min).

---

## Repo structure

```
src/
  model.py       — Encoder, GRU Predictor, IDM (WorldModel)
  train.py       — training loop (VICReg + IDM + pred loss)
  plan.py        — CEM / MPPI planner + --export-viz
  env.py         — Gymnasium helpers
  viz.py         — GIF export

viz/
  index.html     — Three.js split-screen renderer (no build step)
  trajectory.json           — pre-recorded trained trajectory
  trajectory_untrained.json — pre-recorded untrained trajectory

checkpoints/
  model_ep020.pt — trained checkpoint (~420 KB)

notebooks/
  kaggle_run.ipynb — full training pipeline on Kaggle T4

assets/
  demo.mp4       — screen recording of the viz
  loss_curves.png
```

<details>
<summary>Stack</summary>

| Component | Library |
|---|---|
| World model | PyTorch |
| Environment | Gymnasium (CartPole-v1) |
| Visualization | Three.js |
| Data / viz | NumPy, Matplotlib, Pillow |
| Config | PyYAML |
| Training infra | Kaggle (T4 GPU) |

</details>

---

[github.com/Chocolatine75/miniWorldModel](https://github.com/Chocolatine75/miniWorldModel) — HackTheWorld(s) 2026
