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
3. Run all cells — takes ~30 min on CartPole

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
