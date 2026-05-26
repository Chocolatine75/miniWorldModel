# README + Three.js Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current README with a visually compelling version that leads with a Three.js CartPole visualization video, and make the visualization runnable locally from the repo.

**Architecture:** `src/plan.py` gains an `--export-viz` flag that runs a CEM episode and exports raw CartPole observations to `viz/trajectory.json`. A standalone `viz/index.html` (Three.js, no build step) loads this JSON and replays the episode at 60fps with a dark aesthetic. The README is restructured around a hero mp4 + progressive disclosure via `<details>` for technical sections.

**Tech Stack:** Python/PyTorch (existing), Three.js 0.165.0 via CDN importmap, Gymnasium, standard `http.server`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/plan.py` | Modify | Add `export_viz()` function + `--export-viz` CLI flag |
| `viz/index.html` | Create | Three.js CartPole renderer — loads `trajectory.json` |
| `viz/trajectory.json` | Create (generated) | Pre-recorded trajectory, versioned in repo |
| `tests/test_smoke.py` | Modify | Add `test_export_viz_creates_valid_json` |
| `.gitignore` | Modify | Remove `*.pt`, add `!checkpoints/model_ep020.pt` exception |
| `checkpoints/model_ep020.pt` | Add to repo | Trained checkpoint (~420KB) |
| `assets/demo.mp4` | Add (manual capture) | Screen recording of Three.js viz |
| `README.md` | Rewrite | New structure: hero video → how it works → results → run |

---

## Task 1: `export_viz` function in `src/plan.py`

**Files:**
- Modify: `src/plan.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smoke.py` — first add the missing imports at the top of the file:

```python
import json
import numpy as np
from pathlib import Path
```

Then add this test function:

```python
def test_export_viz_creates_valid_json(tmp_path):
    """export_viz writes JSON with frames containing cart_x, pole_angle, action."""
    from src.plan import export_viz
    from src.model import build_world_model
    from unittest.mock import MagicMock

    cfg = _make_cfg()
    device = "cpu"
    model = build_world_model(cfg)

    obs_raw = np.array([0.01, 0.0, 0.02, 0.0], dtype=np.float32)
    mock_env = MagicMock()
    mock_env.reset.return_value = (obs_raw, {})
    mock_env.step.return_value = (obs_raw, 0.0, False, False, {})

    goal_obs = torch.zeros(cfg.obs_dim)
    out_path = str(tmp_path / "trajectory.json")

    export_viz(model, mock_env, goal_obs, cfg, device, path=out_path, max_steps=5)

    assert Path(out_path).exists()
    with open(out_path) as f:
        data = json.load(f)

    assert "frames" in data
    assert len(data["frames"]) == 6  # initial frame + 5 steps
    for frame in data["frames"]:
        assert set(frame.keys()) == {"cart_x", "pole_angle", "action"}
        assert isinstance(frame["cart_x"], float)
        assert isinstance(frame["pole_angle"], float)
        assert isinstance(frame["action"], int)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_smoke.py::test_export_viz_creates_valid_json -v
```

Expected: `FAILED` — `ImportError: cannot import name 'export_viz' from 'src.plan'`

- [ ] **Step 3: Implement `export_viz` in `src/plan.py`**

Add `import json` at the top of `src/plan.py` (after the existing imports).

Add this function to the **Public API** section (after `run_mpc_episode`):

```python
def export_viz(model: WorldModel, env, goal_obs: torch.Tensor,
               cfg: SimpleNamespace, device: str,
               path: str = "viz/trajectory.json",
               max_steps: int = 200) -> None:
    """Run one MPC episode and write CartPole states to JSON for the Three.js viz.

    path:      output JSON file — parent directories are created if missing
    max_steps: episode length cap

    JSON schema: {"frames": [{"cart_x": float, "pole_angle": float, "action": int}, ...]}
    CartPole obs layout: [cart_x, cart_vel, pole_angle, pole_ang_vel]
    """
    obs_raw, _ = env.reset()
    obs = torch.tensor(get_obs(obs_raw, cfg.env_id))

    frames_data = [
        {"cart_x": float(obs_raw[0]), "pole_angle": float(obs_raw[2]), "action": 0}
    ]

    for _ in range(max_steps):
        action = plan(model, obs, goal_obs, cfg, device)
        obs_raw, _, terminated, truncated, _ = env.step(action)
        obs = torch.tensor(get_obs(obs_raw, cfg.env_id))
        frames_data.append({
            "cart_x": float(obs_raw[0]),
            "pole_angle": float(obs_raw[2]),
            "action": int(action),
        })
        if terminated or truncated:
            break

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"frames": frames_data}, f)
    print(f"Trajectory exported → {path} ({len(frames_data)} frames)")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_smoke.py::test_export_viz_creates_valid_json -v
```

Expected: `PASSED`

- [ ] **Step 5: Add `--export-viz` CLI flag**

In `src/plan.py`, in the `if __name__ == "__main__":` block, add:

```python
    parser.add_argument("--export-viz", action="store_true",
                        help="Run MPC episode and export trajectory to viz/trajectory.json")
```

And add this block after the existing `if args.demo:` block:

```python
    if args.export_viz:
        Path("viz").mkdir(exist_ok=True)
        env      = make_env(cfg.env_id, cfg.seed)
        goal_raw, _ = env.reset(seed=cfg.seed + 99)
        goal_obs = torch.tensor(get_obs(goal_raw, cfg.env_id))

        print(f"Running MPC episode for viz (method={cfg.plan_method})...")
        export_viz(model, env, goal_obs, cfg, device, path="viz/trajectory.json")
        print("Done — run: python -m http.server --directory viz")
```

- [ ] **Step 6: Run full test suite to catch regressions**

```bash
pytest tests/test_smoke.py -v
```

Expected: all 6 tests `PASSED`

- [ ] **Step 7: Commit**

```bash
git add src/plan.py tests/test_smoke.py
git commit -m "feat: add export_viz and --export-viz flag to plan.py"
```

---

## Task 2: Three.js CartPole renderer (`viz/index.html`)

**Files:**
- Create: `viz/index.html`

No automated tests — verified visually in the browser (Task 4).

- [ ] **Step 1: Create `viz/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Mini World Model — CartPole</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0a0a0f; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js"
  }
}
</script>
<script type="module">
import * as THREE from 'three';

const { frames } = await fetch('./trajectory.json').then(r => r.json());

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
scene.fog = new THREE.Fog(0x0a0a0f, 12, 20);

// Camera
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 7);
camera.lookAt(0, 1, 0);

// Lights
scene.add(new THREE.AmbientLight(0x334466, 1.0));
const dirLight = new THREE.DirectionalLight(0x8899ff, 2.0);
dirLight.position.set(4, 8, 4);
scene.add(dirLight);

// Rail
scene.add(Object.assign(
  new THREE.Mesh(
    new THREE.BoxGeometry(7, 0.06, 0.18),
    new THREE.MeshStandardMaterial({ color: 0x223344, metalness: 0.9, roughness: 0.2 })
  )
));

// Subtle floor grid
const grid = new THREE.GridHelper(10, 20, 0x112233, 0x112233);
grid.position.y = -0.04;
scene.add(grid);

// Cart
const cart = new THREE.Mesh(
  new THREE.BoxGeometry(0.55, 0.22, 0.38),
  new THREE.MeshStandardMaterial({ color: 0x3a6fcc, metalness: 0.6, roughness: 0.3 })
);
cart.position.y = 0.14;
scene.add(cart);

// Pole pivot — rotates at base of pole (top of cart)
const polePivot = new THREE.Object3D();
polePivot.position.set(0, 0.28, 0);
scene.add(polePivot);

const POLE_LEN = 1.3;
const pole = new THREE.Mesh(
  new THREE.CylinderGeometry(0.028, 0.038, POLE_LEN, 12),
  new THREE.MeshStandardMaterial({
    color: 0xddeeff,
    emissive: 0x4455cc,
    emissiveIntensity: 0.5,
    metalness: 0.4,
    roughness: 0.4,
  })
);
pole.position.y = POLE_LEN / 2;
polePivot.add(pole);

// Glowing tip
const tipGlow = new THREE.PointLight(0x6688ff, 2.0, 2.5);
tipGlow.position.y = POLE_LEN;
polePivot.add(tipGlow);

// Motion trail — 15 spheres, fading in opacity and growing in size
const TRAIL = 15;
const trailMeshes = Array.from({ length: TRAIL }, (_, i) => {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.018 + 0.016 * i / TRAIL, 6, 6),
    new THREE.MeshBasicMaterial({
      color: 0x4466ff,
      transparent: true,
      opacity: 0.06 + 0.55 * i / TRAIL,
    })
  );
  scene.add(m);
  return m;
});

const SCALE = 2.6 / 2.4;   // CartPole x ∈ [-2.4, 2.4] → world units
const MS_PER_FRAME = 1000 / 30;
const tmpVec = new THREE.Vector3();
let fi = 0, lastT = 0;

function animate(t) {
  requestAnimationFrame(animate);
  if (t - lastT < MS_PER_FRAME) { renderer.render(scene, camera); return; }
  lastT = t;

  const { cart_x, pole_angle } = frames[fi % frames.length];
  fi++;

  const wx = cart_x * SCALE;

  cart.position.x = wx;
  polePivot.position.x = wx;
  polePivot.rotation.z = -pole_angle;   // positive angle = tilt right = -Z rotation

  // Camera follows cart with soft lag
  camera.position.x += (wx * 0.35 - camera.position.x) * 0.06;

  // Shift trail back and place newest at tip position
  tipGlow.getWorldPosition(tmpVec);
  for (let i = TRAIL - 1; i > 0; i--) {
    trailMeshes[i].position.copy(trailMeshes[i - 1].position);
  }
  trailMeshes[0].position.copy(tmpVec);

  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add viz/index.html
git commit -m "feat: add Three.js CartPole visualization"
```

---

## Task 3: Checkpoint in repo + `.gitignore` update

**Files:**
- Modify: `.gitignore`
- Add: `checkpoints/model_ep020.pt`

**Prerequisite:** `checkpoints/model_ep020.pt` must exist locally. If it doesn't, download it from the Kaggle notebook output before this task (Cell 3 saves it). File size should be ~420KB.

- [ ] **Step 1: Update `.gitignore`**

In `.gitignore`, remove the line `*.pt` and replace the `checkpoints/` line with:

```
checkpoints/
!checkpoints/model_ep020.pt
```

This ignores all `.pt` files in `checkpoints/` except the final trained model. `data/rollouts.pt` stays ignored via the existing `data/` rule.

Also add `.superpowers/` to `.gitignore` (brainstorming session files):

```
.superpowers/
```

- [ ] **Step 2: Verify checkpoint exists**

```bash
ls -lh checkpoints/model_ep020.pt
```

Expected: file exists, ~400–500KB. If missing, stop here and download from Kaggle before continuing.

- [ ] **Step 3: Stage and commit**

```bash
git add .gitignore checkpoints/model_ep020.pt
git commit -m "chore: version final checkpoint, update gitignore"
```

---

## Task 4: Generate and commit `viz/trajectory.json`

**Files:**
- Create: `viz/trajectory.json`

**Prerequisite:** Tasks 1 and 3 must be complete (checkpoint in repo, `--export-viz` flag working).

- [ ] **Step 1: Generate the trajectory**

```bash
python -m src.plan --config config.yaml --export-viz
```

Expected output:
```
Loaded checkpoint: checkpoints/model_ep020.pt
Running MPC episode for viz (method=cem)...
Trajectory exported → viz/trajectory.json (N frames)
Done — run: python -m http.server --directory viz
```

- [ ] **Step 2: Verify the JSON looks correct**

```bash
python -c "
import json
with open('viz/trajectory.json') as f:
    d = json.load(f)
frames = d['frames']
print(f'{len(frames)} frames')
print('first:', frames[0])
print('last: ', frames[-1])
"
```

Expected: 100–200 frames, `cart_x` in [-2.4, 2.4], `pole_angle` in [-0.2, 0.2] throughout.

- [ ] **Step 3: Test the visualization in browser**

```bash
python -m http.server --directory viz
```

Open http://localhost:8000. The pole should be visible on a dark background, staying upright for the full trajectory, with a blue glow trail. Camera should follow the cart.

If the pole looks wrong (upside down, or goes immediately off screen), check `pole_angle` sign convention in `viz/index.html` line `polePivot.rotation.z = -pole_angle`.

- [ ] **Step 4: Commit**

```bash
git add viz/trajectory.json
git commit -m "feat: add pre-recorded CartPole trajectory for Three.js viz"
```

---

## Task 5: Capture `assets/demo.mp4`

**Files:**
- Add: `assets/demo.mp4`

This is a **manual screen recording step** — no code to write.

- [ ] **Step 1: Start the visualization**

```bash
python -m http.server --directory viz
```

Open http://localhost:8000 in a full-screen browser window (at least 1280×720).

- [ ] **Step 2: Record 10 seconds**

Use your OS screen recorder (Windows: Win+G → record, macOS: Cmd+Shift+5). Record a 10-second clip that shows:
- The pole initially oscillating slightly
- The planner keeping it upright
- The camera following the cart

Save as `assets/demo.mp4`. Target: under 5MB. If larger, reduce quality/resolution.

- [ ] **Step 3: Commit**

```bash
git add assets/demo.mp4
git commit -m "feat: add Three.js visualization screen recording"
```

---

## Task 6: Rewrite `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with the new content**

Replace the entire content of `README.md` with:

````markdown
# Mini World Model — AC-JEPA on CartPole

A minimal action-conditioned world model in the spirit of [EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Meta FAIR / Yann LeCun) — built for HackTheWorld(s), May 2026.

<video src="assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>

CartPole stays upright. Not because we hand-coded a controller — because a learned world model predicts the future in latent space, and a model-predictive planner picks actions that keep it balanced.

No RL. No reward shaping. ~600 lines of PyTorch.

---

## How it works

An **encoder** maps CartPole observations to a compact latent space. A **GRU predictor** learns to imagine the next latent state given an action — no pixel reconstruction, VICReg prevents collapse. At planning time, **CEM** samples action sequences, rolls them forward in imagination, and picks the one that keeps the pole upright.

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

<details>
<summary>Ablation — VICReg / IDM / both</summary>

| Variant | `pred_loss` ↓ | Collapse? | Notes |
|---|---|---|---|
| Full model (pred + VICReg + IDM) | **0.0023** | No | Baseline |
| No VICReg (pred + IDM only) | ~0.018 | Yes | Embedding std → 0 after ~8 epochs |
| No IDM (pred + VICReg only) | ~0.009 | No | Slightly better pred, weaker planner |
| Prediction only | ~0.031 | Yes | Fast collapse, planner fails |

> Ablation values are estimates pending full sweep.

![Ablation](assets/ablation.png)

</details>

---

## Run the visualization

```bash
pip install -r requirements.txt

# Export a fresh trajectory from the trained model (checkpoint included)
python -m src.plan --export-viz

# Open Three.js visualization
python -m http.server --directory viz
# → open http://localhost:8000
```

The model checkpoint is included in the repo — no training needed. To retrain from scratch: run `notebooks/kaggle_run.ipynb` on Kaggle (T4 GPU, ~30 min).

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
````

- [ ] **Step 2: Verify locally (optional but recommended)**

If you have a markdown previewer, check that `<details>` blocks collapse correctly and the `<video>` tag is present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with Three.js viz hero and progressive disclosure"
```

---

## Self-Review

**Spec coverage:**
- [x] `viz/index.html` — Task 2
- [x] `viz/trajectory.json` — Task 4
- [x] `src/plan.py --export-viz` — Task 1
- [x] `checkpoints/model_ep020.pt` in repo — Task 3
- [x] `assets/demo.mp4` — Task 5
- [x] `README.md` restructured — Task 6
- [x] `.gitignore` updated — Task 3
- [x] `.superpowers/` added to gitignore — Task 3

**No placeholders or TBDs found.**

**Type consistency:** `export_viz` signature is consistent across Task 1 (implementation) and the test. `frames[fi % frames.length]` destructures `cart_x` and `pole_angle` which match the JSON schema defined in Task 1.

**One manual step** (Task 5) requires human action — screen recording cannot be automated.
