# 👾 Lunar Lander RL Project

This project implements and compares three policy-gradient algorithms — **REINFORCE**, **PPO**, and **GRPO** — on the discrete Lunar Lander environment from Gymnasium. We evaluate all methods on **mean return** and **success rate** (return ≥ 200) and provide training curves, hyperparameter sweeps, and animated demos of the trained agents.

---

## Environment Description

**Reference:** [Lunar Lander – Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/#arguments)

### Problem description

**Goal:** Land the lunar lander safely on the landing pad at coordinates **(0, 0)**. The agent must choose thrust (main engine and left/right orientation engines) so that the lander touches the ground at low speed, roughly upright, with both legs in contact, and without crashing. Fuel is unlimited.

### Environment dynamics

#### State space (observation)

8-dimensional vector: 6 continuous, 2 boolean components

| Index | Meaning            | Min    | Max     |
|-------|--------------------|--------|---------|
| 0     | x position         | -2.5   | 2.5     |
| 1     | y position         | -2.5   | 2.5     |
| 2     | x velocity         | -10    | 10      |
| 3     | y velocity         | -10    | 10      |
| 4     | angle (rad)        | -2π    | 2π      |
| 5     | angular velocity   | -10    | 10      |
| 6     | left leg contact   | 0 or 1 |         |
| 7     | right leg contact  | 0 or 1 |         |

- **Initial state:** The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

#### Action space

`Discrete(4)`:
- 0: do nothing
- 1: left engine
- 2: main engine
- 3: right engine

#### Transition logic

- **Gravity:** Constant vertical acceleration (default `gravity = -10`, bounded in [0, -12]).
- **Engines:** Main engine: vertical thrust; side engines: thrust + torque (orientation-dependent per implementation). Discrete: on/off.
- **Wind (optional):** If `enable_wind=True`, wind uses  
  `tanh(sin(2k(t+C)) + sin(πk(t+C)))` with `k=0.01`, `C` random in [-9999, 9999] at reset; `wind_power` and `turbulence_power` scale linear and rotational wind.
  - Remark: We did not use wind in our Project

So: **next state = physics(state, action, gravity, wind)** (Box2D step).

#### Deterministic vs stochastic transitions

- **Stochastic:**   
  - If `enable_wind=True`: wind (and thus transitions) depend on random `C` and time.  
- **Deterministic:** With `enable_wind=False`, given the same initial state and action sequence, transitions are deterministic.

### Episode termination

An episode ends when any of the following happens:

| Condition | Meaning |
|-----------|---------|
| **Crash** | Lander body contacts the moon surface (unsafe landing or collision). |
| **Out of viewport** | `x` > 1 (lander leaves the visible area). |
| **Not awake** | Box2D marks the body as sleeping (at rest, no collisions). |

Termination is detected at the *start* of the step after the event (e.g. crash reward is given when stepping into the terminal state).

### Reward function

Per-step and terminal rewards (episode return = sum of step rewards + terminal reward):

| Event | Reward |
|-------|--------|
| Closer to landing pad (0,0) | Positive (scaled by distance) |
| Farther from landing pad | Negative (scaled by distance) |
| Slower horizontal/vertical speed | Positive |
| Faster horizontal/vertical speed | Negative |
| More tilted (angle away from horizontal) | Negative |
| Each leg in contact with ground | **+10** per leg per step |
| Side engine firing (per frame) | **-0.03** |
| Main engine firing (per frame) | **-0.3** |
| **Landing safely** (episode end) | **+100** |
| **Crashing** (episode end) | **-100** |

"Solved" episode: Total Return $\ge$ 200

---

## Algorithms

All algorithms use a shared neural network architecture (`NeuralPolicy`): a 3-layer MLP with ReLU activations producing action logits, and optionally a separate `ValueNetwork` of the same shape outputting a scalar state value.

### REINFORCE

Implemented in `src/reinforce.py`. A single `train_reinforce()` function supports **four baseline variants** controlled by the `--baseline` flag:

| Variant | `--baseline` flag | Description |
|---------|-------------------|-------------|
| **Vanilla REINFORCE** | `none` | No baseline; raw Monte-Carlo returns $G_t$ used as advantage. |
| **NN Value baseline** | `nn` | Learned `ValueNetwork` $V_\phi(s)$ as baseline; advantage $A_t = G_t - V_\phi(s_t)$, no GAE. |
| **GAE without NN** | `gae_no_nn` | GAE with per-timestep batch-averaged returns as value estimate (no learned network). |
| **GAE + NN** | `gae_nn` | Full GAE ($\lambda = 0.95$) with a learned `ValueNetwork`. |

All variants support entropy regularization (`--entropy-coef`) and advantage normalization (on by default).

### PPO (Proximal Policy Optimization)

Implemented in `src/ppo.py`.

$$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

Supports an optional KL-divergence penalty (`--beta-kl`) for additional policy regularization:

$$L = L^{\text{CLIP}} + c_v L^{\text{value}} - c_e H[\pi] + \beta_{\text{KL}} \, D_{\text{KL}}(\pi_{\text{old}} \| \pi)$$

### GRPO (Group Relative Policy Optimization)

Implemented in `src/grpo.py`. Like PPO but **without a value network**. Advantages are Monte-Carlo returns normalized across the rollout batch (the "group"):

$$A_t = \frac{G_t - \mu_G}{\sigma_G + \varepsilon}$$

Uses the same clipped surrogate objective as PPO but relies solely on group normalization instead of a learned baseline.

---

## Project Structure

```
lunarlander-rl/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train_reinforce.sh
│   ├── train_ppo.sh
│   ├── train_grpo.sh
│   └── eval.sh
├── src/
│   ├── reinforce.py
│   ├── ppo.py
│   ├── grpo.py
│   └── eval.py
├── images/                 # Training curves
├── results/                # Evaluation outputs (JSON, GIFs)
└── visualize.ipynb         # Visualization notebook
```

### Source files and classes

| File | Classes / Functions | Purpose |
|------|-------------------|---------|
| `src/reinforce.py` | `NeuralPolicy`, `ValueNetwork`, `compute_returns`, `compute_gae`, `train_reinforce` | Policy and value network definitions (shared by all algorithms), REINFORCE training with 4 baseline modes |
| `src/ppo.py` | `compute_gae_buffer`, `train_ppo` | PPO training with clipped surrogate, GAE, optional KL penalty |
| `src/grpo.py` | `compute_mc_buffer`, `train_grpo` | GRPO training — value-free PPO-style with group-normalized MC returns |
| `src/eval.py` | `evaluate_policy`, `_load_policy` | Load a saved checkpoint, run evaluation episodes, compute metrics, save GIF |

---

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `torch`, `numpy`, `imageio`, `box2d`, `pygame`, `matplotlib`, `tqdm`, `pandas`.

---

## Usage

### Training

All training scripts are in `./scripts/`. Each script trains from scratch, saves periodic checkpoints and `best_policy.pt`, and logs per-iteration metrics to a CSV file.

**REINFORCE:**

```bash
CUDA_VISIBLE_DEVICES=0 python src/reinforce.py \
  --n-iter            300       \
  --n-episodes        512       \
  --num-envs          64        \
  --alpha             1e-3      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --baseline          nn        \
  --baseline-lr       1e-4      \
  --hidden-size       512       \
  --seed              42        \
  --exp-dir           runs/reinforce_nn \
  --save-every-n      50
```

**PPO:**

```bash
CUDA_VISIBLE_DEVICES=0 python src/ppo.py \
  --n-iter            300       \
  --num-envs          64        \
  --alpha             3e-4      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --hidden-size       1024      \
  --seed              42        \
  --save-every-n      10        \
  --exp-dir           runs/ppo_exp \
  --batch-size        512       \
  --clip-eps          0.1       \
  --max-grad-norm     1.0       \
  --beta-kl           0.5
```

**GRPO:**

```bash
CUDA_VISIBLE_DEVICES=0 python src/grpo.py \
  --n-iter            500       \
  --num-envs          64        \
  --alpha             3e-4      \
  --gamma             0.99      \
  --entropy-coef      0.01      \
  --hidden-size       1024      \
  --seed              42        \
  --save-every-n      10        \
  --exp-dir           runs/grpo_exp \
  --batch-size        512       \
  --clip-eps          0.1       \
  --max-grad-norm     1.0
```

Or simply run via the provided shell scripts:

```bash
bash scripts/train_reinforce.sh
bash scripts/train_ppo.sh
bash scripts/train_grpo.sh
```

#### Training output

Each training run creates an experiment directory (`--exp-dir`) containing:
- `metrics.csv` — per-iteration logs: mean/best/min/max return, policy loss, value loss, entropy, KL divergence (PPO)
- `checkpoints/` — periodic model checkpoints (`.pt` files)
- `best_policy.pt` — state dict of the policy with the highest mean return during training

### Evaluation

Use `src/eval.py` to load a trained checkpoint, run evaluation episodes, and save results:

```bash
python src/eval.py \
  --checkpoint  runs/ppo_exp/best_policy.pt \
  --results-dir results/ppo \
  --n-episodes  100
```

The evaluation script:
1. Loads the policy network from the checkpoint
2. Runs `--n-episodes` episodes (stochastic by default, `--deterministic` for greedy)
3. Saves a **JSON** file with mean return, std, min, max, and success rate
4. Saves a **GIF** of a successful landing (or the last episode if none succeed)

Batch evaluation examples are provided in `scripts/eval.sh`.

---

## Results

### Quantitative comparison

- Evaluation performed over 100 sampled trajectories:

| Algorithm | Mean Return | Success Rate |
|-----------|------------|--------------|
| REINFORCE (NN Value baseline) | 137.1 | 0.01 |
| PPO ($\beta_{\text{KL}}$ = 0.5) | **288.97** | **0.99** |
| GRPO ($\beta_{\text{KL}}$ = 0) | 97.6 | 0.09 |

PPO with a moderate KL penalty achieves near-perfect success rate and well above the 200-point "solved" threshold. REINFORCE with NN baseline learns a reasonable policy but fails to consistently solve the task. GRPO without KL regularization underperforms both, suggesting that a learned value baseline is critical for stable learning in this environment.

### Training curves

**REINFORCE** (all baseline variants):

![REINFORCE training curves](images/reinforce.png)

**PPO** — effect of KL penalty coefficient:

![PPO KL sweep](images/ppo_kl.png)

**PPO** — effect of clipping epsilon:

![PPO clip epsilon sweep](images/ppo_clip_eps.png)

**GRPO:**

![GRPO training curves](images/grpo.png)

### Trained agent demos

**Best REINFORCE agent** (NN Value baseline):

![REINFORCE best policy](results/reinforce/reinforce_nn_v3/best_policy.gif)

**Best PPO agent** ($\beta_{\text{KL}}$ = 0.5):

![PPO best policy](results/ppo/ppo_exp_005_kl_0.2/best_policy.gif)

**Best GRPO agent:**

![GRPO best policy](results/grpo/best_policy.gif)

---

## Reproducibility

- All experiments use `--seed 42` for reproducible results
- Full hyperparameter configurations are documented in the training scripts (`scripts/`)
- Evaluation results (JSON) and checkpoints are saved in `results/` and `runs/`
