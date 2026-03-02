# 👾 Lunar Lander RL Project

## Environment Description

**Reference:** [Lunar Lander – Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/#arguments)

### Problem description

**Goal:** Land the lunar lander safely on the landing pad at coordinates **(0, 0)**. The agent must choose thrust (main engine and left/right orientation engines) so that the lander touches the ground at low speed, roughly upright, with both legs in contact, and without crashing. Fuel is unlimited.

### Environment dynamics

#### State space (observation)

8-dimensional vector: 6 continious, 2 boolean components

| Index | Meaning            | Min    | Max     |
|-------|--------------------|--------|---------|
| 0     | x position         | -2.5   | 2.5     |
| 1     | y position         | -2.5   | 2.5     |
| 2     | x velocity         | -10    | 10      |
| 3     | y velocity         | -10    | 10      |
| 4     | angle (rad)        | -2π    | 2π      |
| 5     | angular velocity   | -10    | 10      |
| 6     | left leg contact   |       0 or 1      |
| 7     | right leg contact  | 0 or 1      |

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

So: **next state = physics(state, action, gravity, wind)** (Box2D step).

#### Deterministic vs stochastic transitions:

- **Stochastic:**   
  - If `enable_wind=True`: wind (and thus transitions) depend on random `C` and time.  
- **Deterministic:** With `enable_wind=False`, given the same initial state and action sequence, transitions are deterministic

### Episode termination

An episode ends when any of the following happens:

| Condition | Meaning |
|-----------|---------|
| **Crash** | Lander body contacts the moon surface (unsafe landing or collision). |
| **Out of viewport** | `x` &gt; 1 (lander leaves the visible area). |
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

“Solved” episode: Total Return $\ge$ 200

---

## Installation:

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```