# direct-obs-scale — Observation scaling  `L1 · FILL`

**This is the only Stage 2 exercise that deals with the Direct stack.** Manager declares observations as a list of
`ObsTerm`s, but Direct assembles the tensors directly inside `_get_observations`.
This is where you see how the two APIs do the same thing differently.

## Goal

**Bring each group of the observation to O(1) magnitude.**

```
lin_vel   x 2.0     joint_pos x 1.0
ang_vel   x 0.25    joint_vel x 0.05
gravity and actions are already O(1) — not scaled
```

## What to fill in

`TODO(direct-obs-scale)` in `starter/dreamwaq_env.py` — four lines.

## Why this matters — a bug that actually happened

This is porting bug #1 recorded in `debugging.qmd`.

> **Missing observation scale** — raw `dof_vel` (±20 rad/s) overwhelms the commands (±1)

In numbers it's clear. Without scaling, 12 of the 45 observation dims (joint velocities) have
**values 20× larger than the commands**. From the network's point of view, the commands become nearly invisible noise.

The result is **a policy that ignores the commands and stands still.** It's exactly the same failure as the
"0.4426 reward floor when standing still" in the Stage 3 toy environment.

Check 7 measures this ratio directly.

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | The observation is 45-dimensional |
| 2 | `ang_vel` × 0.25 |
| 3 | `joint_vel` × 0.05 |
| 4 | `joint_pos` × 1.0 |
| 5 | **gravity is not scaled** |
| 6 | **`_true_lin_vel_b` keeps the raw value from *before* scaling** |
| 7 | **After scaling, the `joint_vel`/command ratio is < 3** (20× without scaling) |

Number 6 is subtle. `_true_lin_vel_b` is **CENet's training target** (Stage 4).
If you store the scaled value, CENet learns velocity at the wrong scale.
The original also grabs the raw value with `.clone()` right before scaling.

## How the checker runs without Isaac Sim

`_get_observations` is a method, but the checker builds a fake `self` with only the needed attributes and
**calls it directly, unbound**.

```python
get_obs = load_module(path).DreamWaQEnv._get_observations
get_obs(fake_self)      # doesn't launch an env
```

`exercises/tools/fake_isaaclab.py` intercepts `isaaclab.*` imports with a meta path finder,
so the 25-second Isaac Sim startup isn't needed. It's done in 2 seconds.
