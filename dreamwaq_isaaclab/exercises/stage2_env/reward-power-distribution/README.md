# reward-power-distribution — Power distribution penalty  `L2 · BUILD`

## Goal

Penalize **load concentrating on particular motors**.

```
reward = var_j(tau_j * theta_dot_j)^2
```

**Power distribution** in Table I of the paper. Whereas the previous exercise (`joint_power`) penalizes the **total** power,
this term penalizes the **imbalance** between joints.

## What to fill in

`TODO(reward-power-distribution)` in `starter/rewards.py` — the body of `power_distribution_l2`.

**Two differences from the previous exercise**

1. **No absolute value** — it's the variance of the signed power as-is
2. Look at the **variance over the joint axis (`dim=-1`)**, not the sum, and **square it once more**

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | Returns `(num_envs,)` |
| 2 | 0 when all joint powers are equal |
| 3 | **0 if uniform, even with large power** — it looks at imbalance, not the total |
| 4 | The `var²` value is exact on a known distribution |
| 5 | **The variance was squared once more** (detects a missing `torch.square`) |
| 6 | **No absolute value was used** |
| 7 | Respects `asset_cfg.joint_ids` |

Number 3 is the very definition of this term. 5·6 directly target common mistakes.

## Why this term exists — the paper's explanation

Paper §II-B-4:

> However, this reward minimizes the overall power without considering each motor's power usage
> balance. Consequently, in the long run, some motors might overheat faster than others.
> Therefore, we introduced a power distribution reward to **reduce motor overheating in the real
> world** by penalizing motors' power with high variance over all motors used on the robot.

Motors don't burn out inside a simulator. **This term is for deployment on real hardware.**
It connects to the paper's §III-G outdoor experiment, which reported that when climbing a hill "front legs' motors may easily overheat and
enter the overheat protection mode".

## A story about the weight (worth reading)

The weight of this term **differs by 10× between the paper and the authors' implementation.**

| | Value |
|---|---|
| Paper Table I | `−10⁻⁵` |
| Authors' `legged_gym` config (`a1_config.py`) | `−1.0e−6` |

Which one is right? This repo first used the paper value, then **measured it and switched to `−1e−6`.**
There are three reasons.

1. **The other 11 terms match to the decimal.** That config class even has the comment `"""SAME reward functions
   with the paper"""`. With only 1 of 12 terms off by exactly 10×, it's more natural to read it as
   a **typo in print** than a design change.
2. **We actually measured the magnitudes.** Weighted per-step `|r|` in a 512 env × 300 step probe:

   | Term | Mean | Max |
   |---|---:|---:|
   | `track_lin_vel_xy_exp` (tracking, the main objective of this env) | 14.2 | 75 |
   | `power_distribution` @ `−1e−5` | **16.2** | **8447** |
   | `power_distribution` @ `−1e−6` | 1.62 | 845 |

   At `−1e−5` **the penalty is larger than the main objective.** It's a dominating term, not a regularizer.
   `var(τ·θ̇)²` scales with the **4th power** of torque and velocity, so it blows up like this.
3. In practice, training with `−1e−5` the terrain curriculum didn't rise (`PAPER.md` §2).

**The lesson is neither "the paper is right" nor "the code is right".** When the two differ, **measure which one is right**.
Here, just measuring the term's magnitude side by side with the other terms gave the answer.
The assumption that copying the paper's table as-is will reproduce the results broke several times in this project.
