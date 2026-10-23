# Stage 3 — Training algorithm PPO

This project uses `rsl_rl`'s PPO **as-is.** Nothing is implemented in-house.
So this stage is covered **separately**.

```
1) Implement a minimal PPO yourself      (L2)  <- standalone script that runs without Isaac Sim
2) Read rsl_rl's PPO and map it          (L0)  <- "what I wrote is this class right here"
3) cfg hyperparameter exercise           (L1)
```

## Toy environment — a miniature of locomotion

Instead of an unrelated problem like CartPole, we use the **same "commanded velocity tracking"** problem as DreamWaQ.
Even the reward function has the same form as the paper's Table I.

| toy (`toy_env.py`) | DreamWaQ |
|---|---|
| `v` (1D velocity) | `base_lin_vel` |
| `v_cmd` | `velocity_commands` |
| `a` (acceleration) | joint target offset (`action * 0.25`) |
| `exp{-4 e²}` | `track_lin_vel_xy_exp` (`std=√0.25`) |
| `drag` | ground friction / gravity |

```bash
python toy_env.py
```

```
Perfect tracking reward : 1.0000
Standing-still reward   : 0.4426   <- training fails if it can't beat this
Random policy mean      : 0.5333
```

**`0.4426` is this stage's floor.** It's the same idea as the failure in DreamWaQ where "standing still is optimal"
— if the policy learns nothing, it stays around here.

## Exercise list

| # | Exercise | Level | Check |
|:---:|---|:---:|:---:|
| 01 | [`task01_gae`](task01_gae/) | **L2** | 8 |
| 02 | [`task02_ppo_update`](task02_ppo_update/) | **L2** | 10 |
| 03 | [`task03_rsl_rl_map`](task03_rsl_rl_map/) | **L0** | — |
| 04 | [`task04_ppo_cfg`](task04_ppo_cfg/) | **L1** | 14 |

01·02 fill in different parts of the same file, `ppo.py`. Each `starter.py` has only its own exercise's
part blank and the answer in the rest, so **you can start with either one.**

03 is reading only — it only makes sense after finishing 01·02. 04 fills in the adaptive learning rate
that 03 passed over as "something we don't have". **Stick to the order 01·02 → 03 → 04.**

## Once everything is filled in, training actually runs

```bash
cd task01_gae && python starter.py
```

```
Standing-still reward (floor) : 0.4426

  iter    0 | mean reward 0.4379 | std 1.011
  iter   20 | mean reward 0.8670 | std 0.944
  ...
  iter  199 | mean reward 0.9562 | std 0.727

Final mean reward : 0.9455  (floor 0.4426, perfect tracking 1.0)
Training succeeded
```

It takes about 40 seconds on CPU. Also watch `std` gradually decrease from 1.0 —
that's the policy reducing exploration as it gains confidence. (This connects to the log_std story in Stage 2.)

## Correspondence with the real project

`ppo.py` only shrinks the scale; the skeleton is the same.

| `ppo.py` | Real project |
|---|---|
| `ActorCritic` 64-64 | actor/critic 512-256-128 (`rsl_rl_ppo_cfg.py`) |
| `log_std` parameter | `init_noise_std=1.0` + the log_std clamp in `train.py` |
| `compute_gae` | `rsl_rl` `RolloutStorage.compute_returns` |
| `ppo_losses` | `rsl_rl` `PPO.update` |
| `train()` loop | `OnPolicyRunner.learn()` |
| `clip=0.2, γ=0.99, λ=0.95, lr=1e-3` | **Same as paper §III-B** |

The last row matters — the toy script's PPO hyperparameters weren't picked arbitrarily; they are
the values specified in the paper.
