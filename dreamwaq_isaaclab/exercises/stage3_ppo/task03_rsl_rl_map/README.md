# task03_rsl_rl_map — What I wrote is this class right here  `L0 · READ`

**Stage 3 · PPO**

## Goal

There's no code to write. **Read.**

In task01·task02 you wrote GAE and the PPO losses yourself. But this project uses `rsl_rl`'s PPO
as-is. If you don't pin down **where** in that library what you wrote lives,
you won't know where to look when training misbehaves later.

Finish task01·task02 first. Reading without having written it yourself leaves nothing behind.

## Run

```bash
cd exercises/stage3_ppo/task03_rsl_rl_map
python solution.py
```

This script doesn't hard-code explanations. It **pulls the source of the installed `rsl_rl` directly with `inspect`**
and prints it side by side with our `ppo.py`. So when the rsl_rl version goes up, the output follows.

## What you'll see

| # | Our `ppo.py` | `rsl_rl` |
|:---:|---|---|
| 1 | `compute_gae` | `PPO.compute_returns` |
| 2 | `ppo_losses` | inside `PPO.update` |
| 3 | (none) | `desired_kl` adaptive learning rate |
| 4 | `train()` loop | `OnPolicyRunner.learn` |

## What to check while reading

<details>
<summary>1. The signs differ, so why is it the same formula</summary>

```
ours    : -min( ratio*A , clip(ratio)*A )
rsl_rl  :  max( -A*ratio , -A*clip(ratio) )
```

`-min(x, y) == max(-x, -y)`. Both just "turn the objective you want to maximize into
a minimization problem by negating it". The paper writes the formula in maximization form,
and the optimizer only minimizes, so it gets flipped once somewhere.
</details>

<details>
<summary>2. clipped value loss — a branch we don't have</summary>

We only used `(value - returns)²`. When `use_clipped_value_loss` is on, `rsl_rl`
also limits the critic prediction to **±clip_param around the value at rollout time**.

This repo's cfg has `use_clipped_value_loss=True`, so **it actually takes this branch.**
It's a mechanism that stops the critic from jumping too far in a single update.
</details>

<details>
<summary>3. Where advantage normalization happens</summary>

We normalized over the whole batch after flattening in `train()`.
`rsl_rl` does it at the end of `compute_returns` (when `normalize_advantage_per_mini_batch` is off).

Only the location differs; the formula is the same. There is, however, an option to normalize **per minibatch**,
in which case the statistics differ per batch and the results differ slightly.
</details>

<details>
<summary>4. learn() is 79 lines but the algorithm is 3 lines</summary>

Most of `OnPolicyRunner.learn` is logging, checkpointing and multi-GPU handling.
The algorithm skeleton is three lines, exactly like our `train()`.

```
alg.act() / alg.process_env_step()   <- rollout
alg.compute_returns(obs)             <- GAE
alg.update()                         <- epoch x minibatch
```

**This fact leads into Stage 4.** This project's `OnPolicyRunnerWaq` overrides exactly this
`learn()` to insert a CENet call right in the middle of the rollout loop.
It didn't change the algorithm; it **changed the observation**.
</details>

## Pass criteria

It's enough for `python solution.py` to end with `[PASS] comparison complete`. There's no code to check,
so there's no `check.py` either.

## Next

- [`task04_ppo_cfg`](../task04_ppo_cfg/) — implement item 3 above (adaptive learning rate) yourself (L1)
