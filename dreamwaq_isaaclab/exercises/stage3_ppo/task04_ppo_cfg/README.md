# task04_ppo_cfg — The meaning of cfg values and this repo's real-world issues  `L1 · FILL`

**Stage 3 · PPO**

## Goal

Implement yourself the **adaptive learning rate** that task03 passed over as "we don't have it".
And see with your own eyes the problem this repo actually ran into: **log_std blow-up**.

## What to fill in

```
starter.py  →  TODO(ppo-cfg)  inside  adapt_learning_rate()
```

This is the only blank. `clamp_log_std()` is given — just read and run it.

## Why an adaptive learning rate is needed

PPO's clipping keeps the policy from changing too much at once. But **how much it changed**
differs per problem and per training phase. A fixed learning rate is too large early and too small late.

`rsl_rl` measures the KL between the previous and current policy every minibatch and sets its own step size.

```
KL > desired_kl * 2    moved too much     ->  lr / 1.5   (lower bound 1e-5)
KL < desired_kl / 2    moved too little   ->  lr * 1.5   (upper bound 1e-2)
in between             about right        ->  unchanged
```

The **dead zone** in the middle (`desired/2 ~ desired*2`) is the key. Without it, lr oscillates up and down
every step. This repo uses `desired_kl = 0.01`, `schedule = "adaptive"`.

## Checks

```bash
cd exercises/stage3_ppo/task04_ppo_cfg
python check.py            # fast check — no Isaac Sim needed, 1 s
python starter.py          # print cfg values + lr trajectory + log_std blow-up demo
```

Pass criteria (14):

1. If `KL > desired*2` then `lr / 1.5`
2. If `KL < desired/2` then `lr * 1.5`
3. **In the dead zone (boundaries included) lr stays the same** — five values `0.005, 0.008, 0.01, 0.015, 0.02`
4. Doesn't exceed the lower bound `1e-5` or the upper bound `1e-2`
5. **Doesn't raise it if `KL == 0`** ← `rsl_rl`'s `kl_mean > 0.0` condition. Easy to miss
6. Changing `desired_kl` moves the boundaries along with it (hard-coding gets caught)
7. The trajectory of successive applications matches the `rsl_rl` rule
8. Returns the value without side effects
9. `clamp_log_std` regression check

## Hints

<details>
<summary>Watch out for two boundary conditions</summary>

- Is the inequality `>` or `>=` — `rsl_rl` uses `kl_mean > desired_kl * 2.0`.
  `KL == 0.02` is in the **dead zone**, not the lowering zone.
- Without the `kl_mean > 0.0` condition, even when KL is exactly 0 it is judged as "moved too little"
  and lr keeps growing. That causes lr to blow up after the policy has fully converged.
</details>

<details>
<summary>Lower / upper bounds</summary>

`max(1e-5, lr / 1.5)` and `min(1e-2, lr * 1.5)`. Don't mix up the directions —
**when lowering, `max` blocks the floor; when raising, `min` blocks the ceiling.**
</details>

## log_std clamp — a problem this repo actually ran into

Run `starter.py` and you'll see it yourself in section 3.

The actor's action distribution has a learnable `log_std`. When the adaptive lr and a narrow observation distribution
coincide, this value drifts to extremes during training. Once it becomes `±inf`, `Normal` sampling spits out `NaN`
and the whole policy dies — **and that's after hours of training.**

So `train.py` hooks the optimizer and bounds `log_std` to `[-5, 0]` after every step.

```
dreamwaq_manager/scripts/rsl_rl/train.py
  _register_std_clamp_for_all_runners()   <- optimizer.register_step_post_hook
```

The reason the upper bound is `0.0` was originally "since actions are clipped to `[-1, 1]` (`clip_actions = 1.0`),
`std > 1` mostly gets cut off when sampled and doesn't help exploration".
**Now `clip_actions = 4.0`, so that premise is gone** — because that ±1 clip was actually the **cause** of `std`
sticking to the upper bound (log-probability and entropy are computed on the Gaussian **before** clipping,
so the extra noise of `σ>1` is free while the entropy bonus keeps accruing → there's no equilibrium below the upper bound.
`PAPER.md` §6). With clip 4.0 an equilibrium appears inside at `σ*≈0.374`, and
the clamp remains only as a **NaN-prevention safeguard** that no longer binds.

> The same monkey patch is in the `train.py` of both `dreamwaq_manager` and `dreamwaq_direct`.

## The cfg values are the paper's values

Section 1 of `starter.py` **reads the values directly** from `rsl_rl_ppo_cfg.py` and prints them.
`clip=0.2`, `γ=0.99`, `λ=0.95`, `lr=1e-3`, `desired_kl=0.01`, `epochs=5`, `minibatches=4` —
They weren't picked arbitrarily; they are the values specified in §III-B of the DreamWaQ paper.

## Next

Stage 3 is complete. Go to [Stage 4 — CENet](../../stage4_cenet/).
