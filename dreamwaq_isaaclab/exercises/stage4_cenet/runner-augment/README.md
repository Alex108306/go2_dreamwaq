# runner-augment — actor observation 45 → 64  `L2 · BUILD`

**Stage 4 · CENet (VAE)**

## Goal

**Actually plug** the values CENet produces **into the actor input.** This one-line concatenation separates Base from Waq.

```
Base   : actor observation 45                                     ← knows neither terrain nor velocity
Oracle : actor observation 48 = 45 + actual base_lin_vel(3)       ← just receives privileged info (upper bound)
Waq    : actor observation 64 = 45 + est_vel(3) + context(16)     ← CENet estimates it
```

`Waq − Base` is **the pure contribution of CENet**, and `Oracle − Waq` is **the remaining gap**.
In Stage 5 you'll check these two values on actual training curves.

## What to fill in

```
starter/dreamwaq_runner.py  →  TODO(runner-augment)  inside  OnPolicyRunnerWaq._build_actor_obs()
```

This method is called every step in the rollout loop (inside `learn()`, right before `self.alg.act()`).

Already given:

| | |
|---|---|
| `self.obs_rms` | **running mean/std normalizer** for the base observation (45) (`rsl_rl.modules.EmpiricalNormalization`). The runner updates its statistics with raw observations every step during rollout |
| `self.true_vel_rms` | The same kind of normalizer but **dedicated to the actual base linear velocity (3)**. Used to normalize CENet's velocity target |
| `self._norm_base(base_obs)` | Normalizes the base observation (45) with `obs_rms` |
| `self._extract_extero(src_obs)` | Detaches only the height_scan part from `src_obs` |
| `self.use_exteroception` | `True` for rough, `False` for flat |

## Where normalization is applied

It is applied **only to the 45 base observations**. `vel_input` and `context_vec` are concatenated as-is without normalization.

Reason: `est_vel` is already learned in the space normalized by `true_vel_rms`,
and `context_vec` stays roughly near the standard normal thanks to the KL regularization. Both are already at the right scale.
Feeding them through `obs_rms` (statistics for 45 dims) again wouldn't even match the number of dims.

## Checks

```bash
cd exercises/stage4_cenet/runner-augment
~/IsaacLab/_isaac_sim/python.sh check.py   # fast check — no Isaac Sim needed, 1 s
```

> The checker is pure torch but **is run with the bundled kit python.** The system `python3` doesn't have torch.

The checker detaches `_build_actor_obs` from the class and calls it with a **fake self**. Launching the whole runner
would need an env, but it's possible because this method only uses three things from self.

Pass criteria (12):

1. `45 + 3 + 16 = 64` dims on flat
2. The order is `[norm(base), vel, context]`
   - the first 45 are the **normalized** base
   - the next 3 are `vel_input` **as-is**
   - the last 16 are `context_vec` **as-is**
3. On rough, height_scan (187) is appended **at the very end** for 251 dims. The first 64 are the same as on flat
4. `NaN` → `0.0`, `+inf` → `10.0`, `-inf` → `-10.0`
5. **Finite large values pass through unchanged** — it must be `nan_to_num`, not `clamp`

## Hints

<details>
<summary>Step 1 — the order is the contract</summary>

The order of `torch.cat` defines the actor's input layout. Training and deployment must use the same order.
Change the order here and the checkpoints become incompatible with the upstream `deploy_sim2sim` stack.

`get_inference_policy()` calls the same method, so fixing it in one place makes training, evaluation and deployment all follow along.
</details>

<details>
<summary>Step 2 — exteroception is conditional</summary>

On rough terrain the critic receives height_scan. If `use_exteroception` is `True`,
append `self._extract_extero(src_obs)` **at the very end**. Don't touch the 64 before it.
</details>

<details>
<summary>Step 3 — why nan_to_num</summary>

If CENet diverges, `NaN` shows up in `est_vel` or `context_vec`. If that goes straight into the actor,
the policy weights become `NaN` wholesale and training dies silently. You only notice hours later.

`torch.nan_to_num(..., nan=0.0, posinf=10.0, neginf=-10.0)` changes **only abnormal values** and
leaves normal values alone. Using `clamp` cuts even healthy large values and distorts the learning signal.

`clamp` appears in step 5 of the rollout timing below, but that's not a contradiction. **The CENet outputs (`est_vel`,
`context_vec`) are already clamped to ±10 by the caller *before* calling `_build_actor_obs`.**
By the time they come into `_build_actor_obs` it's already a complete observation, and cutting the 45-dim base observation
along with it here would distort normal observations. That's why inside here it's `nan_to_num`, not `clamp`.
</details>

## Call timing within the rollout (reading)

This method sits right in the middle of `learn()`'s rollout loop. The order matters.

```
1. take base_obs, true_vel out of the observation
2. obs_rms.update(base_obs)      <- update the normalization statistics with raw values first
3. obs_history = normalized history
4. cenet.before_action(...)      <- forward + store (history, true_vel) in the rollout storage
5. clamp est_vel / context_vec to ±10   <- CENet outputs only. The caller cuts them beforehand
6. _build_actor_obs(...)         <- here. It receives already-cut values, so it doesn't clamp again inside
7. alg.act() -> env.step()
8. cenet.after_action(normalized next_obs)  <- store the recon target, storage.step += 1
```

Pay attention to two things.

- **4 and 8 are a pair.** `before_action` stores `(observation history, actual velocity)`, and `after_action` stores
  `the next step's observation`. Split into before and after stepping the env, they complete one transition.
- **The recon target stored in 8 is `_norm_base(next_base_obs)` — a normalized value.**
  This matches the scale of the encoder input (normalized history) and the decoder target.
  Storing raw values here contaminates `recon_loss` and CENet can't do its job.

`cenet.update()` is called once after the rollout ends, together with the PPO update.

## When you're done

Stage 4 is complete. The Waq task actually runs.

```bash
cd dreamwaq_manager
python scripts/rsl_rl/train.py --task=DreamWaQ-Waq-Official-Rough-PPO-v0 \
    --headless --num_envs=64 --max_iterations=30
```

> **Why `-Official-` and not the `DreamWaQ-Manager-Go2-Waq-v0` seen in the exercises?**
> The repo has two env recipes. The `Manager-Go2-*` recipe, which copies the paper's rewards as-is,
> **was abandoned because it couldn't walk** (78% base-contact terminations), and the actual experiments are run with the `*-Official-*`
> family, which uses Isaac Lab's official Go2 rewards. The CENet code is identical on both sides, so what you fixed here is used as-is.
> The detailed comparison is in §0 of [`PAPER.md`](../../../PAPER.md).

Success if `Learning iteration` is printed along with the 3 CENet losses.
The performance comparison is done in Stage 5 with the instructor's distributed assets.
