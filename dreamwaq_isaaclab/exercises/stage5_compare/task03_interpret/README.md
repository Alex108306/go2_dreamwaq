# task03_interpret — Interpret the two deltas  `Written`

**Stage 5 · Comparing results**

## Goal

Using the numbers from task02, write **what CENet did**. There's no code.

Fair warning: **the results aren't clean.** On rough the paper ordering came out; on flat
it didn't. One of the side metrics even points the opposite way. **How to read that** is this assignment.

## Why this comparison holds

Check this first. The three runs are **identical in all of the following.**

- Same env (`WaqOfficialRoughEnvCfg` / `OracleOfficialRoughEnvCfg` — the official recipe)
- Same rewards, same terrain curriculum, same reset randomization
- Same PPO hyperparameters (`[512,256,128]`, clip 0.2, desired_kl 0.01, ...)
- Same budget (**4000 iter** × 4096 envs), same seed (42)
- All three critics get the privileged observation (including height_scan on rough)

The **only** difference is **the actor observation**.

| | actor observation | What it knows |
|---|---|---|
| Base | 45 | Only its own joints and IMU |
| Waq | 64 = 45 + est_vel(3) + context(16) | The velocity and terrain context **estimated** by CENet |
| Oracle | 48 = 45 + actual base_lin_vel(3) | **Just receives** the actual velocity |

Only with this control can the differences be read as the effect of observation design. If even one thing differed,
you couldn't say "thanks to CENet" — that's a confound.

## Measured results

The screen output of `figures/summary.csv` and `compare_runs.py`.
(converged section = mean of the last 400 iterations, ± is the standard error of the mean over that section)

| Terrain | Base (45) | Waq (64) | Oracle (48) | Waq−Base | Oracle−Waq |
|---|---|---|---|---|---|
| **rough** | 0.5492 ± 0.0015 | 0.5565 ± 0.0015 | 0.5822 ± 0.0017 | **+0.0073 ± 0.0021** (significant) | +0.0257 ± 0.0023 |
| **flat** | 0.9383 ± 0.0003 | 0.9390 ± 0.0003 | 0.9460 ± 0.0002 | +0.0007 ± 0.0004 (**noise**) | +0.0070 ± 0.0004 |

Side metrics (rough, same section):

| Tag | Base | Waq | Oracle |
|---|---|---|---|
| `Curriculum/terrain_levels` | 2.984 | 2.997 | **3.163** |
| `Train/mean_episode_length` | 936.5 | **918.7** | 950.0 |
| `Episode_Termination/base_contact` | 0.093 | **0.123** | 0.068 |
| `Metrics/base_velocity/error_vel_xy` | 0.423 | 0.410 | 0.405 |

And **what's visible only on flat** — exploration noise (`Policy/mean_std`, mean over the converged section):

| Terrain | Base | Waq | Oracle |
|---|---|---|---|
| rough | 0.9999 | 0.9999 | 0.9999 | ← all three are stuck 100% at the upper bound σ=1 |
| flat | 0.973 | **0.767** | 0.948 | ← only Waq is distinctly lower |

## What to write — four questions

### 1. Measure the **headroom** of the two terrains first

`Waq−Base` is +0.0073 on rough and +0.0007 on flat. A tenfold difference.
Before jumping to "CENet is useless on flat", **compute the following yourself**.

- What is each terrain's **total headroom** `Oracle − Base`? (for each of the two terrains)
- What % of that headroom is `Waq−Base`?
- What score is flat's Base already at? What is the upper bound of this metric?

Then answer.

- Is flat's small `Waq−Base` because **CENet didn't do its job**, or because
  **there was no room to measure in the first place**? What evidence distinguishes the two explanations?
- If `Oracle−Base` is also small on flat, does that mean "velocity information itself
  doesn't help much on flat"? If so, what is it right to expect from CENet?
- Is this result **in the same direction as, or opposite to,** the paper's claim (that **terrain imagination** is the key)?
- On flat, **only Waq's exploration noise is especially low** (0.767 vs Base 0.973). The tracking score is the same, but
  it means the policy moved with much more **confidence**. Can this be used as a counterexample to "CENet did nothing"?
  If so, under what conditions; if not, why not?
  (Something to think about: is a low std good? What might it have sacrificed?)

> A common mistake here flips the conclusion. Comparing the two terrains' deltas **in absolute terms**
> erases the fact that one of them is stuck at the ceiling. Fix a denominator, then compare.

### 2. `Oracle − Waq` = remaining gap. Can it be called "close"?

Oracle is the **upper bound** that gets the actual velocity for free.

- What % of the total headroom did CENet recover on rough? (you already computed this in 1)
- Seeing that number, can you write "Waq came close to Oracle"? **If not, why not?**
- Waq receives the 3 velocity dims **plus 16 more context dims**. In terms of information alone, it has more than Oracle.
  Yet it can't win. Is the problem the **quantity** or the **quality** of the information?
  How could you check this? (Hint: `scripts/rsl_rl/collect_velocity.py` extracts the estimated and actual velocity
  trajectories side by side. `Loss/cenet_vel` is also in tensorboard)

### 3. The side metrics **disagree**. How will you write about this?

On rough, Waq tracks velocity better than Base, but

- `base_contact` is **higher** (0.123 vs 0.093 — it falls more often)
- `mean_episode_length` is **shorter** (918.7 vs 936.5)
- `terrain_levels` is about the same (2.997 vs 2.984)

Oracle, on the other hand, is better than Base on **all four metrics**.

- Does this table **support or limit** the claim "CENet helps"?
- Come up with two or more hypotheses explaining the combination of better velocity tracking and more falls.
  (e.g. it walks more aggressively / it reacts dangerously when the estimated velocity is wrong / it's just noise)
- What **experiment would distinguish** those hypotheses? Is there anything you can do with the assets you have now?
- Is the paper's claim about "velocity tracking" or "stability"? Which metric is the right one to judge by?

**Leaving this item out costs points.** Cherry-picking favorable metrics is exactly what this course guards against.

### 4. State the limitations yourself

The following are **all constraints that actually apply to this result**. Pick the relevant ones and,
for each, write one line on **in which direction** it could sway the conclusion.

| Constraint | Measured evidence |
|---|---|
| **Single seed** | Only seed 42. `±` measures only fluctuation **within** one run, not variation across seeds |
| **Exploration noise at the ceiling** | For all three rough runs, `Policy/mean_std` over the converged section is **0.9999** — **100%** of the last 400 iterations were spent at the upper bound σ=1. It's the gain from a 3-dim velocity estimate measured while actions were drowned in maximum noise (flat isn't like that — 0.77–0.97) |
| **AdaBoot is time-based** | AdaBoot is a mechanism that raises, during training, the probability `p_boot` of feeding the actor the **estimated velocity** (with the remaining probability it feeds the true velocity — `boot_mask` in the `runner-augment` exercise). The paper sets it by the coefficient of variation of the reward (`p_boot = 1 − tanh(CV(R))`, eq.8), so it automatically reverts to the true velocity if the estimator breaks, but the implementation is a **linear ramp** and doesn't revert (`Loss/boot_prob` fixed at 0.9 from it 2000) (PAPER.md §5) |
| **Missing rewards and randomization** | 5 of the 12 terms in the paper's Table I aren't in the actual training, and much of Table II's domain randomization is missing too (PAPER.md §2·§6) |
| **Different robot** | The paper uses A1, here it's Go2. Absolute numbers can't be compared directly with the paper |

**Writing down limitations earns points, not loses them.** Not writing them loses points.

## If the ordering didn't come out

On flat it actually didn't. And **in an earlier run of this project it didn't come out on rough either**
(`Waq − Base = −0.0166`; CENet was actually harmful). In that case, the order of checks is this.

| Suspicion | How to check |
|---|---|
| The difference is buried in noise | Enlarge the averaging window and compare with the standard error (task02). If the delta exceeds `Oracle−Base`, it doesn't make sense in the first place |
| No room to measure | Is `Oracle−Base` itself small? Flat is this case |
| Insufficient budget | Is the curve still rising, or has it flattened? (`curves_rough.png`) |
| **The estimator died** | Feed random inputs into the checkpoint's encoder and see whether the output moves → [`task04_diagnose`](../task04_diagnose/) |
| The estimate is inaccurate | The `Loss/cenet_vel` curve, and the estimated vs actual trajectories from `scripts/rsl_rl/collect_velocity.py` |
| Seed luck | 1–2 more runs with different seeds (expensive — low priority) |

`Loss/cenet_{total,vel,recon,kl,beta}` and `Loss/boot_prob` are in the tensorboard of the distributed runs.
They are **not** in the archived earlier run — that logging was added later, so back then
checkpoint forensics was all that could be done. **What you can't diagnose, you can't fix.**

## Length and grading

One A4 page. One table + three or four paragraphs.

| Item | |
|---|---|
| Were the numbers transcribed accurately? | With significant figures and ± |
| Were the two headrooms compared **normalized by a denominator**? | The core of question 1 |
| Was the recovery rate written honestly? | Write the computed % as-is. "Close to Oracle" only when that % supports it |
| Were the disagreeing side metrics **not hidden**? | Question 3 |
| Were the limitations stated by yourself? | Question 4. **Bonus points** |
| Was the mechanism explained? | Writing only "what came out" without "why" loses points |

**Bad answer**: "Waq is 0.0073 higher than Base. CENet is effective."

**Good answer**: state the numbers → judge significance against the standard error → normalize by the two terrains' headroom
to explain the mechanism → disclose that the side metrics disagree → don't exaggerate
the recovery rate → write down the remaining uncertainty yourself.

## Next

- [`task04_diagnose`](../task04_diagnose/) — this result was once flipped. How was the cause found then (L2)
