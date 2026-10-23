# Stage 5 — Comparing inference experiment results

Check **what the things built in the previous four stages actually produced**.

## Premise: learners do not run full training

The actual training that produced the distributed assets was 4096 envs × **4000 iter**; on an RTX 4080 16GB,
one rough run took **about 2 h 50 min** and one flat run about 33 min. The 6 runs needed for the comparison
**take over 10 hours.**

So learners only check that their own implementation **"runs"** (`quick_test` scale, 64 envs / 30 iter),
and **interpret results using the distributed assets the instructor ran in advance**.

```bash
cd dreamwaq_manager && ./scripts/quick_test.sh     # just whether my implementation runs (64 envs / 30 iters)
```

## Distributed assets

`logs/` and `figures/` are gitignored, so they're not in this repo. The instructor distributes them separately.

```
dreamwaq_manager/logs/
└── rsl_rl/                                  ← the only folder the Stage 5 exercises use
    ├── BaseDwq-Official-Rough-PPO-v0/
    │   └── <timestamp>_DreamWaQ-BaseDwq-Rough-PPO-v0_seed42_envs4096/
    │       ├── model_0.pt … model_3999.pt        9 checkpoints (save_interval=500, 4000 it)
    │       ├── events.out.tfevents.*             tensorboard logs
    │       ├── params/{env,agent}.yaml           settings for reproduction — the values actually applied are here
    │       ├── git/{IsaacLab,go2_dreamwaq}.diff  source diff at training time
    │       └── videos/play/rl-video-step-0.mp4   20-second tracking video (400 steps)
    ├── OracleDwq-Official-Rough-PPO-v0/…        actor 48 (+ true base linear velocity)
    ├── Waq-Official-Rough-PPO-v0/…              actor 64 (+ CENet est_vel·context)
    ├── TerrainOracle-Official-Rough-PPO-v0/…    actor 232 (+ height map) — the oracle the paper talks about
    ├── BaseDwq-Official-Flat-PPO-v0/…
    ├── OracleDwq-Official-Flat-PPO-v0/…
    └── Waq-Official-Flat-PPO-v0/…               all seven runs have the same structure
```

> **There's only `rsl_rl/`.** During development, several ablation/probe experiment folders pile up under `logs/`
> (`_archive_*`, `_cenet_regression`, `_*_probe`, etc.), but they're all **intermediate outputs and
> are deleted at cleanup time.** The exercises only use `rsl_rl/`. The collapsed runs in task04's measurement table also
> disappeared in that cleanup, so **that table is a record the instructor measured at the time, not something to reproduce** —
> the exercises pass without the assets (the actual-checkpoint section isn't graded).

Extract the archive you received **as-is under `dreamwaq_manager/logs/`**. The folder names directly under `rsl_rl/`
(`experiment_name`) are the keys `play.py` and `compare_runs.py` use to find runs,
so don't change them. The timestamp folder names below them are free — the tools pick the most recent one.

If you extracted them elsewhere, just pass the path as an argument.

```bash
cd dreamwaq_manager
~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py --logdir /wherever/you/extracted/rsl_rl
```

`compare_runs.py` rebuilds `figures/` (`curves_{flat,rough}.png`,
`reward_{flat,rough}.png`, `terrain_level_rough.png`, `bar_final.png`, `summary.csv`).
Runs that haven't been trained only produce a warning and are skipped.

## Exercise list

| # | Exercise | Level | What it does |
|:---:|---|:---:|---|
| 01 | [`task01_play_compare`](task01_play_compare/) | **L0** | Compare 3 checkpoints visually (videos) |
| 02 | [`task02_curves`](task02_curves/) | **L2** | Extract curves from tfevents into a comparison plot + summary table |
| 03 | [`task03_interpret`](task03_interpret/) | Written | Interpret the two deltas |
| 04 | [`task04_diagnose`](task04_diagnose/) | **L2** | Use checkpoints to pin down the collapse mechanism of the CENet latent |

task04 **passes even without** the assets above (it's graded on synthetic checkpoints). If the assets are there,
it just finds and prints the archived runs, and you can also give the path directly with `--run`.
The "Measured — the four archived runs" table in the task04 README is reproduced as-is **when you've received all** of the two
`_archive_4000it_*` above and `_archive_3000it_stairheavy`. If you received only some of the three folders,
only those rows won't be printed; grading is unaffected.

## This stage's two-line conclusion

The three were trained with **the same env, the same hyperparameters and the same budget**. Only the actor observation differs.
So the differences can be read as the effect of observation design.

```
Waq − Base     = pure contribution of CENet
                 (how much estimating terrain and velocity helped)

Oracle − Waq   = remaining gap
                 (how far the estimate falls short of true privileged info)
```

If the paper is right, the ordering **Base < Waq < Oracle** should come out.

Measured (seed 42, 4000 it, converged section = mean of the last 400 iterations):

| Terrain | Base (45) | Waq (64) | Oracle (48) | Waq−Base | Oracle−Waq |
|---|---|---|---|---|---|
| **rough** | 0.5492 ± 0.0015 | 0.5565 ± 0.0015 | 0.5822 ± 0.0017 | **+0.0073 ± 0.0021** (significant) | +0.0257 ± 0.0023 |
| **flat** | 0.9383 ± 0.0003 | 0.9390 ± 0.0003 | 0.9460 ± 0.0002 | +0.0007 ± 0.0004 (**noise**) | +0.0070 ± 0.0004 |

On rough the ordering came out; on flat it didn't. **Why is the question of task03.**
`±` measures only **fluctuation within one run** and does not include variation across seeds — it's a single seed.

## Metric: underlying tracking

The total reward (`Train/mean_reward`) is the sum of the reward terms, so it doesn't tell you **what got better**.
So we detach only the velocity-tracking term and divide the weight back out.

```
underlying = Episode_Reward/track_lin_vel_xy_exp ÷ 1.5
           = mean exp(-‖v_cmd − v‖² / 0.25)      ∈ [0, 1]
```

`1.0` means the commanded velocity was followed perfectly. Without dividing out the weight (1.5),
the value shifts whenever the reward design changes and runs can't be compared.

The weight **1.5 is the official recipe's** value (Isaac Lab `UnitreeGo2RoughEnvCfg`). In the DreamWaQ recipe of
`velocity_env_cfg.py` that you read in Stage 2, the same term is `track_lin_vel_xy` with weight **1.0**, so
when analyzing runs of that family you must divide by 1.0 (`PAPER.md` §2). If in doubt, open that run's
`params/env.yaml` and look at `weight` directly.

## Other metrics worth looking at

Measured values for the 3 rough runs are listed together (mean over the same converged section). **They don't all point the same way** —
that's material for task03.

| Tag | Why look at it | Base | Waq | Oracle |
|---|---|---|---|---|
| `Curriculum/terrain_levels` | The better a policy walks, the harder the terrain it gets promoted to | 2.984 | 2.997 | **3.163** |
| `Train/mean_episode_length` | Shorter when it falls. The closer to 1000, the less it falls | 936.5 | **918.7** | 950.0 |
| `Episode_Termination/base_contact` | Rate of base-contact terminations (lower is better) | 0.093 | **0.123** | 0.068 |
| `Metrics/base_velocity/error_vel_xy` | The velocity error itself (lower is better) | 0.423 | 0.410 | 0.405 |
