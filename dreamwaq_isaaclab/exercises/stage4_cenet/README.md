# Stage 4 — Terrain-context learning CENet (VAE)

**This is the stage the course is named after.** The previous three stages were preparation for this.

DreamWaQ solves one problem — **the robot has neither a sensor that measures its base velocity nor a camera
that sees the terrain.** It only has joint angles and an IMU. Yet walking on rough terrain needs both.

CENet's answer: **if you force it to reconstruct "what happens next" from the past observation history (225) alone**,
terrain information seeps into that latent vector on its own. This is what the paper calls
*implicit terrain imagination*. It is never told to learn the terrain directly.

## Built up in 3 levels

```
task01_vae     plain VAE                 (L3)  <- toy data. No Isaac Sim, no robot
cenet-forward  extend to CENet structure (L2)  <- forward of the real cenet.py
cenet-loss     CENet's losses            (L2)  <- update of the real cenet.py
runner-augment wire into the runner      (L2)  <- actor observation 45 -> 64
```

Each step grows directly out of what you wrote in the previous one.

| | task01 (toy) | CENet (real) |
|---|---|---|
| Input | `x` (8) | `obs_history` (225 = 45 × 5 frames) |
| Latent | `mu, logvar` (2) | `mu, logvar` (16) + **`est_vel` (3)** |
| Output | `x_hat` (8) — itself | `est_next_obs` (45) — **the next step** |
| Loss | recon + β·KL | recon + β·KL + **vel** |

There are only two differences.

1. **One more branch, `est_vel`, is attached.** The first 3 dims of the encoder output are used for velocity estimation
   and supervised with the actual base linear velocity. This is the part that "knows its velocity without a velocity sensor".
2. **The reconstruction target is the next step's observation, not itself.** Reconstructing itself
   only learns compression. Reconstructing the next one learns *dynamics*, and the terrain is in there.

## Exercise list

| # | Exercise | Level | Format | Checks |
|:---:|---|:---:|---|:---:|
| 01 | [`task01_vae`](task01_vae/) | **L3** | Standalone script | 20 |
| 02 | [`cenet-forward`](cenet-forward/) | **L2** | Real `cenet.py` | 18 |
| 03 | [`cenet-loss`](cenet-loss/) | **L2** | Real `cenet.py` | 14 |
| 04 | [`runner-augment`](runner-augment/) | **L2** | Real `dreamwaq_runner.py` | 12 |

02–04 were made by **cutting the production source of `dreamwaq_manager/` as-is**
(`exercises/specs/*.toml`). Not a single character of the source is touched.

## Stick to the order

`task01_vae` → `cenet-forward` → `cenet-loss` → `runner-augment`.

If you skip 01 and go to 02, you'll meet the reparameterization trick for the first time inside real code.
It's much faster to write it by hand once on the toy first.

## Checks

All finish within 1 second without Isaac Sim. But **they must be run with the bundled kit python** —
the system `python3` doesn't have torch.

```bash
PY=~/IsaacLab/_isaac_sim/python.sh
cd exercises/stage4_cenet/task01_vae && $PY check.py
cd ../cenet-forward   && $PY check.py
cd ../cenet-loss      && $PY check.py
cd ../runner-augment  && $PY check.py
```

Add `--solution` to check the finished version (production source). Use it when the checker seems off.

If the starter doesn't exist, generate it.

```bash
python3 exercises/tools/make_exercise.py --id vae-core
python3 exercises/tools/make_exercise.py --id cenet-forward
python3 exercises/tools/make_exercise.py --id cenet-loss
python3 exercises/tools/make_exercise.py --id runner-augment
```

## When you're done

The Waq task actually runs. Unlike Base (45), the actor receives 64 dims.

> **What the command below runs is not the `starter/cenet.py` you filled in.** This stage's `starter/` is
> a **read/practice copy** cut from the production source, and what `train.py` imports is the original
> in the installed package (`dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/`).
> So what this command confirms is "**the Waq wiring runs**", not "my answer is right" —
> whether your answer is right is judged by `check.py`.
>
> If you want to run real training with your own answer, replace **only the corresponding function body** in the production source
> with yours. Copying the whole starter file over it also wipes out what was hidden when making the exercise (the `DWQ_CENET_*` experiment
> gates, the latent diagnostic logging `cenet_kl_active_dims` / `cenet_kl_nats` / `cenet_dec_w_*`).
> To revert, there is `git checkout`.

```bash
cd dreamwaq_manager
python scripts/rsl_rl/train.py --task=DreamWaQ-Waq-Official-Rough-PPO-v0 \
    --headless --num_envs=64 --max_iterations=30
```

> **Why `-Official-` shows up for the first time.** The repo has two env recipes. The `DreamWaQ-Manager-Go2-*` recipe,
> which copies the paper's rewards as-is, **was abandoned because it couldn't walk** (78% base-contact terminations),
> and actual training and comparisons are run with the `*-Official-*` family, which uses Isaac Lab's official Go2 rewards.
> The CENet code is identical on both sides, so what you wrote in this stage goes in as-is.
> The comparison table is in §0 of [`PAPER.md`](../../PAPER.md) and the Task section of the root [`README.md`](../../README.md).

**All you're checking here is "it runs", not "performance".** The proper comparison
is done in Stage 5 with the **4000-iteration** outputs the instructor ran in advance.
