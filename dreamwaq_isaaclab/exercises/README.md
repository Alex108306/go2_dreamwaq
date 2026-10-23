# Exercises

A 5-stage set of exercises in which you **build DreamWaQ up yourself** on top of Isaac Sim / Isaac Lab.
The finished version lives in this repo's `dreamwaq_manager/` · `dreamwaq_direct/`, and the exercises were made by
cutting that code out stage by stage.

## Exercise intensity — 4 levels

Each exercise carries a level label. Check what is being asked of you before you start.

| Level | Name | What you do |
|:---:|---|---|
| **L0** | READ | Run it and just check the output. Only read the code |
| **L1** | FILL | Fill in `TODO` blanks of 1–5 lines. The skeleton is given |
| **L2** | BUILD | Only the signature, docstring and tests are given. You write the whole body |
| **L3** | DESIGN | Only the requirements are given. You write the whole file |

Don't spend time on L0. What matters is concentrated in L2 and L3.

## Verification has two tiers

Iteration speed determines the learning experience. Isaac Sim takes 25 seconds just to start,
so every check that can run without Isaac Sim was made to do so.

| | Command | Time | What it checks |
|---|---|---|---|
| **Fast check** | `python check.py` | ~1 s | shapes · formulas · boundary conditions. Uses synthetic tensors only |
| **Slow check** | `quick_test.sh` / `python solution.py` | ~40 s+ | Whether the real env comes up and runs a few iterations |

Pass the fast check first, then run the slow check once at the end.

The slow check is not a per-exercise script; it **differs by stage.**

- **Stage 1** — each `solution.py` / `starter.py` launches Isaac Sim itself and prints `[PASS]`
- **Stage 2·4** — these modify project modules, so use `dreamwaq_manager/scripts/quick_test.sh` (64 envs / 30 iters)

> **The `starter/` you fill in does not go into training.** The `starter/` in Stages 2·4 is a **read/practice copy**
> cut from the production source, and what `train.py` / `quick_test.sh` import is the
> original in the installed package (`dreamwaq_manager/source/.../algorithms/cenet.py`, etc.). So the slow
> check verifies **that the production source runs, not your answer**. If you want to run real training with your own answer,
> replace **only the corresponding body** in the production source with yours — overwriting it with the whole starter file
> also wipes out what was hidden when making the exercise (the `DWQ_CENET_*` gates, latent diagnostic logging, etc.).
> To revert, there is `git checkout`.
- **Stage 3** — `python starter.py` actually trains on a toy environment (40 s on CPU)

> Checkers are run with the **bundled kit python** — the system `python3` doesn't have torch.
> ```bash
> ~/IsaacLab/_isaac_sim/python.sh check.py
> ```

## Stage layout

| Stage | Topic | Format | Exercises | Composition |
|:---:|---|---|:---:|---|
| **1** | Exploring Isaac Sim / Lab | Standalone scripts | 5 | L0 ×1 · L1 ×3 · L2 ×1 |
| **2** | Training Env class (Manager vs Direct) | Partially completed project modules | 6 | L1 ×2 · L2 ×4 |
| **3** | Training algorithm PPO | Standalone script + code reading | 4 | L0 ×1 · L1 ×1 · L2 ×2 |
| **4** | Terrain-context learning CENet (VAE) | Standalone → project integration | 4 | **L2 ×3 · L3 ×1** |
| **5** | Comparing inference experiment results | Analysis of distributed assets | 4 | L0 ×1 · L2 ×2 · written ×1 |

See the full list with `python exercises/tools/make_exercise.py --list`.

### Stage 1 — Exploring Isaac Sim / Lab

Empty scene → spawn Go2 → joint PD control → terrain generation → reading sensor (contact / raycaster) values.

The first four are L0–L1. **Only the last one, "reading sensor values", is raised to L2** — because it is the exact skill
you use when building observations in Stage 2.

### Stage 2 — Training Env class

What sets this course apart is implementing **the same functionality twice, with two APIs**.

| Target | Manager style | Direct style | Level |
|---|---|---|:---:|
| Observations | `ObsGroup` + `ObsTerm` | `_get_observations` inline | L2 |
| Rewards | `rewards.py` + `RewTerm` weights | `_get_rewards` inline | L2 |
| Termination | `DoneTerm(illegal_contact)` | `_get_dones` | **L2** |
| Reset / randomization | `EventTerm` | `_reset_idx` | L0 |

You are not asked to write `velocity_env_cfg.py` (442 lines) and `dreamwaq_env.py` (834 lines) wholesale.
**Only 4 rewards + Direct observation scaling + Direct termination** are carved out; the rest is provided.
On the Manager side, merely importing `ObsGroup`/`DoneTerm` starts the kit kernel, so no fast check can be attached —
they are replaced by reading (L0). For the detailed reason see `stage2_env/README.md`.

### Stage 3 — PPO

This project uses `rsl_rl`'s PPO as-is. So it is covered **separately**.

1. **Implement a minimal PPO yourself** (L2/L3) — a standalone script that runs without Isaac Sim.
   GAE, clipped surrogate, value loss and entropy are each a blank
2. **Read `rsl_rl`'s PPO and map it** (L0) — "what I wrote is this class right here"
3. **cfg exercise** (L1) — the meaning of `RslRlPpoAlgorithmCfg` values, `desired_kl` adaptive lr,
   and this repo's real-world issue, the log_std clamp

### Stage 4 — CENet (VAE) ← maximum intensity

This is the part the course is named after. It is built up in 4 steps.

1. **Plain VAE** `task01_vae` (L3) — toy data. encode / reparameterize / decode / ELBO
2. **Extend to the CENet structure** `cenet-forward` (L2) — `obs_history(225)` → `est_vel(3)` + `context(16)`
3. **The three loss terms** `cenet-loss` (L2) — vel / recon / KL, matching reduction axes and the effective β
4. **Runner integration** `runner-augment` (L2) — augmenting actor observation 45 → 64, rollout storage timing,
   where `update()` is called

The toy VAE and CENet differ in only two places — an `est_vel` branch is attached, and the reconstruction target is not itself
but **the next step's observation**. For details see `stage4_cenet/README.md`.

### Stage 5 — Comparing inference experiment results

**Premise: learners do not run full training.** The distributed assets were trained with 4096 envs × **4000 iter**;
one rough run takes about 2 h 50 min, and 6 runs take over 10 hours. Instead, **the checkpoints · videos · tensorboard logs
of the 6 runs the instructor ran in advance are used as distributed assets**.

Learners only confirm that their own implementation "runs" at `quick_test` scale (64 envs / 30 iter),
and interpret results using the distributed assets.

- Compare Base / Oracle / Waq checkpoints with `play.py` + videos (L0)
- Extract curves from tensorboard logs into a 3-way comparison plot (L2)
- Interpret **Waq − Base = pure contribution of CENet**, **Oracle − Waq = remaining gap** (written)
- When results flip, diagnose CENet collapse via **checkpoint forensics** (L2)

Measured conclusion: on rough the paper ordering held, `Base(0.5492) < Waq(0.5565) < Oracle(0.5822)`,
while on flat `Waq−Base` is indistinguishable from noise (Base is already near the ceiling at 0.938).
**Both are teaching material** — why this happens is the question of Stage 5.

## Directory conventions

```
exercises/
├── README.md              # this document
├── tools/
│   └── make_exercise.py   # solution → starter generator
├── stage1_isaac_basics/
├── stage2_env/
├── stage3_ppo/
├── stage4_cenet/
└── stage5_compare/
```

Each exercise folder contains the following.

```
taskNN_<name>/               # standalone exercise (Stage 1·3, task01_vae)
├── README.md      # goal · level · hints · pass criteria
├── solution.py    # finished version (contains markers)
├── starter.py     # file the learner fills in (generated — do not edit directly)
└── check.py       # fast check (none for L0 exercises)

<id>/                        # project module exercise (Stage 2·4)
├── README.md
├── starter/<original_filename>.py   # generated — a copy of the project source with TODO holes punched in
└── check.py
```

## Starters are not made by hand

`starter.py` is **always generated from the solution**. Maintaining two copies by hand inevitably drifts.

```bash
python exercises/tools/make_exercise.py --list           # list all exercises
python exercises/tools/make_exercise.py --id cenet-loss  # generate a starter
python exercises/tools/make_exercise.py --all --check    # check they are up to date (for CI)
```

There are two ways to declare them, split by **whether the solution is production code or course-only code**.

### 1. Spec files — project modules (Stage 2·4)

**Not a single character of the `dreamwaq_manager/` · `dreamwaq_direct/` sources is touched.**
Exercise definitions live in `exercises/specs/<id>.toml`, and `body` contains the section to cut out verbatim.

```toml
id     = "cenet-loss"
level  = 2
stage  = "stage4_cenet"
source = "dreamwaq_manager/.../cenet.py"
task   = "Complete CENet's three loss terms"
hints  = ["...", "..."]
body   = '''            mse_loss = nn.MSELoss()
...
'''
```

The generator **reads the actual source**, finds `body` and replaces it with a TODO. If the source changes and `body`
can't be found, it **fails loudly** — you never get a silently drifted starter.
When that happens, just update `body` to the current code.

### 2. Inline markers — standalone exercises (Stage 1·3)

These files are course material to begin with, so nothing is lost by planting markers in them. Markers are comments, so
`solution.py` runs as-is.

```python
# ex:begin id=pd-target level=1 stage=stage1_isaac_basics task=compute the joint target
#   hint: joint_target = default_joint_pos + action * ACTION_SCALE
return robot.data.default_joint_pos + action * ACTION_SCALE
# ex:end
```

`solution.py` and `starter.py` sit side by side in the exercise folder.
