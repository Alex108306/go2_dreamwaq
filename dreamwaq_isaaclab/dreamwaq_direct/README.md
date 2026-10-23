# dreamwaq_direct (IsaacLab — DirectRLEnv)

> The termination condition is identical to the original IsaacGym DreamWaQ — **base contact force > 1 N**
> (`terminate_after_contacts_on = ["base"]`). [`../dreamwaq_manager/`](../dreamwaq_manager/) also
> uses the same criterion, so the numbers from the two stacks are comparable.
> We measured and confirmed that both stacks learn equivalently on generated terrain with 4096 envs. The
> collision filtering bug fixed along the way is described in [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md).

> Independent implementation of [DreamWaQ](https://arxiv.org/abs/2301.10602), ported from the original IsaacGym code ([upstream `dreamwaq/`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/tree/main/dreamwaq)) to **Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0)** using the `DirectRLEnv` API. Robot platform: **Unitree Go2**.

This document is written in parallel with the *Main Code Structure* section of the original IsaacGym README ([upstream `dreamwaq/README.md`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/blob/main/dreamwaq/README.md)), and explains where each component of the original IsaacGym project was mapped in the IsaacLab `DirectRLEnv` layout.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | Install / train / inference commands |
| [Task Options](#task-options) | Base / Oracle / Waq options |
| [Structure Mapping](#structure-mapping-isaacgym--isaaclab-directrlenv) | Original ↔ Direct mapping table |
| [File-by-file correspondence](#file-by-file-correspondence) | Per-file correspondence |
| [Key Implementation Details](#key-implementation-details) | Key decisions when porting to DirectRLEnv |
| [See Also](#see-also) | Related document links |

---

## Quick Start

### Installation

```bash
cd dreamwaq_direct
pip install -e source/dreamwaq_direct
```

IsaacLab (release/3.0.0-beta2) + Isaac Sim 6.0 must be installed beforehand. For the full environment setup procedure, see [`../setup.qmd`](../setup.qmd).

### Training

```bash
cd scripts
python train.py --task=DreamWaQ-Direct-Go2-Base-v0   --headless
python train.py --task=DreamWaQ-Direct-Go2-Oracle-v0 --headless
python train.py --task=DreamWaQ-Direct-Go2-Waq-v0    --headless
```

Convenience scripts:
- `run_base.sh`, `run_oracle.sh`, `run_waq.sh` — train a single task
- `run_all_trainings.sh` — run all three tasks sequentially (default 4096 envs / 5k iters / wandb)
- `quick_test.sh` — 64 envs / 30 iters smoke test
- `watch.{py,sh}` — monitor training progress

### Play

```bash
python play.py --task=DreamWaQ-Direct-Go2-Waq-Play-v0 --load_run=FOLDER --checkpoint=N
```

Checkpoints are saved to `logs/rsl_rl/[experiment_name]/[timestamp]/`.

---

## Task Options

| Task ID | Actor obs | Critic obs (state) | Memo |
|---------|:---------:|:------------------:|------|
| `DreamWaQ-Direct-Go2-Base-v0`   | 45      | 0   | blind, no privileged obs |
| `DreamWaQ-Direct-Go2-Oracle-v0` | 48      | 238 | includes true `lin_vel` + privileged height scan |
| `DreamWaQ-Direct-Go2-Waq-v0`    | 45 → 64 | 238 | CENet (`est_vel + context`) + privileged |

Each task has a separately registered `-Play-v0` variant for evaluation/visualization, with `num_envs=50`, `terrain.curriculum=False`, `obs_noise=False`, `system_delay=False`, and push/disturb disabled.

Original ↔ Direct task ID mapping:

| Original (`--task=...`) | DirectRLEnv (`--task=...`) |
|---|---|
| `a1_base`   | `DreamWaQ-Direct-Go2-Base-v0`   (+ `-Play-v0`) |
| `a1_oracle` | `DreamWaQ-Direct-Go2-Oracle-v0` (+ `-Play-v0`) |
| `a1_waq`    | `DreamWaQ-Direct-Go2-Waq-v0`    (+ `-Play-v0`) |

---

## Structure Mapping (IsaacGym → IsaacLab DirectRLEnv)

The original `dreamwaq/` is split into two libraries:
- `legged_gym/` — simulation environment (env / config / scripts / utils / resources)
- `rsl_rl/`    — RL algorithms (PPO / actor-critic / runner / VAE)

In the port to IsaacLab, **the simulator plumbing / PPO training loop / actor-critic modules / robot URDF assets are provided by IsaacLab**, so `dreamwaq_direct/` contains only the DreamWaQ-specific logic (env logic, CENet, EstNet, custom runner).

### Top-level package layout

```
dreamwaq_direct                           # External project root (Isaac Lab template layout)
├── scripts/
│   ├── list_envs.py / zero_agent.py / random_agent.py   # Canonical IsaacLab helpers
│   ├── run_{base,oracle,waq}.sh          # Per-task wrappers
│   ├── run_all_trainings.sh              # Runs the three tasks sequentially
│   ├── quick_test.sh                     # 64 envs / 30 iters smoke test
│   ├── watch.sh                          # Training monitor
│   └── rsl_rl/                           # Corresponds to legged_gym/legged_gym/scripts/
│       ├── train.py                      # ← legged_gym/.../scripts/train.py
│       ├── play.py                       # ← legged_gym/.../scripts/play.py
│       ├── watch.py
│       └── cli_args.py                   # IsaacLab RSL-RL CLI helper (bundled)
│
└── source/dreamwaq_direct/               # Installable extension: pip install -e source/dreamwaq_direct
    ├── config/extension.toml             # Kit extension manifest + deps
    ├── docs/CHANGELOG.rst
    ├── pyproject.toml                    # build-system only
    ├── setup.py                          # Reads extension.toml + find_packages()
    ├── README.md
    └── dreamwaq_direct/
        ├── __init__.py
        ├── algorithms/                   # Corresponds to rsl_rl/rsl_rl/ (only the needed parts)
        │   ├── cenet.py                  # ← rsl_rl/.../vae/cenet.py
        │   ├── estnet.py                 # ← rsl_rl/.../vae/estnet.py
        │   └── dreamwaq_runner.py        # ← rsl_rl/.../runners/on_policy_runner.py
        │                                 #   (only OnPolicyRunnerWAQ extracted)
        │
        └── tasks/locomotion/             # Corresponds to legged_gym/legged_gym/envs/
            ├── dreamwaq_env.py           # ← envs/base/legged_robot.py
            ├── dreamwaq_env_cfg.py       # ← envs/base/legged_robot_config.py
            └── config/go2/
                ├── __init__.py           # ← envs/__init__.py (gym.register)
                ├── go2_env_cfg.py        # ← envs/go2/go2_config.py
                └── agents/
                    └── rsl_rl_ppo_cfg.py # PPO + CENet runner config
```

---

## File-by-file correspondence

### Environment / config

| Original (`dreamwaq/`) | Direct (`dreamwaq_direct/`) | Notes |
|---|---|---|
| `legged_gym/envs/__init__.py` (`task_registry.register`) | `tasks/locomotion/config/go2/__init__.py` | Native gymnasium `gym.register`. `task_registry` removed. |
| `legged_gym/envs/go2/go2_config.py` (`Go2RoughCfg`) | `tasks/locomotion/config/go2/go2_env_cfg.py` | `Go2{Base,Oracle,Waq}DirectCfg` (+ `_PLAY` variant) — inherits `DreamWaQDirectEnvCfg`. Go2-specific: PD override (stiffness=20.0, damping=0.5), spawn `0.42 m`, scaled-down terrain, etc. |
| `legged_gym/envs/base/legged_robot.py` (`LeggedRobot`) | `tasks/locomotion/dreamwaq_env.py` (`DreamWaQEnv`) | Inherits `DirectRLEnv`. The original single `step()` is split into the IsaacLab API's `_pre_physics_step` / `_apply_action` / `_get_observations` / `_get_rewards` / `_get_dones` / `_reset_idx`. All logic is inline in one file. |
| `legged_gym/envs/base/legged_robot_config.py` (`LeggedRobotCfg`, `LeggedRobotCfgPPO`) | `tasks/locomotion/dreamwaq_env_cfg.py` (`DreamWaQDirectEnvCfg`, `EventCfg`) | Inherits `DirectRLEnvCfg`. Includes reward scales / domain randomization (`EventCfg`) / CENet options (`len_obs_history`, `num_context`, `num_est_vel`) / curriculum, etc. |
| `legged_gym/utils/task_registry.py` | (excluded) | Replaced by native gymnasium registration. |
| `legged_gym/utils/terrain.py` (custom Terrain class) | (excluded) | Replaced by `isaaclab.terrains.TerrainImporterCfg` + `ROUGH_TERRAINS_CFG`. |
| `legged_gym/utils/logger.py` (matplotlib) | (excluded) | Uses wandb / tensorboard. |
| `legged_gym/resources/robots/a1/` (URDF + meshes) | (excluded) | Platform changed A1 → Go2. Uses `isaaclab_assets.UNITREE_GO2_CFG` (USD asset). |

### Scripts

| Original | Direct | Notes |
|---|---|---|
| `legged_gym/scripts/train.py` | `scripts/rsl_rl/train.py` | Same log_std clamp `[-5, 2]` monkey-patch (prevents policy std blow-up). |
| `legged_gym/scripts/play.py` | `scripts/rsl_rl/play.py` | |
| `legged_gym/scripts/mini_test.py` | (excluded) | |

### Algorithms (RSL-RL)

| Original (`dreamwaq/rsl_rl/`) | Direct (`dreamwaq_direct/algorithms/`) | Notes |
|---|---|---|
| `rsl_rl/algorithms/ppo.py` | (excluded) | Reuses `isaaclab_rl.rsl_rl` (`RslRlPpoAlgorithmCfg`). |
| `rsl_rl/modules/actor_critic.py` | (excluded) | Uses `RslRlMLPModelCfg` (hidden=[512,256,128], activation=elu). |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunner`) | (excluded — default IsaacLab runner) | For the Base / Oracle models. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerWAQ`) | `algorithms/dreamwaq_runner.py` (`OnPolicyRunnerWaq`) | Waq model only. Integrates the CENet training loop. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerEst`) | (excluded) | The EstNet comparison model is not used. |
| `rsl_rl/utils/rms.py` (`RunningMeanStd`) | inside `algorithms/cenet.py` | For learning CENet's normal prior. |
| `rsl_rl/vae/cenet.py` | `algorithms/cenet.py` (`CENet`, `CenetRolloutStorage`) | Ported almost as-is. |
| `rsl_rl/vae/estnet.py` | `algorithms/estnet.py` (`EstNet`) | Ported almost as-is. |

### PPO Runner Config (new in Direct)

`tasks/locomotion/config/go2/agents/rsl_rl_ppo_cfg.py` holds the PPO + CENet settings based on `RslRlOnPolicyRunnerCfg`. It restructures fragments of the original `LeggedRobotCfgPPO` into IsaacLab's standard cfg class form.

| Class | Applies to task | `class_name` | `max_iterations` |
|---|---|---|---|
| `Go2BasePPORunnerCfg`   | `Base-v0`   | (default `OnPolicyRunner`) | 1500 |
| `Go2OraclePPORunnerCfg` | `Oracle-v0` | (default `OnPolicyRunner`) | 1500 |
| `Go2WaqPPORunnerCfg`    | `Waq-v0`    | `OnPolicyRunnerWaq` | 5000 |

`Go2WaqPPORunnerCfg` additionally defines the `waq` (training hyperparams) and `vae` (CENet learning rate / β / scheduler) dicts.

---

## Key Implementation Details

### DirectRLEnv porting decisions

- The original single `LeggedRobot.step()` method is split into the 6 hooks enforced by IsaacLab's `DirectRLEnv`:
  - `_pre_physics_step(actions)` — clip + compute PD target
  - `_apply_action()`            — apply joint position targets
  - `_get_observations()`        — body-frame obs / obs noise / system delay / obs history / privileged obs (height scan 187 ray)
  - `_get_rewards()`             — the 13 DreamWaQ rewards (exactly as in the paper)
  - `_get_dones()`               — base contact threshold + timeout
  - `_reset_idx(env_ids)`        — randomized state, terrain curriculum, command resample, episode reward logging
- Unlike the Manager package (`dreamwaq_manager/`), the Manager classes (Reward / Observation / Termination / Curriculum) are not used; **all logic is written inline inside the single `DreamWaQEnv` class**. This keeps a 1:1 correspondence with the original IsaacGym `LeggedRobot`.

### Robot / scene

- `UNITREE_GO2_CFG` (USD asset, `isaaclab_assets`) + PD override (`stiffness=20.0`, `damping=0.5`).
- Spawn height `0.42 m` — leaves a margin to avoid collisions with the `boxes` sub-terrain (≤ 0.1 m) + xy ±0.5 m reset randomization. (Higher than the paper's 0.34 m.)
- Height scanner: `RayCasterCfg` (`/Robot/base` offset z=20 m), `GridPattern` resolution=0.1, size=[1.6, 1.0] → 187 rays.

### Terrain

- `ROUGH_TERRAINS_CFG` (10 levels × 20 types, 6 sub-terrains: pyramid_stairs / pyramid_stairs_inv / boxes / random_rough / hf_pyramid_slope / hf_pyramid_slope_inv).
- Scaled down to Go2's size: `boxes.grid_height_range = (0.025, 0.1)`, `random_rough.noise_range = (0.01, 0.06)`, `noise_step = 0.01`.
- `curriculum = True`, `max_init_terrain_level = 4` — training starts at levels ≤ 4 → on reset, progresses `+1` (distance traveled ≥ commanded distance / 2) / `-1` (≤ 1/4).

### Privileged observation (`state_space = 238`)

```
lin_vel(3) + ang_vel(3) + gravity(3) + commands(3)
+ joint_pos(12) + joint_vel(12) + actions(12)
+ height_scan(187) + disturb_force(3)
```

### CENet input

```
obs_history (5 timesteps × 45 obs = 225)  →  est_vel(3) + context(16)
```

### Domain randomization (`EventCfg`)

`physics_material` (friction 0.2–1.25), `add_base_mass` (-1 ~ +2 kg), `base_com` (±5 cm), `randomize_pd_gains` (×0.9–1.1), `push_robot` (1 Hz, ±1 m/s), `reset_base` (xy ±0.5 m, yaw ±π, vel ±0.5).

### Alignment with the Manager package

Reward / observation / domain randomization / reset logic keeps 1:1 alignment with `dreamwaq_manager/`. For an analysis of the differences, see `comparison.qmd` in the upstream repo.

---

## See Also

- [`../README.md`](../README.md) — overview of this repo
- [`../setup.qmd`](../setup.qmd) — IsaacLab installation / asset download
- [`../dreamwaq_manager/`](../dreamwaq_manager/) — ManagerBasedRLEnv implementation of the same algorithm (active stack)

The following documents are not included in this course repo; they are in the [upstream repo](https://github.com/curieuxjy/IsaacLab_DreamWaQ):

- `index.qmd` — overall project overview (Quarto home)
- `comparison.qmd` — Manager ↔ Direct comparison analysis
- `plan.qmd` — IsaacGym → IsaacLab migration plan
- `report.qmd` — experiment results report
- `dreamwaq/README.md` — original IsaacGym README (read-only reference)
