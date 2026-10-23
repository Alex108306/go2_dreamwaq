# dreamwaq_manager (IsaacLab — ManagerBasedRLEnv)

> Independent implementation of [DreamWaQ](https://arxiv.org/abs/2301.10602), ported from the original IsaacGym code ([upstream `dreamwaq/`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/tree/main/dreamwaq)) to **Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0)** using the `ManagerBasedRLEnv` API. Robot platform: **Unitree Go2**.

This document is written in parallel with the *Main Code Structure* section of the original IsaacGym README ([upstream `dreamwaq/README.md`](https://github.com/curieuxjy/IsaacLab_DreamWaQ/blob/main/dreamwaq/README.md)), and explains where each component of the original IsaacGym project was mapped in the IsaacLab `ManagerBasedRLEnv` layout. It is also the ManagerBased counterpart to [`../dreamwaq_direct/README.md`](../dreamwaq_direct/README.md); the two packages share the same reward / observation / domain randomization semantics.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | Install / train / inference commands |
| [Task Options](#task-options) | Base / Oracle / Waq options |
| [Structure Mapping](#structure-mapping-isaacgym--isaaclab-managerbasedrlenv) | Original ↔ Manager mapping table |
| [File-by-file correspondence](#file-by-file-correspondence) | Per-file correspondence |
| [Manager API mapping](#manager-api-mapping) | How the Manager classes absorb the original code |
| [Key Implementation Details](#key-implementation-details) | Key decisions when porting to ManagerBasedRLEnv |
| [Direct vs Manager](#direct-vs-manager-comparison-with-dreamwaq_direct) | Comparison |
| [See Also](#see-also) | Related document links |

---

## Quick Start

### Installation

```bash
cd dreamwaq_manager
pip install -e source/dreamwaq_manager
```

IsaacLab (release/3.0.0-beta2) + Isaac Sim 6.0 must be installed beforehand. For the full environment setup procedure, see [`../setup.qmd`](../setup.qmd).

### Training

```bash
cd scripts
python train.py --task=DreamWaQ-Manager-Go2-Base-v0   --headless
python train.py --task=DreamWaQ-Manager-Go2-Oracle-v0 --headless
python train.py --task=DreamWaQ-Manager-Go2-Waq-v0    --headless
```

Convenience scripts:
- `run_base.sh`, `run_oracle.sh`, `run_waq.sh` — train a single task
- `run_all_trainings.sh` — run all three tasks sequentially (default 4096 envs / 100k iters / wandb)
- `quick_test.sh` — 64 envs / 30 iters smoke test
- `watch.{py,sh}` — monitor training progress

### Play

```bash
python play.py --task=DreamWaQ-Manager-Go2-Waq-Play-v0 --load_run=FOLDER --checkpoint=N
```

Checkpoints are saved to `logs/rsl_rl/[experiment_name]/[timestamp]/`.

---

## Task Options

| Task ID | Actor obs | Critic obs (state) | Memo |
|---------|:---------:|:------------------:|------|
| `DreamWaQ-Manager-Go2-Base-v0`   | 45      | (none, symmetric AC) | blind |
| `DreamWaQ-Manager-Go2-Oracle-v0` | 48      | 238 | includes true `lin_vel` + privileged height scan |
| `DreamWaQ-Manager-Go2-Waq-v0`    | 45 → 64 | 235 | CENet (`est_vel + context`) + privileged |

Each task has a separately registered `-Play-v0` variant for evaluation/visualization, with `num_envs=50`, `terrain.curriculum=False`, observation noise/push disabled, and debug viz on.

Original ↔ Manager task ID mapping:

| Original (`--task=...`) | ManagerBasedRLEnv (`--task=...`) |
|---|---|
| `a1_base`   | `DreamWaQ-Manager-Go2-Base-v0`   (+ `-Play-v0`) |
| `a1_oracle` | `DreamWaQ-Manager-Go2-Oracle-v0` (+ `-Play-v0`) |
| `a1_waq`    | `DreamWaQ-Manager-Go2-Waq-v0`    (+ `-Play-v0`) |

---

## Structure Mapping (IsaacGym → IsaacLab ManagerBasedRLEnv)

The original `dreamwaq/` is split into two libraries:
- `legged_gym/` — simulation environment (env / config / scripts / utils / resources)
- `rsl_rl/`    — RL algorithms (PPO / actor-critic / runner / VAE)

In the port to ManagerBasedRLEnv, **the simulator plumbing / PPO training loop / actor-critic modules / robot URDF assets are provided by IsaacLab**, and in addition IsaacLab's Manager abstractions (`ObservationManager`, `RewardManager`, `EventManager`, `CommandManager`, `CurriculumManager`, `TerminationManager`) are used heavily to decompose the original `LeggedRobot` class into a composition of cfg classes.

### Top-level package layout

```
dreamwaq_manager                          # External project root (Isaac Lab template layout)
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
└── source/dreamwaq_manager/              # Installable extension: pip install -e source/dreamwaq_manager
    ├── config/extension.toml             # Kit extension manifest + deps
    ├── docs/CHANGELOG.rst
    ├── pyproject.toml                    # build-system only
    ├── setup.py                          # Reads extension.toml + find_packages()
    ├── README.md
    └── dreamwaq_manager/
        ├── __init__.py
        ├── algorithms/                   # Corresponds to rsl_rl/rsl_rl/ (only the needed parts)
        │   ├── cenet.py                  # ← rsl_rl/.../vae/cenet.py
        │   ├── estnet.py                 # ← rsl_rl/.../vae/estnet.py
        │   └── dreamwaq_runner.py        # ← rsl_rl/.../runners/on_policy_runner.py
        │                                 #   (only OnPolicyRunnerWAQ extracted)
        │
        ├── tasks/locomotion/             # Corresponds to legged_gym/legged_gym/envs/
        │   ├── velocity_env_cfg.py       # ← envs/base/legged_robot.py + legged_robot_config.py
        │   │                             #   (Scene/Cmd/Action/Obs/Event/Reward/Term/Curriculum
        │   │                             #    decomposed into 7 Cfg classes)
        │   ├── mdp/
        │   │   ├── __init__.py           # Exposes both IsaacLab's default mdp and this package's custom terms
        │   │   ├── rewards.py            # ← legged_robot.py:_reward_*  (4 custom)
        │   │   ├── events.py             # ← legged_robot.py:_push_robots (disturb tracking)
        │   │   └── observations.py       # ← disturb_force slot of legged_robot.py:compute_observations
        │   └── config/go2/
        │       ├── __init__.py           # ← envs/__init__.py (gym.register)
        │       ├── go2_base_cfg.py       # ← envs/go2/go2_config.py:Go2RoughBaseCfg
        │       ├── go2_oracle_cfg.py     # ← envs/go2/go2_config.py:Go2RoughOracleCfg
        │       ├── go2_waq_cfg.py        # ← envs/go2/go2_config.py:Go2RoughWaqCfg
        │       └── agents/
        │           └── rsl_rl_ppo_cfg.py # PPO + CENet runner config
        │
        └── utils/                        # Helper utilities (rerun visualizer, etc.)
```

---

## File-by-file correspondence

### Environment / config

| Original (`dreamwaq/`) | Manager (`dreamwaq_manager/`) | Notes |
|---|---|---|
| `legged_gym/envs/__init__.py` (`task_registry.register`) | `tasks/locomotion/config/go2/__init__.py` | Native gymnasium `gym.register`. `task_registry` removed. |
| `legged_gym/envs/go2/go2_config.py` (`Go2Rough{Base,Oracle,Waq}Cfg`) | `tasks/locomotion/config/go2/go2_{base,oracle,waq}_cfg.py` | Inherits `DreamWaQVelocityRoughEnvCfg`. Go2-specific overrides (PD gains, spawn height, scaled-down terrain). |
| `legged_gym/envs/base/legged_robot.py` (`LeggedRobot`) | `tasks/locomotion/velocity_env_cfg.py` (7 Manager Cfgs) + `mdp/` | The Managers absorb the step loop / reset / reward / obs entirely. Every function inside the original `LeggedRobot.step()` is decomposed into `ObservationManager` / `RewardManager` / `EventManager` calls. |
| `legged_gym/envs/base/legged_robot_config.py` (`LeggedRobotCfg`) | `tasks/locomotion/velocity_env_cfg.py` (`DreamWaQVelocityRoughEnvCfg` + 7 Cfg classes) | Inherits `ManagerBasedRLEnvCfg`. domain_rand → `EventCfg`, rewards.scales → `RewardsCfg`, noise → `Unoise(n_min, n_max)`. |
| `legged_gym/utils/task_registry.py` | (excluded) | Replaced by native gymnasium registration. |
| `legged_gym/utils/terrain.py` (custom Terrain class) | (excluded) | Replaced by `isaaclab.terrains.TerrainImporterCfg` + `ROUGH_TERRAINS_CFG`. |
| `legged_gym/utils/logger.py` (matplotlib) | (excluded) | Uses wandb / tensorboard. |
| `legged_gym/resources/robots/a1/` (URDF + meshes) | (excluded) | Platform changed A1 → Go2. Uses `isaaclab_assets.UNITREE_GO2_CFG` (USD asset). |

### Scripts

| Original | Manager | Notes |
|---|---|---|
| `legged_gym/scripts/train.py` | `scripts/rsl_rl/train.py` | Same log_std clamp `[-5, 2]` monkey-patch (prevents policy std blow-up). |
| `legged_gym/scripts/play.py` | `scripts/rsl_rl/play.py` | |
| `legged_gym/scripts/mini_test.py` | (excluded) | |

### Algorithms (RSL-RL)

| Original (`dreamwaq/rsl_rl/`) | Manager (`dreamwaq_manager/algorithms/`) | Notes |
|---|---|---|
| `rsl_rl/algorithms/ppo.py` | (excluded) | Reuses `isaaclab_rl.rsl_rl` (`RslRlPpoAlgorithmCfg`). |
| `rsl_rl/modules/actor_critic.py` | (excluded) | Uses `RslRlMLPModelCfg` (hidden=[512,256,128], activation=elu). |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunner`) | (excluded — default IsaacLab runner) | For the Base / Oracle models. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerWAQ`) | `algorithms/dreamwaq_runner.py` (`OnPolicyRunnerWaq`) | Waq model only. Integrates the CENet training loop. |
| `rsl_rl/runners/on_policy_runner.py` (`OnPolicyRunnerEst`) | (excluded) | The EstNet comparison model is not used. |
| `rsl_rl/utils/rms.py` (`RunningMeanStd`) | inside `algorithms/cenet.py` | For learning CENet's normal prior. |
| `rsl_rl/vae/cenet.py` | `algorithms/cenet.py` (`CENet`, `CenetRolloutStorage`) | Ported almost as-is. |
| `rsl_rl/vae/estnet.py` | `algorithms/estnet.py` (`EstNet`) | Ported almost as-is. |

Uses the same algorithms/ code as `dreamwaq_direct` — trained checkpoints from the two packages are compatible.

---

## Manager API mapping

This maps how everything that lived in the single original `LeggedRobot` class (step / reset / reward) was decomposed into IsaacLab's 7 Manager Cfgs. All Cfg classes are defined in `velocity_env_cfg.py`.

| Original function / method | Manager Cfg | Location |
|---|---|---|
| `_compute_torques`, action scale | `ActionsCfg.joint_pos` (`mdp.JointPositionActionCfg`) | `velocity_env_cfg.py:ActionsCfg` |
| `_resample_commands` + `_post_physics_step_callback` (heading) | `CommandsCfg.base_velocity` (`mdp.UniformVelocityCommandCfg`) | `velocity_env_cfg.py:CommandsCfg` |
| `compute_observations` (actor obs) | `ObservationsCfg.PolicyCfg` (ObsTerm × 6) | `velocity_env_cfg.py:ObservationsCfg.PolicyCfg` |
| `compute_observations` (privileged) | `ObservationsCfg.CriticCfg` (ObsTerm × 9) | `velocity_env_cfg.py:ObservationsCfg.CriticCfg` |
| `_reward_*` (13) | `RewardsCfg` (RewTerm × 13) | `velocity_env_cfg.py:RewardsCfg` + custom in `mdp/rewards.py` |
| `check_termination` (base contact + timeout) | `TerminationsCfg.{time_out, base_contact}` | `velocity_env_cfg.py:TerminationsCfg` |
| `_update_terrain_curriculum` | `CurriculumCfg.terrain_level` (`mdp.terrain_levels_vel`) | `velocity_env_cfg.py:CurriculumCfg` |
| `_process_rigid_shape_props` (friction) | `EventCfg.physics_material` (`mdp.randomize_rigid_body_material`) | `velocity_env_cfg.py:EventCfg` |
| `_process_rigid_body_props` (mass + CoM) | `EventCfg.{add_base_mass, base_com}` | `velocity_env_cfg.py:EventCfg` |
| `_init_buffers` PD scaling | `EventCfg.randomize_pd_gains` (`mdp.randomize_actuator_gains`) | `velocity_env_cfg.py:EventCfg` |
| `_push_robots` + `disturb_force` tracking | `EventCfg.push_robot` (`mdp.push_robot_with_disturb`) + `mdp.last_disturb_force` ObsTerm | `mdp/events.py`, `mdp/observations.py` |
| `_reset_root_states` | `EventCfg.reset_base` (`mdp.reset_root_state_uniform`) | `velocity_env_cfg.py:EventCfg` |
| `_reset_dofs` | `EventCfg.reset_robot_joints` (`mdp.reset_joints_by_scale`) | `velocity_env_cfg.py:EventCfg` |

### Where custom rewards / events / obs were added

The original's 4 custom rewards and the event/obs for tracking disturb_force do not exist in IsaacLab's defaults, so they were written from scratch:

| Function | File | Original |
|---|---|---|
| `joint_power_l1` | `mdp/rewards.py` | `legged_robot.py:_reward_joint_power` (L1602-1606) |
| `power_distribution_l2` | `mdp/rewards.py` | `legged_robot.py:_reward_power_distribution` (L1608-1611) |
| `foot_clearance_l2` | `mdp/rewards.py` | `legged_robot.py:_reward_foot_clearance` (L1699-1711) |
| `action_smoothness_l2` (jerk) | `mdp/rewards.py` (ManagerTermBase, t-2 buffer) | `legged_robot.py:_reward_smoothness` (L1629-1637) |
| `push_robot_with_disturb` | `mdp/events.py` | `legged_robot.py:_push_robots` (L793-802) |
| `last_disturb_force` | `mdp/observations.py` | `legged_robot.py:355` (`disturb_force` slot in priv obs) |

---

## Key Implementation Details

### The Manager pattern

- **Pros**: reward/event combinations can be changed by editing cfg only (ablation without code changes). High reuse of IsaacLab's default mdp functions.
- **Cons**: hard to compare directly with the original IsaacGym's inline flow. The cfg → manager call cycle adds a layer of indirection.
- **Keeps alignment with `dreamwaq_direct`**: the two implementations are semantically identical in reward / observation / domain randomization. wandb log keys are also identical (`Episode_Reward/track_lin_vel_xy`, etc.).

### Robot / scene

- `UNITREE_GO2_CFG` (USD asset, `isaaclab_assets`) + PD override (`stiffness=20.0`, `damping=0.5`).
- Spawn height `0.42 m` — leaves a margin to avoid collisions with the `boxes` sub-terrain (≤ 0.1 m) + xy ±1 m reset randomization. (Higher than the paper's 0.34 m.)
- Height scanner: `RayCasterCfg` (`/Robot/base` offset z=20 m), `GridPattern` resolution=0.1, size=[1.6, 1.0] → 187 rays.

### Terrain

- `ROUGH_TERRAINS_CFG` (10 levels × 20 types, 6 sub-terrains).
- Scaled down `boxes.grid_height_range` and `random_rough.noise_range/step` to Go2's size.
- `terrain_levels_vel` curriculum (`CurriculumCfg.terrain_level`) — ±1 level on reset based on distance.

### Privileged observation

Built the same way as the original IsaacGym critic input = `obs_buf ⊕ privileged_obs_buf` (concatenated in the runner):

```
[clean actor obs] + disturb_force(3) + heights(187)
Oracle: 48 + 190 = 238
Waq   : 45 + 190 = 235  (lin_vel excluded, critic.base_lin_vel = None in Go2WaqEnvCfg)
```

### Domain randomization (`EventCfg`)

| Term | mode | params | Original |
|---|---|---|---|
| `physics_material` | startup | friction U(0.2, 1.25), 64 buckets | `_process_rigid_shape_props` |
| `add_base_mass` | startup | base mass + U(-1, +2) kg | `_process_rigid_body_props` |
| `base_com` | startup | CoM xyz ± 5 cm, all bodies | `_process_rigid_body_props` |
| `randomize_pd_gains` | startup | stiffness/damping × U(0.9, 1.1) | `_init_buffers` (per-env scaling) |
| `reset_base` | reset | xy ±1 m, vel ±0.5 (lin+ang) | `_reset_root_states` |
| `reset_robot_joints` | reset | pos × U(0.5, 1.5), vel = 0 | `_reset_dofs` |
| `push_robot` | interval (1 s) | xyz lin_vel ±1.0, **disturb tracked** | `_push_robots` |

### Observation noise

Effective values of the original `_get_noise_scale_vec` (= `noise_scales × noise_level × obs_scales`):
- `lin_vel`: ±0.2 (actor obs for Oracle only)
- `ang_vel`: ±0.05
- `gravity`: ±0.05
- `dof_pos`: ±0.01
- `dof_vel`: ±0.075

enable_corruption per variant:
- **Base, Oracle**: `enable_corruption = True` (original default `add_noise = True`)
- **Waq**: `enable_corruption = False` (original `Go2RoughWaqCfg.noise.add_noise = False`)

---

## Direct vs Manager (comparison with `dreamwaq_direct`)

| Item | `dreamwaq_direct` | `dreamwaq_manager` |
|---|---|---|
| Base API | `DirectRLEnv` | `ManagerBasedRLEnv` |
| Main file | `dreamwaq_env.py` (single class, inline) | `velocity_env_cfg.py` (7 Cfg classes) |
| Reward code | All inline (`_get_rewards`) | RewTerm × 13 (IsaacLab defaults + custom) |
| Domain randomization | Inline IsaacGym-style functions (`_randomize_friction`, etc.) | `EventCfg` + manager functions |
| Reset | Calls `_reset_root_states` / `_reset_dofs` directly inside `_reset_idx` | `EventManager` automatically calls reset-mode EventTerms |
| Push event | Interval check inside `step()` | `EventCfg.push_robot` (interval mode) |
| Step loop | `_pre_physics_step` / `_apply_action` / `_get_*` hooks | Called automatically by the Managers |
| Direct comparison with original IsaacGym | Easy (1:1 mapping) | Requires going through the manager abstraction |
| Ablation via cfg alone | Hard (requires code changes) | Easy (enable/disable RewTerm/EventTerm) |

The two packages **should produce the same results** and serve as cross-checks for each other. Differences are analyzed in `comparison.qmd` in the upstream repo.

---

## See Also

- [`../README.md`](../README.md) — overview of this repo
- [`../setup.qmd`](../setup.qmd) — IsaacLab installation / asset download
- [`../dreamwaq_direct/`](../dreamwaq_direct/) — DirectRLEnv implementation of the same algorithm (for cross-checking; termination condition differs)

The following documents are not included in this course repo; they are in the [upstream repo](https://github.com/curieuxjy/IsaacLab_DreamWaQ):

- `index.qmd` — overall project overview (Quarto home)
- `comparison.qmd` — Manager ↔ Direct comparison analysis
- `plan.qmd` — IsaacGym → IsaacLab migration plan
- `report.qmd` — experiment results report
- `dreamwaq/README.md` — original IsaacGym README (read-only reference)
