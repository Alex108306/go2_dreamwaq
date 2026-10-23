# Stage 2 — Training Env class (Manager vs Direct)

Assemble the ingredients gathered in Stage 1 (joint state · action · terrain · sensors) into a **training environment**.

What sets this course apart is implementing **the same functionality twice, with two APIs**.

| Target | Manager style | Direct style | Level |
|---|---|---|:---:|
| Rewards | `rewards.py` + `RewTerm` weights | `_get_rewards` inline | **L2** |
| Observations | `ObsGroup` + `ObsTerm` | `_get_observations` inline | L2 |
| Termination | `DoneTerm(illegal_contact)` | `_get_dones` | **L2** |
| Reset / randomization | `EventTerm` | `_reset_idx` | L0 |

You are not asked to write `velocity_env_cfg.py` (442 lines) and `dreamwaq_env.py` (834 lines) wholesale.
**Only the core pieces are carved out**; the rest is provided.

## Exercise list

| Exercise | Level | Target | What you learn |
|---|:---:|---|---|
| [`reward-joint-power`](reward-joint-power/) | L1 | `rewards.py` | Energy penalty. The property that zero angular velocity means no penalty |
| [`reward-power-distribution`](reward-power-distribution/) | **L2** | `rewards.py` | A term that penalizes **imbalance**, not the total — guarding against motor overheating on hardware |
| [`reward-smoothness`](reward-smoothness/) | **L2** | `rewards.py` | Action jerk (second-order difference), a reward term that holds state |
| [`reward-foot-clearance`](reward-foot-clearance/) | **L2** | `rewards.py` | Foot height above terrain, the design of velocity weighting |
| [`direct-obs-scale`](direct-obs-scale/) | L1 | `dreamwaq_env.py` (**Direct**) | Observation scaling — porting bug #1 that actually happened |
| [`direct-dones`](direct-dones/) | **L2** | `dreamwaq_env.py` (**Direct**) | Termination conditions. Reducing a 4-dim contact force to `(N,)` |

The recommended order is top to bottom. Warm up with `joint-power`, get the contrast of "total vs imbalance"
in `power-distribution`, then move on to `smoothness`, which holds state.
Do observations (`direct-obs-scale`) and termination (`direct-dones`) after finishing the 4 rewards.

### Why there are no exercises for Manager's observation/termination config

`velocity_env_cfg.py` **starts the Isaac Sim kit kernel just by being imported.** So
no fast check (`check.py`) can be attached. Instead the same concepts are covered with **inline code on the Direct side**
(`direct-obs-scale`, `direct-dones`) — if anything, it's better for learning since the tensors are visible.

Manager's `ObsGroup`/`DoneTerm` declaration style is covered instead by **reading (L0)** `dreamwaq_manager/README.md` and
`velocity_env_cfg.py`.

## The production source is not touched

All Stage 2 exercises use the **spec file approach**. No markers or comments go into the
`dreamwaq_manager/` source. Exercise definitions live in `exercises/specs/*.toml`, and
the generator reads the actual source to build the starter.

```bash
python ../tools/make_exercise.py --id reward-smoothness
```

## Checks run without Isaac Sim

`rewards.py` imports `isaaclab.*`, but the computation the exercises check is pure torch.
`exercises/tools/fake_isaaclab.py` plants stubs that only let the import pass, so checking takes **1 second**.

```bash
cd reward-smoothness
python check.py              # my code
python check.py --solution   # finished version (for checking the checker itself)
```
