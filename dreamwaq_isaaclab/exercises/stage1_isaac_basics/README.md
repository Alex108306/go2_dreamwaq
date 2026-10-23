# Stage 1 — Exploring Isaac Sim / Isaac Lab

Before building DreamWaQ, get hands-on with the simulator underneath it.
Everything here is a **standalone script**. Project code isn't touched yet.

## Why this order

The training environment you'll build in Stage 2 is ultimately a combination of these five pieces.

```
scene (sim) + robot (articulation) + action (PD control) + terrain + sensor
   ↓          ↓                    ↓                ↓              ↓
task01     task02              task03           task04         task05
                                                                  │
                    Stage 2's observations, rewards and terminations all come from here ┘
```

## Exercise list

| # | Exercise | Level | What you get |
|:---:|---|:---:|---|
| 01 | [`task01_empty_scene`](task01_empty_scene/) | **L0** | The skeleton of `AppLauncher` → `SimulationContext` → step loop |
| 02 | [`task02_spawn_go2`](task02_spawn_go2/) | **L1** | `Articulation`, joint order, reading state via `data.*` |
| 03 | [`task03_pd_control`](task03_pd_control/) | **L1** | action → joint target → PD torque. The meaning of DreamWaQ's action scale 0.25 |
| 04 | [`task04_terrain`](task04_terrain/) | **L1** | plane vs generator, `env_origins`, the foundation of the terrain curriculum |
| 05 | [`task05_sensors`](task05_sensors/) | **L2** | `ContactSensor` / `RayCaster` — raw material for Stage 2 observations and termination |

Breeze through the first four. **Really dig into 05** — you'll use it as-is in Stage 2.

## How to run

Every script is run with Isaac Sim's bundled python (or a conda env).

```bash
cd exercises/stage1_isaac_basics/task01_empty_scene
python solution.py             # run the answer — see what should come out first
python starter.py              # run my code
python solution.py --viz kit   # view in a GUI window
```

**Headless is the default.** If you want a GUI, pass `--viz kit`. Viewing it in the GUI once
at the start is recommended — you need to see with your own eyes how the robot actually moves to get a feel for it.

> The `--headless` flag seen in older material or in this repo's `train.py --headless`
> is deprecated in the current Isaac Lab. It works, but prints a warning.

Each script **checks itself**. If it runs to the end, `[PASS]` lines are printed;
if something is wrong, it stops right there with an `AssertionError` telling you what differed from expectations.

> Stage 1 checks need Isaac Sim to start, so each takes ~25 seconds.
> From Stage 2 on, most exercises come with a `check.py` that runs in 1 second without Isaac Sim.

## starter is a generated file

You may edit `starter.py` directly, but if you want to revert it you can regenerate it anytime.

```bash
python ../../tools/make_exercise.py --id <exercise id>
```

The answer is in `solution.py` in the same folder. You may open it if you get stuck —
but **wrestle with it yourself for 20 minutes first**.
