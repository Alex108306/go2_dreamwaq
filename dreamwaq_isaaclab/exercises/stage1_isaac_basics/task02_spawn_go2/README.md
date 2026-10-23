# task02 — Spawn Go2 and read its state  `L1 · FILL`

## Goal

Put a robot into the scene and read its state. **The 45-dim observation you'll build in Stage 2
is entirely a combination of the values read here.**

## What to fill in

`TODO(spawn-go2)` in `starter.py` — 3–4 lines.

Copy `UNITREE_GO2_CFG`, set the following, and create an `Articulation`.

| Item | Value |
|---|---|
| `prim_path` | `"/World/Robot"` |
| `init_state.pos` | `(0.0, 0.0, SPAWN_HEIGHT)` |

`SPAWN_HEIGHT = 0.42` is the value this project uses. The paper uses 0.34 m, but
to clear the generated terrain's `boxes` (up to 0.1 m) together with the xy ±0.5 m reset randomization, it needs to be higher.

> Don't forget `.copy()`. `UNITREE_GO2_CFG` is a module-global object, so modifying it directly
> affects other code in the same process.

## Run

```bash
python starter.py              # my code
python solution.py             # the answer
python solution.py --viz kit   # watch it sink in the GUI
```

## Pass criteria

1. 12 joints (4 legs × hip/thigh/calf)
2. A `base` body exists — **the termination condition looks at this body's contact**
3. 4 feet (`*_foot`) — the foot clearance reward uses these bodies
4. The final height is above the ground (it didn't sink below the ground)
5. `projected_gravity_b[2]` is in the range `[-1.05, 0.05]`

## Read and move on

The values printed in the output are the ingredients of the observation.

| Value | Meaning | In the observation |
|---|---|---|
| `root_lin_vel_b` | Body-frame linear velocity | Received directly by **Oracle only**. Base/Waq can't see it |
| `projected_gravity_b` | Body-frame gravity direction | 3 dims. Tells the orientation |
| `joint_pos` / `joint_vel` | Joint angles / angular velocities | 12 + 12 = 24 dims |

**If no action is ever given, the robot sinks.** Obvious, but important —
even just standing requires giving joint targets at every step. That's task03.
