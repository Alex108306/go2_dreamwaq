# task05 — Sensors: contact and height scan  `L2 · BUILD`

**The last and most important exercise of Stage 1.** The two values read here are
the raw material for the termination condition and privileged observation you'll build in Stage 2.

```
ContactSensor  →  is the base touching the ground?        →  termination condition
RayCaster      →  187 terrain heights around the robot    →  Oracle's privileged observation
```

## What to fill in

`TODO(height-scan)` in `starter.py` — one line, but **you have to understand its meaning to get it right.**

`ray_hits_w[..., 2]` is the **world-frame z** of the ray hit points. If you put this straight into the observation,
the same terrain becomes completely different numbers depending on whether the robot is on a hill or in a valley.

The observation must tell **"how much higher my base is than the terrain under my feet"**.

```
height_scan = base_z - hit_z - BASE_HEIGHT_TARGET
```

- `base_z` is `(num_envs,)`, `hit_z` is `(num_envs, num_rays)` — get the broadcasting right
- Since `BASE_HEIGHT_TARGET = 0.30` is subtracted, **the result is 0 when standing at the target height**
- This is what the project's `mdp.height_scan` does

## Run

```bash
python starter.py
python solution.py --viz kit
```

## Pass criteria

1. 187 rays (`17 x 11`)
2. The 4 feet are on the ground (contact force sum > 1 N)
3. The base isn't touching the ground (< 1 N)
4. On flat ground every hit point z is near 0
5. Relative height = base height − terrain height − 0.30

## What you get when you run it

```
base contact force    0.00 N   ← the termination condition looks at this value (terminates if > 1 N)
FL_foot contact force 39.46 N
FR_foot contact force 42.25 N
RL_foot contact force 33.85 N
RR_foot contact force 32.47 N       sum 148 N ≈ Go2 weight 15 kg × 9.81
```

The contact force sum matches the robot's weight. That means the physics is running correctly.

## Why 187

`0.1 m` spacing, `1.6 x 1.0 m` grid → `17 x 11 = 187`.

These 187 dims are **privileged information that only Oracle receives**.

| Model | actor observation | height scan |
|---|---:|---|
| Base | 45 | ✗ |
| Oracle | 48 | ✗ (only the critic receives it, as 235 dims) |
| Waq | 64 | ✗ — **CENet estimates it from past observations only** |

DreamWaQ's core claim is right here. None of the three actors can see the height scan directly.
Instead, Waq **imagines** the terrain context from its past observation history. That's Stage 4.

## Caution: the contact sensor has a different body order

The `ContactSensor`'s body indices are **different** from the `Articulation`'s. If you mix them up
you read the wrong body (this is a bug that actually happened in this project).

```python
feet_idx, _ = contact.find_sensors(".*foot")   # for the contact sensor
feet_ids, _ = robot.find_bodies(".*foot")      # for the articulation — may be different values
```
