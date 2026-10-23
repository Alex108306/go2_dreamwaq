# task03 — PD control and action scale  `L1 · FILL`

## Goal

See how the policy's action turns into a joint target, and **what the PD gains determine**,
by standing two robots side by side.

## Key formula

```
joint_target = default_joint_pos + action * 0.25
               └─ use_default_offset=True        └─ scale=0.25
```

The policy outputs **"how far to deviate from the default pose"**, not "put this joint at X rad".
Since the action is clipped to ±1, the joint target is confined to within **±0.25 rad** of the default pose.

## What to fill in

`TODO(pd-target)` in `starter.py` — one line in `action_to_target()`.

## Run

```bash
python starter.py
python solution.py --viz kit   # the two standing side by side
```

## Pass criteria

1. `action=+1` → the joint target deviates from the default pose by exactly `0.25 rad`
2. `action=0` → target = default pose
3. `Kp↑` → joint tracking error ↓
4. `Kp↑` → base height ↑

## What you get when you run it

| | Kp | Height | Joint tracking error |
|---|---:|---:|---:|
| Soft (project value) | 20 | 0.232 m | **0.604 rad** |
| Stiff (control group) | 80 | 0.309 m | 0.068 rad |

**Both received the same command, `action=0`, i.e. "hold the default pose".**

## Why this matters

`Kp=20` can't even fully support its own weight and sags by 0.6 rad. Yet this project
uses those soft gains **on purpose** (`go2_base_cfg.py` overrides IsaacLab's default 25.0 with 20.0).

- The real Go2's joints are about that soft. Making them stiff only in sim makes it diverge that much more when transferred to hardware
- The policy learns **to make up the shortfall with its action at every step**

It's also no coincidence that the Stiff height `0.309 m` is almost the same as DreamWaQ's `base_height_target = 0.30`
— the default pose is set up to stand at the target height.

## Try it yourself (optional)

Raise `ACTION_SCALE` to `1.0` and watch in the GUI. With joint targets jumping up to ±1 rad,
what happens to the robot? You'll see why this project uses `scale=0.25`.
