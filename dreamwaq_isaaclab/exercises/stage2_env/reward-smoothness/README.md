# reward-smoothness — Action jerk penalty  `L2 · BUILD`

## Goal

Build a term that penalizes the policy for moving its joints **jerkily**.

```
jerk = a_t - 2*a_{t-1} + a_{t-2}          (second-order finite difference)
reward = sum_j jerk_j^2                    (sum over 12 joints)
```

A second-order difference of 0 means **the action is changing at a constant rate**.
Only sudden acceleration and sudden stops are penalized; smooth change passes.

## What to fill in

`TODO(reward-smoothness)` in `starter/rewards.py` — `action_smoothness_l2.__call__`.

**There are two traps.**

1. IsaacLab's `action_manager` gives **only** `action` (t) and `prev_action` (t−1).
   This class must hold t−2 itself in `self.prev_prev_action`.
2. Plain assignment makes it point to the same tensor. Think about why `.clone()` is needed.

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | Returns `(num_envs,)` — one scalar per env |
| 2 | Jerk 0 if the action stays 0 |
| 3 | **Jerk 0 for a linear ramp** — constant-rate change isn't penalized |
| 4 | `a=(0,0,1)` → `12` (12 joints × 1²) |
| 5 | **The t−2 buffer is updated** — if not, `a=(0,1,2)` gives 48 |
| 6 | `prev_prev_action` is not an alias (`.clone()`) |
| 7 | `reset()` resets the buffer to 0 |

Numbers 5 and 6 are the core of this exercise. Getting just the formula right passes 1–4, but
**if you get the state management wrong, wrong rewards flow silently during training.**

## Why the state must be held directly

IsaacLab's `ManagerTermBase` exists for cases like this. If a reward term has state,
`reset(env_ids)` must also be implemented — if the buffers of envs whose episodes ended aren't cleared,
**the last action of the previous episode leaks into the new episode's t−2.**

The Direct side manages this directly as arrays inside `dreamwaq_env.py`. How the two APIs solve
the same problem differently is the thing to watch in Stage 2.
