# reward-foot-clearance — Foot lifting penalty  `L2 · BUILD`

## Goal

Build a term that makes the robot move its feet by **lifting them rather than dragging**.

```
reward = sum_feet (foot_height_above_terrain - 0.12)^2 * lateral_velocity
```

Same form as the original IsaacGym (`legged_robot.py:_reward_foot_clearance`).

## What to fill in

`TODO(reward-foot-clearance)` in `starter/rewards.py` — the body of `foot_clearance_l2`.

## Three design points

**1. "Height above terrain", not world z**

Using `body_pos_w[:, :, 2]` as-is gives completely different values to a robot on a hill and a robot in a valley.
With the same idea as Stage 1 task05's height scan, approximate the terrain height with the mean of the height scanner's ray hits
and subtract it.

> The original samples terrain height separately for each foot. Doing that in IsaacLab would need 4 more RayCasters,
> so here it is approximated with the mean of the scanner under the base — exact on flat ground,
> approximate on slopes. `base_height_l2` uses the same approximation.

**2. Why weight by lateral velocity**

Clearance is demanded only while the foot is **being moved**. A planted foot (velocity 0)
is not penalized even if its height differs from the target. Without this weighting the robot tries to stand
with all four feet lifted.

z velocity is not included — moving vertically up and down is no reason to penalize in itself.

**3. Ray hits can be contaminated**

If a ray hits nothing, you get `NaN`/`inf`. Averaging that as-is makes the whole reward
`NaN` and training silently breaks. Clean it up with `torch.nan_to_num`.

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | Returns `(num_envs,)` |
| 2 | Penalty 0 when the feet are exactly at the target height |
| 3 | **Penalty 0 when lateral velocity is 0** (planted foot) |
| 4 | height 0 / velocity 1 / 4 feet → `4 × 0.12² = 0.0576` |
| 5 | The penalty is linearly proportional to lateral velocity |
| 6 | **Subtracts terrain height to use 'height above terrain'** |
| 7 | Cleans NaN/inf in ray hits to 0 |
| 8 | Doesn't include the z component in lateral velocity |
| 9 | Falls back to world z if the sensor is `None` |

6·7·8 are the core of this exercise. The rest pass as long as the formula is right.
