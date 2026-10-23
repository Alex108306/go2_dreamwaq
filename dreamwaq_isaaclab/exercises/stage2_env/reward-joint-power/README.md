# reward-joint-power — Joint power penalty  `L1 · FILL`

## Goal

A term that keeps the robot **from wasting energy**.

```
reward = sum_j |tau_j| * |theta_dot_j|
```

**Joint power** in Table I of the paper, weight `−2×10⁻⁵`.

## What to fill in

`TODO(reward-joint-power)` in `starter/rewards.py` — the body of `joint_power_l1`.

| What to read | Meaning |
|---|---|
| `asset.data.applied_torque` | The torque actually applied to the joints |
| `asset.data.joint_vel` | Joint angular velocity |
| `asset_cfg.joint_ids` | Target joints (may not be all of them) |

Take the **absolute value** of both — power is treated as "consumption", so the sign is meaningless.

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | Returns `(num_envs,)` |
| 2 | torque=1, angular velocity=1, 12 joints → `12` |
| 3 | 0 if torque is 0 |
| 4 | **0 if angular velocity is 0** — holding torque at standstill isn't penalized by this term |
| 5 | **Sign-independent** (absolute values used) |
| 6 | Respects `asset_cfg.joint_ids` |

## Why this term exists

Paper §II-B-4:

> The complex reward function for learning a locomotion policy usually includes a motor power
> minimization term.

Number 4 is interesting — if angular velocity is 0, the penalty is 0 no matter how large the torque. **Holding still under load
isn't penalized by this term.** That's why the next exercise's `power_distribution` is needed separately.

## Next

[`reward-power-distribution`](../reward-power-distribution/) — a term that penalizes **imbalance**, not the total.
