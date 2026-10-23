# direct-dones — Termination conditions  `L2 · BUILD`

**Stage 2 · Training Env class**

## Goal

Decide when to end an episode. There are two cases.

| | What | Nature |
|---|---|---|
| `time_out` | Used up the allotted number of steps | **Successful termination** — must bootstrap |
| `base_contact` | The base touched the ground = fell over | **Failure termination** |

Distinguishing the two is important. The last state of an episode cut off because time ran out is not a "bad state".
So `_get_dones` returns the two **separately**, and the learner bootstraps the value
only for timeouts. If you merge them into one, a wrong penalty gets attached to "surviving long" itself.

## Why base contact

It is **exactly the same** as `legged_robot.py:check_termination` in the original IsaacGym DreamWaQ.

```
reset_buf = any( norm(contact_forces[:, termination_contact_indices, :]) > 1.0 )
```

`terminate_after_contacts_on = ["base"]`, threshold `1.0 N`. The Manager stack
uses the same body and threshold via `mdp.illegal_contact`. **Both stacks and the original all agree.**

A 1 N threshold is very low — the Go2 base merely brushing against something terminates the episode. It's meant to stop the robot
from flailing around after falling and scraping up reward.

## What to fill in

```
starter/dreamwaq_env.py  →  TODO(direct-dones)  inside  DreamWaQEnv._get_dones()
```

Given:

| | shape / meaning |
|---|---|
| `self.episode_length_buf` | `(N,)` number of steps each env has taken in the current episode |
| `self.max_episode_length` | maximum number of steps |
| `self._contact_sensor.data.net_forces_w_history.torch` | `(N, 3, bodies, 3)` — 3 history frames |
| `self._termination_contact_ids` | base body indices |
| `self.cfg.termination_contact_force` | `1.0` |

Create the two variables `base_contact` and `time_out`, and the code below returns them.

## Reducing axes is all there is to this exercise

`net_forces_w_history` is 4-dimensional. You must reduce it to an `(N,)` boolean.

```
(N, history=3, bodies, 3)   force vectors
  → norm(dim=-1)            (N, 3, bodies)     convert to magnitude
  → max(dim=1)[0]           (N, bodies)        max over history
  → > threshold             (N, bodies)        boolean
  → any(dim=1)              (N,)               any of the bodies
```

**Use the max over history, not the mean.** If it hit hard even once in the 3 frames, it fell.
Using the mean misses short, strong impacts.

## Checks

```bash
cd exercises/stage2_env/direct-dones
python check.py            # fast check — no Isaac Sim needed, 2 s
```

Pass criteria (10):

1. Returns two `(N,)` tensors; if nothing happens both are `False`
2. Only envs with `episode_length >= max - 1` time out
3. Only envs with a force above the threshold on the base terminate
4. **Exactly at the threshold is not a termination** — it's `>`, not `>=`
5. **Hitting in just one of the 3 history frames terminates** — using the mean gets caught
6. **Foot contact is not a termination** — miss this and the robot dies on its first step
7. **Force is the vector norm, not the components** — `(0.8, 0.8, 0.8)` has every component below 1 but its norm is 1.39
8. timeout and base_contact don't affect each other

## Hints

<details>
<summary>Why max_episode_length - 1</summary>

`episode_length_buf` counts from 0. If `max_episode_length = 1000`, the valid step indices are
`0..999`. So cutting at `>= 999` gives exactly 1000 steps.
Writing `>= max_episode_length` takes one extra step.
</details>

<details>
<summary>Why `.torch` is attached</summary>

Isaac Lab sensor data can come as a warp array (`ProxyArray`). `.torch` converts it to a torch tensor.
Leave it out and you get a type error later in `torch.linalg.norm`.
</details>

<details>
<summary>How to filter out foot contact</summary>

**Index first** with `self._termination_contact_ids`.
Keep only the base body with `net_contact_forces[:, :, self._termination_contact_ids]`, then
reduce the axes. If you try to select after reducing everything, the information is already smeared together.
</details>

## How does the Manager side do it

It does the same thing with one line of config.

```python
base_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0},
)
```

If you open up `mdp.illegal_contact`, it contains **the same axis reduction** as written above.
Direct writes it directly, while Manager picks an existing term — that's the difference between the two APIs.

For why there's no exercise on the Manager side, see the [Stage 2 README](../README.md).

## Next

Stage 2 is complete. Go to [Stage 3 — PPO](../../stage3_ppo/).
