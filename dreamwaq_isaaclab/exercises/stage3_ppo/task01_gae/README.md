# task01 — GAE advantages  `L2 · BUILD`

## Goal

Compute "how much better than average this action was". It's the signal that decides what PPO
reinforces and what it suppresses.

```
delta_t = r_t + γ V(s_{t+1})(1 - done_t) - V(s_t)        <- TD error
A_t     = delta_t + γ λ (1 - done_t) A_{t+1}             <- recursion
returns = A + V                                          <- the critic's regression target
```

`λ` controls the bias–variance tradeoff. `λ=0` gives 1-step TD (high bias, low variance),
`λ=1` gives Monte Carlo (no bias, high variance). The real project uses `λ=0.95`.

## What to fill in

`TODO(ppo-gae)` in `starter.py` — the body of `compute_gae`.

**Three traps**

1. Sweep **from back to front** so `A_{t+1}` is already known (`reversed`)
2. At `t = T-1`, `V(s_{t+1})` is the `last_value` passed in — the rollout was cut off mid-episode,
   so bootstrap from there
3. At `done` steps there's no continuation to the next state, so cut with `(1 - done)` in **two places**

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | Returns two tensors of shape `(T, N)` |
| 2 | `returns == advantages + values` |
| 3 | Values match the checker's independent implementation |
| 4 | **With `λ=0` it equals the 1-step TD error** |
| 5 | **With `γ=λ=1` it equals Monte Carlo** |
| 6 | Cuts the bootstrap at `done` steps |
| 7 | `done` actually changes the result |
| 8 | The last step uses `last_value` |

4·5 are good checks — GAE interpolates between two extremes, so if it matches the known formulas at both ends,
the middle is likely right too. 6·7·8 directly target common mistakes.

## When you're done

`python starter.py` runs real training (about 40 s). Success if the reward rises from 0.44 → 0.95.
