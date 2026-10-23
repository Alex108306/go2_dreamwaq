# task02 — PPO loss function  `L2 · BUILD`

## Goal

Build the **clipped surrogate objective** that makes PPO PPO.

```
ratio = exp(log_prob - old_log_prob)
surrogate = -mean( min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A ) )
value_loss = mean( (V - returns)² )
entropy_loss = -mean(entropy)
total = surrogate + c_v · value_loss + c_e · entropy_loss
```

## What to fill in

`TODO(ppo-losses)` in `starter.py` — the body of `ppo_losses`.

**The key is `min`**

Using only `clip` is wrong. It must be `min(unclipped, clipped)`.

| Situation | Why |
|---|---|
| `A>0`, `ratio` deviates **upward** | Clip → don't push even a good action too far at once |
| `A>0`, `ratio` deviates **downward** | **No clip** → leaves room to undo it |
| `A<0`, `ratio` deviates downward | Clip |
| `A<0`, `ratio` deviates upward | No clip |

`min` means "**always take the pessimistic side**". Check 8 catches exactly this.

## Checks

```bash
python check.py
```

| # | Pass criterion |
|:---:|---|
| 1 | All four values are scalars |
| 2 | With `ratio=1`, `surrogate = -mean(A)` |
| 3 | `value_loss` = MSE |
| 4 | `entropy_loss = -mean(entropy)` — **watch the sign** |
| 5 | `total` is the coefficient-weighted sum |
| 6 | `A>0`, `ratio` above the upper bound → clipped |
| 7 | `A<0`, `ratio` below the lower bound → clipped |
| 8 | **`A>0`, `ratio` below the lower bound → not clipped** (`min` was used) |
| 9 | The advantage is not clipped |
| 10 | Gradient flows |

## One sign flips training

- **surrogate**: the objective is to **maximize** `ratio·A`, but the optimizer minimizes, so add a `-`
- **entropy**: we want entropy to be **large**, so it enters the loss with a `-`.
  Flip the sign and the policy immediately collapses to deterministic and exploration disappears

## When you're done

`python starter.py` runs real training (about 40 s). Success if the reward rises from 0.44 → 0.95.
