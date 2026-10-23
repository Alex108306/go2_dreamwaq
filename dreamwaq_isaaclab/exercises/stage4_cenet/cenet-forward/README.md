# cenet-forward — CENet's forward  `L2 · BUILD`

**Stage 4 · CENet (VAE)**

## Goal

Extend the VAE built in `task01_vae` into the **real CENet**. Write `CENet.forward` in its entirety.

There's just one difference from the toy VAE — **one more velocity-estimation branch is attached** to the encoder output.

```
obs_history ─▶ encoder ─▶ h (latent_dim1)
                          │
              ┌───────────┴───────────┐
              ▼           ▼           ▼
          est_vel(v)    mu(c)     logvar(c)
              │           └─────┬─────┘
              │                 ▼
              │              reparam
              │                 ▼
              │          context_vec(c)
              └────────┬────────┘
                       ▼
                cat (latent_dim2) ─▶ decoder ─▶ est_next_obs
```

`v` is the velocity dimension and `c` the context dimension. A single encoder output `h` contains three pieces,
and splitting them exactly is this exercise. **Only `v` is fixed** — `c` is set by the cfg, so
it must be measured from the size of `h`'s last axis; hard-coding numbers breaks when the cfg changes.
Don't back it out from `latent_dim1` / `latent_dim2`. Those two look mutually consistent even when the cfg
is wrong, so they can't catch a bad cfg (pass criterion 6).

## Why attach a velocity branch

The Go2 has no sensor that measures base velocity. Yet the main term of the reward function is
`track_lin_vel_xy_exp` — **how well it follows the commanded velocity**. You can't follow it without knowing your velocity.

So CENet **estimates** velocity from the observation history and is supervised with the actual velocity,
which is only accessible during training. At deployment only the estimate is used. These are 3 of the dims that separate Base (45) from Waq (64).

## What to fill in

```
starter/cenet.py  →  TODO(cenet-forward)  inside  CENet.forward()
```

`CENet.update()` in the same file is also blank. **That belongs to the next exercise (`cenet-loss`)**, so
it was deliberately removed, and this exercise's checker doesn't call `update()`. For now
fill in only `TODO(cenet-forward)`.

> **This `starter/cenet.py` is not exactly the same as the production source.** The real `cenet.py` has
> experiment switches (`DWQ_CENET_*`) turned on by env vars, but they are **all off by default** and
> have nothing to do with this exercise, so they were stripped at generation time. What to turn on and why is in
> the 'The real source has more switches' section of the [`cenet-loss`](../cenet-loss/) README.

Already given:

| | |
|---|---|
| `self.encoder` | `input_dim → 128 → 64 → latent_dim1` |
| `self.decoder` | `latent_dim2 → 64 → 128 → 48 → output_dim` |
| `self.reparameterize(mu, logvar)` | The same function you wrote in task01. **Already implemented** |

## Return convention

```python
return est_next_obs, est_vel, mu, logvar, context_vec
```

You must keep this order. `before_action()` and `update()` receive them in this order.

| Name | shape | Where it's used |
|---|---|---|
| `est_next_obs` | `(B, output_dim)` | prediction for `recon_loss` |
| `est_vel` | `(B, v)` | `vel_loss` + **actor input** |
| `mu`, `logvar` | `(B, c)` | `kl_loss` |
| `context_vec` | `(B, c)` | decoder input + **actor input** |

`est_vel` and `context_vec` concatenated are exactly the actor observation augmentation (`runner-augment` exercise).

## Checks

```bash
cd exercises/stage4_cenet/cenet-forward
~/IsaacLab/_isaac_sim/python.sh check.py   # fast check — no Isaac Sim needed, 1 s
```

> The checker is pure torch but **is run with the bundled kit python.** The system `python3` doesn't have torch.

Pass criteria (18):

1. Returns 5 values as a tuple
2. The shapes of the five tensors are correct
3. **Splits the encoder output exactly into one velocity estimate and two sets of distribution parameters** — the three must neither overlap
   nor leave anything over; the checker runs the encoder separately and compares which position of `h` each piece is
4. **The width the decoder receives is `latent_dim2`, and its latent part is a sample, not the distribution's mean** —
   the checker intercepts the decoder input to tell these two apart
5. For the same input `mu`/`logvar` are deterministic and only `context_vec` changes every time (sampling actually happens)
6. Given a `latent_dim1` whose distribution parameters can't be split into two sets, it raises an exception instead of passing silently
   (which exception it should be is in hint step 2)
7. Gradient flows to both encoder and decoder

> The checker prints which item failed, down to position and numbers. **Failing first and reading that output**
> leaves more behind than reading the answer here in advance.

## Hints

<details>
<summary>Step 1 — the encoder output into three pieces</summary>

Rather than cutting into three pieces at once, **cutting in two passes** is easier to read.
First detach the velocity branch, then split the rest in half again.

Don't hard-code the cut widths. Only the velocity's dimension count is fixed; the rest
must be measured from the size of `h`'s last axis. That way, even if the cfg changes the context size,
`forward` doesn't need to change.
</details>

<details>
<summary>Step 2 — guarding against odd counts</summary>

To split the remaining piece in half, its count must be even. Odd means
`latent_dim1` was given wrong, so don't pass silently — raise an `AssertionError`.
It's better than finding out after training for hours with a wrong cfg.
</details>

<details>
<summary>Step 3 — what goes into the decoder</summary>

The width the decoder receives is `latent_dim2` = 19. Start by counting what must be concatenated with what to get 19.

There are two candidates for the trailing 16 positions of the 19 — the distribution's **mean** (`mu`), or
a **sample** drawn from it. Feed the mean and the randomness disappears, making it just an autoencoder,
and the KL term has nothing to regularize. The checker distinguishes these two separately.
</details>

## Why `.requires_grad_(True)` is attached

Looking at the finished version, `.requires_grad_(True)` is attached to `mu`, `logvar`, `context_vec`.
The `split` results come from the encoder output and already carry gradient, so **it works without it.**
It's a remnant of porting the original DreamWaQ implementation as-is. The checks pass without it.

## Next

- [`cenet-loss`](../cenet-loss/) — build the three loss terms from these five values (L2)
