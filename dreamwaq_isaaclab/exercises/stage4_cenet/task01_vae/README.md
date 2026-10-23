# task01_vae — Plain VAE  `L3 · DESIGN`

**Stage 4 · CENet (VAE)**

## Goal

Write the `ToyVAE` class **in its entirety**. Neither a skeleton nor signatures are given — only the API spec below.

The four pieces written here (encode / reparameterize / decode / ELBO) become CENet as-is in the next exercise.
This is the step of building just the VAE first, with no robot and no Isaac Sim.

## The toy problem

`make_toy_batch()` produces data whose **true degrees of freedom are only 2 but which is observed in 8 dims**.
It spreads a 2-dim latent vector into 8 dims with a fixed random matrix and adds a little noise.

In DreamWaQ terms, this corresponds to "the true state of the terrain is summarized by a few numbers, but the robot only sees a 225-dim observation
history". The VAE's job is to recover that low-dimensional structure.

## What to fill in

```
starter.py  →  TODO(vae-core)
```

Since this is L3, the TODO is at **module top level**. The moment you `import` it, `NotImplementedError` is raised.
Just write the whole `ToyVAE` class in its place.

`make_toy_batch()` and `main()` are already given. `main()` uses your class as
`ToyVAE()`, `net.loss(data)`, `net.parameters()`, so **you must match the API exactly.**

## Required API

```python
class ToyVAE(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, latent_dim=2, beta=1.0)
    def encode(self, x)            -> (mu, logvar)          # each (B, latent_dim)
    def reparameterize(self, mu, logvar) -> z               # (B, latent_dim)
    def decode(self, z)            -> x_hat                 # (B, input_dim)
    def forward(self, x)           -> (x_hat, mu, logvar, z)
    def loss(self, x)              -> (total, recon, kl)    # 3 scalar tensors
```

Follow two conventions. **They are set to match the conventions CENet uses.**

- The `kl` returned by `loss` **already has `self.beta` multiplied in.** So `total = recon + kl`.
- The KL is **averaged over latent dims, and also averaged over the batch**. That's because `recon` is `nn.MSELoss()` —
  i.e. a mean per input dimension. **Matching the reduction axes of the two terms** is the core of the convention
  (for why, see the "What β does" section below).

## Checks

```bash
cd exercises/stage4_cenet/task01_vae
~/IsaacLab/_isaac_sim/python.sh check.py     # fast check — no Isaac Sim needed, 1 s
~/IsaacLab/_isaac_sim/python.sh starter.py   # train it yourself (~5 s)
```

> The checker is pure torch but **is run with the bundled kit python.** The system `python3` doesn't have torch.

Pass criteria (20):

1. `encode` returns `mu`, `logvar`, each `(B, latent_dim)` — the same half isn't used twice
2. `reparameterize`
   - with very small `logvar`, `z ≈ mu` (i.e. `std = exp(0.5 * logvar)`)
   - calling twice with the same arguments gives **different samples**
   - with `logvar = 0`, the sample standard deviation is about 1
   - **gradient flows** to `mu` ← this is the key
3. `decode` / `forward`
   - shapes and number of return values
   - **calling `forward` twice with the same `x` gives different `x_hat`** ← sampling must actually be
     on the `forward` path. With `decode(mu)` all the shapes match but this catches it
4. ELBO
   - `total == recon + kl`
   - `kl == 0` when `mu = 0, logvar = 0`
   - `kl` increases as `mu` or `logvar` moves away from 0
   - `kl` is **averaged** over latent dims (`mu=1, logvar=0` → `kl == 0.5`).
     The checker tells by whether the value stays the same when `latent_dim` goes 2 → 4
5. Doubling `beta` doubles `kl` exactly
6. `recon == 0` if `x_hat == x`
7. The loss actually decreases with 200 steps of training

## Hints

<details>
<summary>Step 1 — the encoder outputs a distribution</summary>

An ordinary autoencoder outputs a latent **vector**. A VAE outputs a latent **distribution**.
So the encoder output is `latent_dim * 2` — the first half is `mu`, the second half `logvar`.

Split it with `torch.split` or slicing. Why `logvar` and not `var`?
The variance must be positive, but it's hard to put a sign constraint on a network output.
Taking it as a log lets `exp` make it positive automatically.
</details>

<details>
<summary>Step 2 — reparameterization trick</summary>

Drawing directly from `z ~ N(mu, sigma²)` is **not differentiable.** Sampling isn't a differentiable operation.

Instead, write it like this.

```
std = exp(0.5 * logvar)
eps ~ N(0, I)            <- push all the randomness here
z   = mu + eps * std     <- gradient flows straight through mu and std
```

The reason you `exp` `0.5 * logvar` rather than `logvar` is that `logvar = log(σ²)`,
so `σ = exp(0.5 · log σ²)`.
</details>

<details>
<summary>Step 3 — ELBO</summary>

Two terms.

- **Reconstruction**: how different `x_hat` is from `x` → MSE
- **Regularization**: how far `q(z|x)` departs from `p(z) = N(0, I)` → KL

The KL between diagonal Gaussians has a closed form.

```
KL(q || p) = -0.5 * sum_j ( 1 + logvar_j - mu_j² - exp(logvar_j) )
```

The textbook form is `sum_j` (sum over latent dims), but here we use the **mean over latent dims** —
matching the axis because `recon` is `nn.MSELoss()` = per-dimension mean. Then the batch mean,
and finally multiply by `beta`.
</details>

## What β does

Increasing `beta` strengthens the regularization and pulls the latent distribution toward the standard normal.
Too strong and you get **posterior collapse** — the latent vector becomes independent of the input and carries no information.
Too weak and it's just an autoencoder.

Run it once with `beta=1.0` and once with `beta=10.0` and see for yourself how `recon` changes.

```bash
~/IsaacLab/_isaac_sim/python.sh starter.py                # beta = 1.0
~/IsaacLab/_isaac_sim/python.sh starter.py --beta 10.0
```

### What to look at

| | `kl` (last) | `recon` (last) |
|---|---|---|
| `beta=1.0` | 0.59 | **0.48** |
| `beta=10.0` | **0.003** | **1.77** |

At `beta=10`, `kl` **sticks to the floor** at 0.003. Looking at the regularization alone, it seems "it worked".
But `recon` **went up** from 0.48 → 1.77.

**`kl`→0 without `recon` coming down with it is the signal of posterior collapse.**
`q(z|x)` slumps to `N(0,I)` regardless of the input so the latent carries no information,
and the decoder ignores the latent and just emits the mean. Unable to use the latent, the reconstruction can only get worse.
(If `recon` gets small along with a small `kl`, that's not collapse — it actually worked.)

**And the toy's `beta=10` is an effective β of 40** (`D/L = 8/2 = 4`, table below).
That's practically the same spot as the old CENet's effective β of 45. **What you just saw is what happened in CENet.**

### But the `beta` you wrote isn't the `beta` you think it is

In the textbook β-VAE both recon and KL are **sums** (sum/sum). Converted to that baseline, the **effective β**
depends on the reduction axes of the two terms.

| recon axis | KL axis | effective β |
|---|---|---|
| D-dim mean | L-dim **sum** | **D · β** ← mismatched axes |
| D-dim mean | L-dim mean | (D / L) · β ← the convention this exercise uses |

Here the toy has `D=8, L=2`, so with matched axes the effective β = 4·β, and with mismatched axes it's 8·β.
The difference is only 2×, so it's hard to notice in the toy.

**In CENet it was noticeable.** With `D=45, L=16`, the effective β of the old code with mismatched axes was **45·β**,
and annealing up to `beta_limit=4` made it **180**. The result was posterior collapse — latent information content of
0.02–0.14 nats, practically 0 bits. Now the KL has been changed to a per-dim mean, bringing the effective β down to (45/16)·β ≈ **2.81·β**,
and annealing was removed from the training path as well.

> **Why we don't write "β in paper eq.7 is constant".** The paper gives β **neither a number nor a
> schedule** (the word "constant" doesn't appear either). "No schedule was written, so it's constant" is **our reading**,
> not a sentence from the paper. We implemented it as a constant following that reading, and moved annealing behind an
> experiment gate (`DWQ_CENET_BETA_ANNEAL`) that is off by default.
> What is the paper's and what is our choice is tabulated in the
> "What the paper fixes vs. what we chose" section of [`cenet-loss`](../cenet-loss/).

> **"Effective β" is also our term.** It isn't in the paper. It's the coefficient attached to `KL_sum` when `recon` is
> converted to a sum, i.e. the ratio looking at **recon : KL only**, leaving out the `vel` term. So for the same training run
> it can be written as 4·β or 8·β depending on which unit you use — **when quoting a number, state the unit
> with it.**

That's why this exercise pins the KL to a **per-dim mean**. More than the value itself, the point is the habit of
**matching the axes of the two terms**. The full story is covered again in the [`cenet-loss`](../cenet-loss/) exercise.

## Next

- [`cenet-forward`](../cenet-forward/) — attach an `est_vel` branch to this structure to make the real CENet (L2)
