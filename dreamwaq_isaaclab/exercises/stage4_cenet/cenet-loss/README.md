# cenet-loss — CENet loss function  `L2 · BUILD`

**Stage 4 · CENet (VAE)**

## Goal

Build CENet's training signal yourself. Complete the three terms.

| Term | What it measures |
|---|---|
| `vel_loss` | Whether the estimated velocity `est_vel(3)` matches the actual base linear velocity |
| `recon_loss` | Whether the **next step's observation (45)** can be reconstructed from the latent representation |
| `kl_loss` | How far the latent distribution departs from the standard normal (β-VAE regularization) |

These three terms are the core idea of DreamWaQ. The robot can't see the terrain directly, but
**if you force it to reconstruct "what happens next" from the past observation history alone**,
terrain information seeps into that latent vector. This is implicit terrain imagination.

## What to fill in

```
starter/cenet.py  →  TODO(cenet-loss)  inside  CENet.update()
```

Values already passed in by `self.forward()`:

| Variable | shape | Meaning |
|---|---|---|
| `est_vel_batch` | `(B, 3)` | Base linear velocity estimated by the encoder |
| `true_vel_batch` | `(B, 3)` | Actual base linear velocity — normalized by the runner's `true_vel_rms` (velocity-only running mean/std normalizer, `rsl_rl`'s `EmpiricalNormalization`) |
| `est_onext_batch` | `(B, 45)` | Next-step observation reconstructed by the decoder |
| `true_onext_batch` | `(B, 45)` | Actual next-step observation — normalized by the runner's `obs_rms` (running mean/std normalizer for the 45 base observations). This matches the scale of the encoder input (normalized history) |
| `mu_batch` | `(B, 16)` | Mean of the latent distribution |
| `logvar_batch` | `(B, 16)` | **Log variance** of the latent distribution |

`self.beta` is the KL weight. In this implementation it **is a constant** — paper eq.7 gives β no schedule, so
we read it as a constant (this is our reading, not a quote — see "What the paper fixes vs. what we chose" below),
and the `beta *= 1.01` annealing that used to be on the training path **was moved behind a gate that is off by default**
(see the "Reduction axis" section and the "The real source has more switches" section below). This exercise's starter
doesn't have that gate, so `self.beta` stays the same value from start to finish.

You must build `total_loss` as well. `total_loss.backward()` below uses it.

## Checks

```bash
cd exercises/stage4_cenet/cenet-loss
~/IsaacLab/_isaac_sim/python.sh check.py   # fast check — no Isaac Sim needed, 1 s
```

> The checker is pure torch but **is run with the bundled kit python.** The system `python3` doesn't have torch.

Pass criteria (15):

1. All three terms are reduced to scalars (`.item()` must work)
2. `vel_loss == 0` if `est_vel == true_vel`, `recon_loss == 0` if `est_onext == true_onext`
3. `kl_loss == 0` if `mu == 0, logvar == 0` (i.e. q = p)
4. `kl_loss` **increases** when `logvar` grows or `mu` moves away from 0.
   And **the KL and recon are reduced over the same axis.** The checker increases the latent dims 16 → 32
   and the observation dims 45 → 90 and judges by **whether neither value changes** —
   if you used a reduction that grows in proportion to the number of dims, it's caught here.
   **Decide yourself which way to match them; if you're wrong, the checker tells you along with the expected value**
5. Doubling `self.beta` makes `kl_loss` **exactly 2x** (vel/recon unchanged)
6. `total_loss` is the sum of the three terms. It must hold **even when `beta ≠ 1`**
   (if β is multiplied only on the `total_loss` side and not into the returned `kl_loss`, it's only right at β=1)
7. After `total_loss.backward()`, gradient flows to both encoder and decoder

## Hints

<details>
<summary>Step 1 — the two MSE terms</summary>

An `nn.MSELoss()` instance, called as `loss(pred, target)`, gives a mean squared error scalar.
`vel_loss` and `recon_loss` are one line each.
</details>

<details>
<summary>Step 2 — the closed form of the KL</summary>

The KL between the diagonal Gaussian `q = N(mu, sigma^2)` and `p = N(0, I)` is

```
KL(q || p) = -0.5 * sum_j ( 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 )
```

Here `log(sigma^2)` is exactly `logvar`, and `sigma^2 = logvar.exp()`.

The textbook formula is `sum_j` — a **sum** over latent dims. But **`recon_loss` right above is
`nn.MSELoss()`, i.e. a 45-dim mean**. The next section shows from measurements what happens when the two terms
are reduced over different axes. Decide yourself which way to go, and ask `check.py`.
(The batch axis must be reduced once more, separately — `backward()` requires a scalar.)
</details>

<details>
<summary>Step 3 — where to multiply beta</summary>

A β-VAE balances reconstruction and regularization with β. β is multiplied **only into the KL term**.
</details>

## Reduction axis — where this implementation once collapsed

Even with the same formulas as the paper, training lives or dies depending on **which axis the two terms are reduced over**.
In this project it actually died; here is the process and the fix, as they happened.

### What happened

The old code was like this.

```python
recon_loss = nn.MSELoss()(est_onext, true_onext)   # 45-dim 'mean'
kl_loss    = klds.sum(1).mean(0, True) * self.beta  # 16-dim 'sum'
self.beta  = min(self.beta * 1.01, self.beta_limit) # on top of that, β grows up to 4
```

A standard β-VAE has **both terms as sums** (sum/sum). Converting the above to that form,
only recon is squashed by 1/45, so

```
effective β = 45 · β        → 45 at β=1, 180 at beta_limit=4
```

The result was **posterior collapse**. Measurements: the trained encoder emitted the same values for different
inputs (`mu→0, logvar→0, σ→1.000`), and the information content of the 16-dim latent was
**0.02–0.14 nats** — practically 0 bits. The 16 context dims the actor received were pure Gaussian noise
drawn fresh every step. Even `est_vel` died along with it through the shared trunk.

**Just removing the annealing isn't enough.** Even with it removed, an effective β=45 remains.

### How it was fixed

**The KL was changed to a mean over latent dims**, unifying all three terms as "mean per dimension".

| Term | Axis | Dims |
|---|---|---|
| `vel_loss` | mean | 3 |
| `recon_loss` | mean | 45 |
| `kl_loss` | **mean** | 16 |

That is, the latent-dim reduction that was `.sum(1)` was lowered to `.mean(1)`. The batch mean and β after it are unchanged.

It could have been matched the other way (= recon as a sum), but then `vel_loss` is a 3-dim mean while
only `recon_loss` becomes 45x, and **the vel:recon ratio is skewed from 1:1 to 1:45.** Two of the three terms
are already means, so lowering the KL to a mean preserves both ratios.

> **45 or 15 — count it by hand.** Let the per-element error be `e`.
> Keep `vel` as a 3-dim **mean**, so `vel = e`, and change only `recon` to a 45-dim **sum**, so
> `recon = 45e` → it's **1:45**. To get `3:45 = 1:15`, **vel must be a sum too**
> (`vel = 3e`). That is, `1:45` is the number for "recon only as a sum", and `1:15` for "both as sums".
> The `vel : recon` row in the table below is `3:45 = 1:15` because that one is the **"everything summed"** reading,
> while the text here is the **"recon only"** scenario, hence `1:45`. Different numbers for different scenarios.
> (This document had `1:15` written here for a while — a value off by 3x.)

Now the effective β is

```
effective β = (45 / 16) · β ≈ 2.81 · β        (cfg β = 0.35 → about 0.98)
```

Not 45 or 180, but a little over 1 — a range where regularization is applied but doesn't kill the latent.

> **Why β is 0.35 and not 1.0.** The `CENet` class default is `beta=1.0`, but the cfg's
> `vae` block **overrides it with 0.35.** Once domain randomization was turned on, the latent died again at β=1.0
> (`|mu|` 2.9e-2 → 6.3e-4), and the value that survived with randomization on was 0.35. By the conversion above,
> 0.35 corresponds to **β ≈ 1 of the textbook sum/sum β-VAE**. Details are in `PAPER.md` §4 and
> [`stage5_compare/task04_diagnose`](../../stage5_compare/task04_diagnose/).
> Always check the setting via that run's `params/agent.yaml` — not the class default.

> **"I matched the axes, so why isn't the effective β equal to β?"** — **"Matching axes" means the two terms use the same kind of reduction
> (per-dim mean); it doesn't mean the numbers become the same as the textbook β.** The textbook
> β-VAE is sum/sum, so both D and L are multiplied in, while mean/mean divides both out.
> What's left is a factor of `D/L`. If you want the effective β to be exactly β, use sum/sum or multiply β by
> `16/45`. What we gained here is **"the ratio doesn't shake when the number of dims changes"**, not
> **"β is the textbook β"**.

Changed at the same time: **β annealing removed from the training path** (paper eq.7 gives β no schedule;
it was later revived behind the default-off gate `DWQ_CENET_BETA_ANNEAL`), **CENet lr 0.01 → 1e-3**
(the paper's Adam lr), **grad clipping `clip_grad_norm_(…, 1.0)` added** (same budget as PPO).

### Was the fix right — after-the-fact verification

"45·β died" alone isn't enough. Whether the new value is **correct or merely less bad** is only known by running it again.
Results of a full Waq rough 4000-iteration run after the fix ([`PAPER.md`](../../../PAPER.md) §4):

- `cenet_kl` **increased** from 0.018 (500it) → **0.030 (4000it)**. With collapse it goes down to 0.
  (This tag is not the KL itself but **`kl_loss` = `β · KL_mean`**. It has **different units** from numbers like
  `KL_sum = 4.18 nats` that come later — details in the "KL axis and β" section below.)
- The checkpoint `|mu|` mean **increases monotonically from 2.3e-2 to 2.9e-2** over the whole run (the collapsed run was at 2.1e-3 at 2500it).
- rough `Waq − Base` **flipped sign** from **−0.0166 → +0.0073.** CENet, which was worse than Base,
  is now better.

However, it's still a **single seed**, and what was recovered is 22% of the total headroom (Oracle−Base).

> **These three lines are from a sweep before domain randomization was turned on.** With randomization on, the latent
> died again with the same code, and this time the culprit was not the reduction axis but **the β value** (1.0 → 0.35). That is,
> **matching the reduction axes was a necessary condition, not a sufficient one.** The method for telling what killed what
> is built in [`stage5_compare/task04_diagnose`](../../stage5_compare/task04_diagnose/).

> There's one lesson here — **"I implemented it per the formula" and "it trains" are different claims.**
> When adding several loss terms, always **match the reduction axes first.**

**However, this section's conclusion applies only to this env.** Don't take "sum + β→4 dies" as a general law
— **there is an ancestor implementation that actually walked with the same sum + β→4.** What was different is
revisited with measurements in the [The real source has more switches](#the-real-source-has-more-switches) section below.

## What the paper fixes vs. what we chose

This separates which of this section's content is **knowledge from the paper and which is this implementation's choice**.
What to memorize differs from what to know as "that's how our code is".

The dividing line is this.

```
What the paper writes  =  the formulas. Which terms exist, and where coefficients attach
What the paper leaves  =  the conventions that turn those formulas into 'a single number' a computer handles
                          (which axis each term is reduced over, which number goes into β)
```

`MSE(ô, o)` is complete as a mathematical statement, but unless it says whether to **sum or average** the 45-dim error vector,
it doesn't become code. That's exactly the spot the paper leaves blank.

| Item | What the paper (arXiv 2301.10602) **writes in its formulas** | What this implementation **chose** |
|---|---|---|
| `L_CE = L_est + L_VAE` (eq.5) | **Explicit.** Simply added without coefficients = both have coefficient 1. Since the same paper writes `β` in eq.7, it shows itself that **it does put a weight where it wants one** | 1:1 as-is. But what's pinned down is `L_est : L_VAE` **at the symbol level** — how large a *number* the two chunks actually become is set by the reduction conventions below |
| `L_VAE = MSE + β·D_KL` (eq.7) | **Explicit.** Including that there are two terms and that **β attaches only to the KL side** | As-is |
| "β is **constant**" | **Not a sentence from the paper.** The paper gives β neither a number nor a schedule (the word "constant" doesn't even appear) | **Our reading** — read as "no schedule was written, so it's constant" and implemented as a constant. Reasonable, but **an interpretation, not a quote** |
| **The axis that reduces each term to one number** | **Blank.** eq.7 doesn't say which axis to reduce over. But `D_KL(q‖p)` **itself**, for a diagonal Gaussian, is defined as a 16-dim **sum**, so there's no choice | **We chose** — per-dim mean for all three. The KL's `mean(1)` is 1/16 of the defined value. But as shown below, **it's one body with β, not an independent degree of freedom** |
| **The value of β** | **No number** | **We chose** — the cfg's `beta=0.35` (the cfg overrides the class default 1.0). "The paper said β=1.0" is a **false statement** |
| **The `vel : recon` ratio** | **Doesn't follow from eq.5.** eq.5 only ties `L_est : L_VAE`, and `recon` is one term **inside** `L_VAE` | **We chose** — using `nn.MSELoss()` (element mean) identically on both gives 1:1 per dim. Matching as sums gives 3:45 = **1:15**. A choice relying on convention, not a sentence from the paper |

### The KL axis and β must not be counted separately

Rows 4 and 5 of the table above are **a single degree of freedom**. One line of multiplication shows it.

```
β · KL_mean  =  β · (KL_sum / 16)  =  (β / 16) · KL_sum

written the other way  β_mean = 16 · β_sum          (β when writing the same loss on the mean axis)
```

**Why both directions are written down**: changing the axis mean → sum **divides** β (÷16),
and changing sum → mean **multiplies** β (×16). Memorize only one direction and you'll get it wrong going the other way.

That is, the current code, using `mean(1)` with β=0.35, is **exactly the same loss** as code using `sum(1)` with
**β = 0.35 / 16 = 0.021875**. Same gradient, same training.

So "we changed the KL axis to mean" is **not a departure from the paper.** It's only a question of which unit β is written in,
and since the paper gave β no number, it's within the paper whichever unit you use.

#### The same β written in four units

Several numbers come up and it gets confusing, so they're gathered in one table, each tagged with **which unit of β it is**.
Each row is the same loss function; each column is the same notation convention.

| Which unit β is written in | Old code (`sum(1)`, annealing) | Now (`mean(1)`) |
|---|---|---|
| **Value actually written in the cfg** (`vae.beta`) | 1.0 → up to 4.0 by annealing | **0.35** |
| β when the KL is written as a **sum** | 1.0 → 4.0 (= the cfg value as-is) | 0.35 / 16 = **0.021875** |
| β when the KL is written as a **mean** | 16 → **64** | **0.35** (= the cfg value as-is) |
| Textbook sum/sum **effective β** (converted with recon as a sum too) | 45 → **180** | (45/16)·0.35 ≈ **0.98** |

How to read it: **the cfg row and the "axis that code uses" row are the same number**. The old code used `sum`, so
the cfg's 4.0 was the sum-unit β; now it uses `mean`, so the cfg's 0.35 is the mean-unit β.
All the other cells are that number rewritten in another unit.

> Then what was the earlier "Reduction axis" section about — what really changed there was **the effective size of β**.
> The bottom row of the table above shows **180 → 0.98**, a drop of about **183x**.
> That 183x wasn't made separately by the axis and by β; it's **one value made by the two together**.
> (Read in mean units it's 64 → 0.35, again 183x — the same event written in different units.)
>
> **Don't confuse it with the cfg edit history.** The actual history of the number written in the cfg is
> `1.0 (+ anneal → 4.0)` → `1.0` → `0.35`, and `64` and `180` are **converted values nobody ever
> typed**. 16 · 45/16 · 45 are all unit-conversion factors, not separate events.

**There's exactly one axis β can't absorb: `vel : recon`.** β attaches only to the KL
(that's how eq.7 writes it), so the ratio between `vel` and `recon` doesn't move however much you turn β.
Change **only** `recon` to a sum and it becomes 1:1 → **1:45** (change both to sums and it's 3:45 = 1:15),
and there's no knob inside the loss formula to undo it.

#### Corollary — "so does the axis not matter at all?"

No. That only holds **when you move the axis and β together**.

The axis itself is notation, but **changing only the axis while keeping β fixed in the cfg** moves the effective pressure by exactly 16x.
That's exactly what `DWQ_CENET_KL_REDUCTION=sum` does — leaving β at 0.35 and pushing only the axis
to `sum` makes it **β = 0.35 × 16 = 5.6** in mean units. It's like pushing the "Now" column of the table above
up one row, and it's **a change of settings, not a change of notation**.

> Summary: "the axis isn't a degree of freedom" = it's the same model **if you move β along with it**.
> "I changed the axis and the result changed" = it means you **didn't move** β.

One practical habit follows from this — **the `Loss/cenet_kl` printed in the logs is not the KL.**
What that tag holds is `kl_loss`, i.e. `β · KL_mean = (β/16) · KL_sum`.
At β=0.35, if `KL_sum` is 4.0 nats, the log shows **0.0875** (a number about 45x smaller).
**When quoting a KL value, always say which axis it's on and whether β is multiplied in or not.**

To sum up — **the paper pinned down the formulas (which terms exist and where coefficients attach), and left blank the conventions
that turn those formulas into a single number (the reduction axes) and the value of β.** And in that blank space the substantive
choices are not three but **two** — **the `vel : recon` axis** and **β** (= KL axis × β value).
Those two **decided whether training works or not.**

> **This document was wrong once too.** An earlier version wrote "the `vel : recon` 1:1 also has grounds in eq.5".
> eq.5 doesn't split `L_VAE` into recon and KL, so that was **a false citation**. It also wrote "β is constant"
> as stated in the paper, and the KL axis as an independent degree of freedom — both were overstatements.
>
> **The same sentence was in the previous exercise too.** [`task01_vae`](../task01_vae/) said "annealing was also
> removed (β in paper eq.7 is constant)" — i.e. this course was structured to **teach the false sentence twice first
> and then overturn it here**. Now the earlier one has been fixed as well. When there are several documents,
> **a correction must be made not in one original place but everywhere that sentence was duplicated**.
> — **Before writing "it's in the paper", pin down which line of the original that sentence is.** If you can't, it's
> an interpretation, not a quote, and must be written as an interpretation.

## The real source has more switches

This exercise's `starter/cenet.py` was cut as-is from the production `cenet.py`, but **one chunk was
deliberately removed.** The real source has several experiment switches turned on by environment variables.

```bash
DWQ_CENET_BETA=0.35            # KL weight
DWQ_CENET_FREE_BITS=0.05       # per-dim KL floor (free bits)
DWQ_CENET_KL_REDUCTION=sum     # KL reduction axis as 'sum'
DWQ_CENET_BETA_ANNEAL=1        # beta *= 1.01
DWQ_CENET_GRAD_CLIP=0          # turn off gradient clipping
DWQ_CENET_REFERENCE=1          # set the four above to the IsaacGym original values at once
```

> **`KL_REDUCTION=sum` and `BETA` are the same knob** (when `free_bits=0`). Pushing to `sum`
> is exactly the same as multiplying β by **16** — for why, see the "The KL axis and β must not be counted separately" section above.
> So when turning both on, it's easy to get confused about **what you're measuring**.

**All are off by default.** With none turned on, the formulas trained are exactly the same as what you filled in in this exercise.
There are two reasons they were removed. One is that a 55-line gate comment right above the TODO mixes what to read
with what you don't need to read; the other is that the helper function the gates use (`_kl_term_gated`)
**contains exactly the KL formula you must fill in now**, making it an answer key.

While removing the gates, **the settings only the gates read** (`free_bits`, `kl_reduction`, `beta_anneal`,
`beta_limit`) and the gate-only module (`_DecoderVelGate`) were removed too. In particular, if `self.kl_reduction`
were left in `__init__`, then at the very moment you decide "which axis to reduce the KL over" in the TODO it would look like
you should write an `== "sum"` branch — when in fact it's a value nobody reads in this exercise.
The only setting the code below actually reads from the starter's `__init__` is `self.grad_clip`.

### But `DWQ_CENET_KL_REDUCTION=sum` is the very axis said to be wrong above

Right. So let's address it.

This project's **IsaacGym ancestor implementation** (`dreamwaq/rsl_rl/rsl_rl/vae/cenet.py`) wrote the KL as a **sum**
and grew β up to 4. And that implementation **actually walked.** Then our conclusion that "the sum was wrong"
becomes odd — so that switch was made to measure it again. The paper doesn't fix the reduction axis,
so which one is right is a question to settle **by experiment, not argument**.

We measured it, and the results are these (rough Waq, 600 it, `PAPER.md` §4).

| Setting | `\|mu\|` | `KL_sum` (nats) | Active latent dims | Verdict |
|---|---|---|---|---|
| Default (mean, β=0.35) | **0.215** | 4.18 | 5 / 16 | Lives |
| `DWQ_CENET_REFERENCE=1` (sum + β→4) | 0.008 | 0.0005 | **0 / 16** | **Dies** |

> `KL_sum` is the **16-dim sum** and the value **before multiplying by β** (the diagnostic `cenet_kl_nats` tag).
> Don't confuse it with `cenet_kl = 0.030` in the "Was the fix right" section above — that one is `β·KL_mean`, so
> it's a number about 45x smaller even in the same state.

**In the current env the effective β must be low** — more precisely, not "the mean is right" but
"`sum + β→4` (= β 64 in mean units) dies in this env". Even writing the same loss as a `sum`,
if β is moved along to 0.021875, you get **exactly the same result** as row 1 above.
And the reason the original lived was uncovered too — not because its loss settings
were good but **because the original env gave z a lot of work to do** (there was a disturbance pushing
±1 m/s every second). A latent variable lives only if it has work to do, and **how hard you can press on it depends on
the amount of that work.** The right answer for a loss function is only half an answer if it doesn't mention the env.

> If you want to try turning the gates on yourself, use the production source. It's about running training, so it needs a GPU,
> and when reporting results, **write in `PAPER.md` which gates you turned on** — never present numbers produced with them on
> as if they were numbers for the default settings.

## Next

- [`runner-augment`](../runner-augment/) — where the actor observation is augmented 45 → 64 (L2)

If you skipped the earlier steps, do [`task01_vae`](../task01_vae/) (L3) → [`cenet-forward`](../cenet-forward/) (L2)
first, in that order. The map of the whole stage is in the [Stage 4 README](../README.md).
