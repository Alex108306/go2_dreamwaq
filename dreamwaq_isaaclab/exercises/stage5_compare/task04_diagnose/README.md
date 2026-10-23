# task04_diagnose — When the latent dies, tell **what died first**  `L2 · BUILD`

**Stage 5 · Comparing results**

## Goal

task03 said "if the ordering doesn't come out, that's a result too". **So how do you find the cause?**

This project's CENet latent variable `z` (16-dim) **died twice, and the two mechanisms were different.**
And the two look completely identical from the outside — `|mu| → 0`, `sigma → 1`, `KL → 0`.

The prescriptions are opposite. One lives if β is lowered; the other doesn't live however much β is lowered.
**So if you pick a prescription by appearance alone, you're wrong half the time.** This exercise measures a second metric
that separates two states with the same appearance.

## The two mechanisms

CENet's decoder takes `[v̂(3), z(16)]` and reconstructs the next step's 45-dim observation (paper Fig.2).
Here, **the only force keeping `z` alive is the reconstruction loss** — the policy gradient doesn't reach the encoder
(Fig.1), and only `v̂` is separately supervised by `vel_loss`. So it dies in two ways.

```
(a) KL overpressure                   (b) The decoder drops z
    ────────────                    ─────────────────────
    Decoder: wants to read z              Decoder: doesn't read z. Reads only v̂
             (|W_z| is large)                      (|W_z| → 0, only |W_vel| grows)
    Encoder: frozen under KL pressure     Encoder: with no recon gradient coming to z,
                                                   it just dies even under weak KL pressure
    Fix:     lower β                      Fix:     lowering β doesn't help.
                                                   z has to be given **work to do**
```

Why does (b) happen? `v̂` learns directly from the ground truth (actual base velocity), so it beats `z` in the
reconstruction competition. From the decoder's point of view, reading only the reliable `v̂` always pays, and once it lets go
of `z`, the gradient to `z` is cut and never comes back. It's **a condition the paper's structure sets up**, not
a mistake in our implementation — the decoder receiving `[v̂, z]` together is drawn in paper Fig.2
(`PAPER.md` §4).

**But it's a "condition", not an "outcome".** There are runs with the same structure where `z` lived just fine (the
`alive` row of the measurement table below). Whether the condition actually goes all the way to collapse is decided by **β and the env** — the
"Why randomization killed z" section below confirms this in both directions. It's not "following the paper always
collapses".

## So what to measure — one more for the encoder, one for the decoder

### Encoder: does the output move when the input changes

A VAE's posterior collapse looks like this.

```
q(z | o^H)  →  N(0, I)          emits the same distribution whatever the input o^H
```

That is, **mu is pressed to 0 and sigma to 1.** Then the 16 context dims the actor receives are
**pure Gaussian noise drawn fresh every step**. The information content is 0 bits.
It becomes natural that `Waq` can't beat `Base` — it's like padding the observation with 19 dims of noise.

No simulator is needed to check. The encoder is just an MLP, so **feed it any input and
see whether the output moves**. If it doesn't move, it's dead.

> The reason to feed N(0,1) instead of real observations: what's measured is **input sensitivity, not accuracy**.
> Observations go in normalized, so N(0,1) is a plausible input distribution, and above all
> **the same yardstick can be applied to any checkpoint without a simulator**.

### Decoder: does it even read z in the first place

This one doesn't even need a probe. **Just look at the weights.**

The decoder's first-layer weight `W` is `[hidden, 19]`. All of input `j`'s influence on what follows
passes through **column `j`** of `W`. So

```
||W[:, j]||  =  how much the decoder reads from input j
```

and since the input is `[v̂(3), z(16)]`, the mean of the first 3 columns is `|W_vel|` and the mean of the last 16 columns is `|W_z|`.
If a column is 0, that input has **no influence at all** on the decoder.

This pair separates (a) from (b).

| | `\|mu\|` | `\|W_z\|` | How to read it |
|---|---|---|---|
| Healthy | large | large | The encoder speaks and the decoder listens |
| (a) KL overpressure | **0** | large | It wants to listen, but the speaker has frozen |
| (b) Decoder dropped z | **0** | **0** | Nobody listens. So there's no reason to speak either |

## What to fill in

```
starter.py  →  TODO(stage5-collapse-probe)  inside  diagnose_checkpoint()
```

```python
def diagnose_checkpoint(cenet_state_dict: dict,
                        num_samples: int = 2048,
                        seed: int = 0) -> dict:
```

| Argument | |
|---|---|
| `cenet_state_dict` | The checkpoint's `cenet_state_dict`. `encoder.*` and `decoder.*` keys are mixed in |
| `num_samples` | Number of encoder probes |
| `seed` | Probe RNG seed. The same seed must give **the same numbers** |

Returns — the six numbers are Python `float`s; only `mechanism` is a string.

| Key | Meaning | On collapse |
|---|---|---|
| `mu_abs` | Mean of \|mu\| | **pressed to 0** (key metric) |
| `sigma` | Mean of `exp(0.5*logvar)` | pressed to 1 |
| `kl` | `KL(q‖N(0,I))`, 16-dim **sum**, sample mean (nats) | 0 |
| `est_vel_std` | Mean **sample-axis** std of the 3 estimated-velocity dims | doesn't move when the input changes |
| `dec_w_vel` | Mean column norm of the **first 3 columns** of the decoder's first-layer weight | (grows, if anything) |
| `dec_w_z` | Mean column norm of the **last 16 columns** of the decoder's first-layer weight | becomes 0 only in mechanism (b) |
| `mechanism` | `alive` / `kl_overpressure` / `decoder_dropped_z` / `nan` | |

`build_encoder()` (state_dict → `nn.Sequential`), `decoder_input_weight()` (the decoder's first Linear
weight), `probe_run()`, `verdict()`, `find_runs()` already exist.
The only thing to fill in is `diagnose_checkpoint()`.

### The verdict order is the logic

```
1. if NaN                → "nan"                  (the 'NaN' section below)
2. if |mu| is large      → "alive"
3. if |W_z|/|W_vel| is large → "kl_overpressure"  wants to listen but can't speak
4. otherwise             → "decoder_dropped_z"    nobody listens
```

There's a reason 2 comes **before** 3 and 4. What the actor receives is not the decoder's output but
the `context` made by the encoder. Even if `|W_z|` has died, as long as `|mu|` is alive the actor is still
receiving information — the death of `|W_z|` is **a sign that "it will die soon"**, not that it's already dead.
(That's why watching `Loss/cenet_dec_w_z` during training is an **earlier alarm** than watching `|mu|`.)

## Don't conclude "health" from a threshold

This exercise has two decision lines (`MU_ALIVE = 1e-2`, `Z_USED = 1e-2`). Both are **reading aids**,
not truth. This is **why the checker doesn't grade real checkpoints**.

A previous version of this exercise hard-coded "this folder is healthy, that folder collapsed" by file path, and to pass,
the thresholds had to separate the two. But **a run that had been labeled healthy later collapsed.** Then
no threshold could satisfy both conditions at once — one side required `≤ 6.3e-4` and the other
`> 4.2e-3`. The exercise became impossible to pass.

The lesson is not to fix the thresholds.

> **"This run is healthy" is not a fact you can read from a file path.** It's something this script
> is meant to judge, not a correct answer to bake into the script in advance.

So the checker now **grades only hand-made synthetic checkpoints.** For synthetic weights the correct answers are
known analytically, and the values can be kept **more than an order of magnitude** away from the decision lines, so **the result is the same however
the thresholds are set within a sensible range** (the concrete ranges are measured in the table in the "Checks" section below).
Real checkpoints are **only printed at the end, not graded.**

## Judge by the **last** checkpoint

This is the counterpart to task02's 'averaging window' lesson. Collapse happens not right after initialization but
**during training**. The archived collapsed runs all had a healthy `|mu|` of 3e-2 ~ 1e-1 at `it 0`.
**Had you looked only at the first checkpoint, you'd have written that all three were healthy.**

Putting the `|W_z|` trajectories side by side shows where they diverge (rough Waq, same probe).

| it | no randomization (`\|W_z\|`) | with randomization, β=1.0 (`\|W_z\|`) |
|---:|---|---|
| 0 | 1.037 | 1.041 |
| 500 | 0.164 ↓ | **0.0114** ↓↓ |
| 1000 | 0.202 ↑ | 0.0066 |
| 2000 | 0.233 ↑ | 0.0045 |
| 3999 | **0.259** ↑ | **0.0044** |

**Both runs switch off once at it 500.** What separates them is whether it **switches back on** afterwards.
One recovers and increases monotonically; the other stays dead.

## Measured — the four archived runs

All with `num_samples=2048, seed=0`, on each run's **last** checkpoint. These numbers come out again exactly
when you run this `check.py` now (if you have the assets).

> **This table is based on the instructor's assets.** The three rows come from `_archive_3000it_stairheavy` /
> `_archive_4000it_clip1_norew` / `_archive_4000it_collapsedwaq` respectively, and all three are
> under `logs/`, so **they may not be in the distributed archive** (see the list of distributed assets in the Stage 5 README).
> If they're missing, those rows just won't be printed, and **this exercise passes with synthetic checkpoints alone, without the assets.**

| Run (rough Waq) | Loss setting | Randomization | `\|mu\|` | `\|W_vel\|` | `\|W_z\|` | `W_z/W_vel` | Mechanism |
|---|---|:---:|---|---|---|---|---|
| `_archive_3000it_stairheavy` | KL 16-dim **sum** + β→4 (effective β 180) | ✗ | 4.17e-3 | 4.39 | **1.91** | 4.4e-1 | **kl_overpressure** |
| `_archive_4000it_clip1_norew` | KL mean, β=1.0 | ✗ | 2.87e-2 | 9.90 | 0.259 | 2.6e-2 | **alive** |
| `_archive_4000it_collapsedwaq` | KL mean, β=1.0 | **✓** | 6.28e-4 | 10.24 | **0.0044** | 4.3e-4 | **decoder_dropped_z** |

**Three rows, three outcomes.** And looking only at `|mu|`, rows 1 and 3 look like the same kind
(both below 1e-2). What separates them is the single `|W_z|` column — 1.91 vs 0.0044, **430x**.

## Why randomization killed z — this course's last lesson

The only thing changed in row 3 is **turning on domain randomization** (`PAPER.md` §6, §4).
Confirmed by ablation: neither the clip nor the reward terms were to blame; **remove only the randomization and z lives.**

The mechanism is this. What randomization shakes is friction, payload, CoM, joint gains and motor strength — all
**env constants** set once at the start of an episode.

- What `z` can see is only **a 0.1-second (5-frame) observation history**. Back-estimating a friction coefficient from that short window
  is practically impossible → **the amount z can contribute to reconstruction shrank.**
- At the same time, **the unpredictable component of `o_{t+1}` grew** → the reconstruction loss itself got noisier.

As the recon gradient to `z` weakened, it couldn't even beat the KL pressure of β=1.0, and the decoder
switched over to the reliable `v̂`. That's mechanism (b).

> **z lives only if it has work to do. How hard you can press on it depends on the amount of work.**

This sentence is confirmed in both directions.

- Moving the IsaacGym original settings (KL sum + β→4, effective β 180) **as-is into the current env kills it**
  (prescription cell `DWQ_CENET_REFERENCE=1`: `|mu|` 0.008, 0 active dims). The original lived not because its loss
  settings were good but **because the env gave z a lot of work to do** — there was a disturbance pushing ±1 m/s
  every second, and the signals the paper points to as what `z` encodes are exactly disturbances and terrain.
- Conversely, with randomization **still on**, lowering β from 1.0 → **0.35** made z live
  (`|mu|` 0.215, KL 4.2 nats, 5/16 active dims). It was walked again over a 4000-iteration horizon to confirm,
  and so **β 0.35 was adopted as the cfg default**.

The paper doesn't give a β value. So this is not a departure from the paper but **filling a spot the paper left blank
with measurements**. And the conclusion reached is one — **the code and the env are a set.**
The sentence "this loss function is right" is only half a sentence unless it says in which env it's right.

## NaN — the fourth outcome

`nan` is in `mechanism` because it actually happened.

The **flat** Waq run of `_archive_3000it_stairheavy` has **all 14 CENet tensors NaN** from `it 500`.
Yet that run's `Train/mean_reward` rose to 35 and the tracking reward is normal too.
How is that possible — because the runner washes it away when building the actor observation.

```python
# dreamwaq_runner.py:203
return torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=10.0, neginf=-10.0)
```

That is, the run trained for all 3000 iterations as **"Base + 19 zeros"**, and looking only at the curves there was
no problem. `nan_to_num` is a safeguard added to keep training from dying, but in exchange it **made the failure
invisible**.

There's also a practical reason to filter NaN first: **NaN is `False` in every comparison**, so
left alone it tumbles down into the final `else` and gets labeled `decoder_dropped_z`, **a mechanism that doesn't
even exist**. A wrong label is worse than no value.

## Checks

```bash
cd exercises/stage5_compare/task04_diagnose
~/IsaacLab/_isaac_sim/python.sh check.py              # grades synthetic checkpoints only — 1 s, no assets needed
~/IsaacLab/_isaac_sim/python.sh starter.py            # if the assets are there, scans every run it finds
~/IsaacLab/_isaac_sim/python.sh starter.py --run <folder> --run <folder>
```

Pass criteria (synthetic checkpoints only, things that don't depend on the thresholds):

1. Returns the six numbers as Python `float`s and `mechanism` as a string in `MECHANISMS`
2. The encoder probe input is **standard normal** (feeding uniform or constant values doesn't match)
3. `KL = -0.5·Σ(1 + logvar − mu² − exp(logvar))` — a 16-dim **sum** (a mean gives 1/16).
   `sigma` is the mean of `exp(0.5·logvar)` — `logvar` must not be used as-is
4. `est_vel_std` is the mean std over the **sample axis (dim=0)** — over the dimension axis it's a different value
5. `dec_w_vel` / `dec_w_z` are means of **column norms**, and the boundary is **3 / 16**
   (the checker catches a split shifted by one to 4/15)
6. Distinguishes the four mechanisms. And **the verdict order** is right — NaN first, `|mu|` before `|W_z|`
7. The same `seed` gives the same numbers, a different `seed` different samples. **Doesn't pollute the global RNG state**
   (`torch.Generator`, not `torch.manual_seed`)
8. `num_samples` is the real sample count (increasing it should reduce the spread across seeds)

The synthetic values are **at least an order of magnitude** away from the decision lines, so **however `MU_ALIVE` / `Z_USED`
are set within a wide range, the grading result is the same.** Actually shaking them and measuring gives this.

| Constant | Default | Range where all of test 6 passes | Narrowest margin |
|---|---|---|---|
| `MU_ALIVE` | `1e-2` | `0 < MU_ALIVE < 0.81` | The collapsed synthetic's `\|mu\|` is **exactly 0**, so the lower margin is infinite |
| `Z_USED` | `1e-2` | `4.30e-4 < Z_USED ≤ 0.435` | The `decoder_dropped_z` synthetic's `\|W_z\|/\|W_vel\| = 4.30e-4` — **23x (1.4 orders of magnitude)** from the default |

That is, **however `Z_USED` is set within `1e-3` ~ `1e-1`**, the grading result is the same. Outside that
it breaks — with `Z_USED=1e-5`, the `decoder_dropped_z` synthetic gets classified as `kl_overpressure`
and test 6 FAILs. **Not "independent of the threshold" but "there's more than an order of magnitude of margin"**
is the accurate statement, and that's the boundary between what this checker can and can't grade.

## Hints

There are only two inline hints. If you get stuck, unfold these step by step — **failing first and reading
`check.py`'s output** leaves more behind than reading the answer in advance.

> The `<details>` below **only fold in the GitHub web view.** If you open this file in a terminal or editor,
> everything is visible. If you want to try without looking, stop scrolling here.
> The hints give **direction only** — the code and formulas are in `starter.py`'s docstring and the "Pass criteria" above.

<details>
<summary>Step 1 — how to build the encoder probe</summary>

After building the encoder with `build_encoder()`, create `num_samples × INPUT_DIM` standard-normal inputs.

```python
g = torch.Generator().manual_seed(seed)
probe = torch.randn(num_samples, INPUT_DIM, generator=g)
```

Run it inside `torch.no_grad()` — no gradient is needed.
</details>

<details>
<summary>Step 2 — how to split the 35-dim output</summary>

It's the same split you wrote in `cenet-forward`. The encoder output is a single tensor with `est_vel`, `mu`, `logvar`
concatenated in that order — `split([NUM_VEL, NUM_LATENT, NUM_LATENT], dim=-1)`.
</details>

<details>
<summary>Step 3 — where you get stuck on the four encoder metrics is a single thing: 'the axis'</summary>

The formulas are in `starter.py`'s docstring and in "Pass criteria" 3·4 above. Where people go wrong is always the axis.

- `sigma` is the **standard deviation**, not the log variance. Averaging `logvar` as-is measures something else.
- `kl` is averaged over samples **after reducing** over latent dims. Swap the two axes and the value is off by the number of dims.
- `est_vel_std` is "**how much the same output component wobbles depending on the input**". Take the axis the other way and
  you end up measuring "how different the 3 components are from each other within one sample", which means something else — go back to
  what you were trying to measure and pick the axis.

Return all four as Python `float`s.
</details>

<details>
<summary>Step 4 — the two decoder metrics</summary>

`decoder_input_weight(state_dict)` gives the `[hidden, 19]` weight. What you want to measure is
"**how much a single input j carries to what follows**", and as seen in the "Decoder" section above it's all in
**column** `j`. So the shape of the norm result must be `[19]` —
write the shapes out by hand and pick which axis to reduce to get that. If you got `[hidden]`,
you took it the other way.

After splitting, the front is `v̂` and the back is `z`. **Don't hard-code the cut widths as `3` / `16`**; use the
constants at the top of the file.
</details>

<details>
<summary>Step 5 — the order for naming the mechanism</summary>

It's a four-way ladder, and **the order is the logic.** What to ask first is explained, reasons included, by the
"The verdict order is the logic" section and the "NaN — the fourth outcome" section above. Just translate that order
into `if / elif` as-is.

Just two additions.

- NaN is **`False` in every comparison**. So `x > threshold` will never catch it;
  to filter it you must ask "is it finite" separately.
- When looking at the ratio of `|W_z|` to `|W_vel|`, writing it as a division means worrying about a zero denominator.
  Rewriting the same inequality as a multiplication removes that worry.
</details>

## Assets

Checkpoints live under `logs/`, and `logs/` is **gitignored**, so they're not in this repo.
This exercise **passes even without** the assets (the synthetic checks are enough).

`find_runs()` finds experiment folders whose names contain `Waq` under `SEARCH_ROOTS` and
picks the most recent timestamp of each. **Missing folders are silently skipped.**

```
dreamwaq_manager/logs/
├── rsl_rl/Waq-Official-{Flat,Rough}-PPO-v0/<ts>/          latest sweep (β=0.35) — appears here once complete
├── _archive_4000it_collapsedwaq/Waq-.../<ts>/             β=1.0 + randomization → decoder_dropped_z
├── _archive_4000it_clip1_norew/Waq-.../<ts>/              before randomization → alive
└── _archive_3000it_stairheavy/rsl_rl/Waq-.../<ts>/        KL sum + β→4 → rough is kl_overpressure,
                                                          flat is NaN
```

If you extracted them elsewhere or want to look at your own training results, give the folder directly. It can be given multiple times.

```bash
~/IsaacLab/_isaac_sim/python.sh starter.py --run /wherever/you/extracted/<timestamp>
```

> **Only the Waq of `_archive_3000it_stairheavy` ends with `model_5999.pt`.** Why 5999 for a 3000 it run
> — at the time `OnPolicyRunnerWaq` **double-counted** `current_learning_iteration`, so
> only the name of the last checkpoint was wrong (fixed in commit `1b4a92b`). The Base/Oracle in the same folder use the
> stock runner, so they're correctly `model_2999.pt`, and the tfevents are step 0..2999 too. **The weights are from
> it 3000; only the file name is wrong.** Since it's a real asset, the name wasn't changed.
> — This is also an example of "when outputs disagree with each other, suspect the settings and code first".

### Don't compare metrics other than `|mu|` point to point

A collapsed encoder **emits a constant for most inputs and reacts strongly only to very rare inputs.** So
`kl` and `est_vel_std` become heavy-tailed statistics — measured on the 2500 it checkpoint,
**89% of the total `est_vel` variance comes from the top 0.1% of samples**. As a result, even for the same checkpoint

| Probe | `est_vel_std` |
|---|---|
| `num_samples=2048, seed=0` (this exercise's default) | 1.91e-2 |
| `num_samples=2048, seed=2` | 7.05e-2 |
| `num_samples=8192, seed=0` | 4.07e-2 |

it wobbles within an order of magnitude like this. **That's why all numbers in this document are based on `num_samples=2048, seed=0`**
(the table in `PAPER.md` §4 uses the same basis). This is another reason `|mu|` and `|W_z|` are used for the verdict
— `|W_z|` is a **weight**, entirely independent of samples, so it doesn't wobble.

## A side note — other clues left in the checkpoint

Besides `cenet_state_dict`, the checkpoint contains other things that reveal the run's settings.

| Key | `_archive_3000it_stairheavy` | The rest |
|---|---|---|
| `cenet_beta` | **4.0** (β annealing went up to the cap) | 1.0 (constant) |

```python
torch.load(ckpt, map_location="cpu", weights_only=False)["cenet_beta"]
```

and check for yourself. **Whether a setting was actually applied is confirmed by the outputs, not the logs** —
the second time this lesson appears in this course.

## Open questions

- The collapsed run's `est_vel_std` also died along with it, 4.8e-1 → 1.9e-2. The KL doesn't directly
  touch `est_vel`, so **why did it die too?** (Hint: it shares the encoder trunk)
  And why does this **support** rather than **weaken** the diagnosis "the KL is the cause"?
- This probe considers it alive if `|mu|` is large. You can also build an encoder whose `|mu|` is large but
  **independent of the input**. What additional metric would you need to filter out such cases?
  (`synth(logvar_bias=...)` in `check.py` is a hint)
- If you meet a checkpoint where `|W_z|` has died but `|mu|` is still alive, this script calls it `alive`.
  **Will you trust that verdict as-is?** What would you check a few hundred iterations later?
- `nan_to_num` kept training from dying but hid the failure for 3000 iterations.
  **Is it better to remove it, or to keep it and print a warning?** What does each cost?

## Course wrap-up

If you've come this far, you've done the following yourself.

```
Stage 1  stood a robot up in Isaac Sim, moved its joints and read its sensors
Stage 2  assembled a training env from those ingredients (rewards · observations · termination)
Stage 3  implemented PPO yourself and compared it with rsl_rl
Stage 4  built CENet (VAE) and wired it into the actor observation
Stage 5  checked whether it actually produced performance, and found the cause when it didn't
```

The last step is the most important. **Having implemented something and it being effective are different claims,
and being able to say why it isn't effective when it isn't is yet another claim.**
