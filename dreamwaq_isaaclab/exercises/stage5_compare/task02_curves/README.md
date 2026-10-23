# task02_curves — Curves and summary table  `L2 · BUILD`

**Stage 5 · Comparing results**

## Goal

Confirm **with numbers** the differences you **saw with your own eyes** in task01.

Write the core function `summarize()`. It builds comparable metrics from the reward logs and
computes the course's conclusion, the two deltas.

## Why the total reward can't be used as-is

`Train/mean_reward` is **the sum of several reward terms** (the official recipe logs 10 terms).
Even if it went up, you can't tell whether velocity tracking improved or an energy penalty shrank.
On top of that, the scale is arbitrary, so there's no sense of "how well it's doing".

So detach only the velocity-tracking term and **divide the weight back out.**

```
underlying = Episode_Reward/track_lin_vel_xy_exp ÷ 1.5
           = mean exp(-‖v_cmd − v‖² / 0.25)      ∈ [0, 1]
```

Now `1.0` has the absolute meaning of perfect tracking. Without dividing out the weight,
the value shifts whenever the reward design changes and runs can't be compared.

> **Where does 1.5 come from?** All 6 runs of the distributed assets were run with the official recipe (`*-Official-*`),
> and its weight is set by Isaac Lab's `UnitreeGo2RoughEnvCfg`
> (`isaaclab_tasks/.../locomotion/velocity/config/go2/rough_env_cfg.py`).
> In the DreamWaQ recipe of `velocity_env_cfg.py` seen in Stage 2, the term is named
> `track_lin_vel_xy` with weight **1.0** — if you analyze such a run, you must pass `weight=1.0`.
> Check the actual run's value in `logs/.../params/env.yaml`
> (`weight: 1.5`). The comparison table is in `PAPER.md` §2.

## What to fill in

```
starter.py  →  TODO(stage5-summary)  inside  summarize()
```

```python
def summarize(series: dict[str, list[float]],
              weight: float = 1.5,
              last_k: int = 300) -> dict[str, float]:
```

| Argument | |
|---|---|
| `series` | `{"Base": [log values...], "Waq": [...], "Oracle": [...]}` — missing configs are absent or empty lists |
| `weight` | Reward weight. Divide by it to get back to underlying |
| `last_k` | Number of last log entries to average for the final value. **Default 300** — see "Why 300" below |

Returns (actual values from the rough distributed assets):

```python
{"Base": 0.5496, "Waq": 0.5567, "Oracle": 0.5824,
 "Waq-Base": +0.0071, "Oracle-Waq": +0.0257}
```

**Don't even put in the key for items with no data.** Filling with `0.0` draws a run that hasn't been trained yet
as "performance 0", and the figure lies.

## Why 300 — the most important part of this exercise

`underlying` **fluctuates with sd ≈ 0.03** per iteration. The difference we want to measure is ~0.01.
**The noise is larger than the signal.**

Averaging only the last 3 points gives a standard error of the mean of `0.03/√3 ≈ 0.017`, about twice the difference.
Below, the distributed assets (seed 42, 4000 it) were re-measured changing only the window size.
You can reproduce it yourself with `compare_runs.py --last-k K`.

**rough** — the sign doesn't flip, but **the magnitude is inflated 4x.**

| Window size | Base | Waq | Oracle | **Waq−Base** | Verdict |
|---|---|---|---|---|---|
| last 3 | 0.5341 | 0.5643 | 0.5796 | **+0.0302** ± 0.0186 | indistinguishable from noise |
| last 30 | 0.5451 | 0.5536 | 0.5842 | **+0.0085** ± 0.0082 | indistinguishable from noise |
| last 100 | 0.5487 | 0.5603 | 0.5843 | **+0.0116** ± 0.0046 | significant |
| last 400 (default) | 0.5492 | 0.5565 | 0.5822 | **+0.0073** ± 0.0021 | significant |

**flat** — here **the sign flips.**

| Window size | Base | Waq | Oracle | **Waq−Base** | Verdict |
|---|---|---|---|---|---|
| last 3 | 0.9412 | 0.9290 | 0.9402 | **−0.0121** ± 0.0110 | indistinguishable from noise |
| last 400 (default) | 0.9383 | 0.9390 | 0.9460 | **+0.0007** ± 0.0004 | indistinguishable from noise |

Had you written up rough looking only at last-3, you'd have recorded CENet's contribution as **4x the real value**.
For flat you'd have outright written "CENet is harmful". A single window size makes the conclusion.

> This isn't hypothetical; it actually happened in this project. In an archived earlier run
> (`logs/_archive_3000it_stairheavy/`), rough's `Waq−Base` was
> **+0.0398 ± 0.0193** at last-3 and **−0.0166 ± 0.0025** at last-300 — **exact opposites**.
> Worse, last-3 also had `|d| > 2·se` and **got a "significant" verdict** —
> looking at the standard error alongside isn't enough; **the window has to cover the converged section**.
> In that run CENet had collapsed — the cause is tracked down in
> [`task04_diagnose`](../task04_diagnose/).

Average enough points over the converged section, and **report the difference together with its standard error**.
That's why `compare_runs.py` prints `mean ± standard error` together with a "significant / indistinguishable from noise" verdict.
Its default window isn't a fixed 300 but **10% of the run length** (min 50), so it's 400 for a 4000 it run.
Since `summarize()` here is an exercise, its default is fixed at 300.

## Checks

```bash
cd exercises/stage5_compare/task02_curves
python3 check.py           # fast check — synthetic logs only, 1 s (doesn't even need torch)
~/IsaacLab/_isaac_sim/python.sh starter.py   # prints with real numbers if the distributed assets are there
```

`starter.py` reads tfevents, so run it with the bundled python, which has tensorboard.
`check.py` only uses synthetic logs, so the system `python3` works too.

Pass criteria (13):

1. `underlying = log value / weight` — divides the weight back out
2. **The final value is the mean of the last `last_k`** + **the default must be 100 or more**
3. Computes `Waq-Base`, `Oracle-Waq`
4. **If Waq is worse than Base, the delta is negative** — using `abs()` loses the direction
5. Missing configs have no key at all. No pair, no delta
6. Empty lists are also treated as "missing"
7. Works even with fewer than `last_k` log entries
8. The `weight` / `last_k` arguments are actually applied (no hard-coded constants)

## The finished version that also draws figures

`summarize()` only produces the metrics. The full implementation that also builds curves, bars and CSV already exists.

```bash
cd dreamwaq_manager
~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py
```

```
figures/curves_{flat,rough}.png     velocity-tracking curves (3 variants)
figures/reward_{flat,rough}.png     total episode reward curves
figures/terrain_level_rough.png     terrain curriculum promotion
figures/bar_final.png               final velocity-tracking bars
figures/summary.csv                 final values + the two deltas
```

Reading `compare_runs.py` shows where `summarize()` corresponds —
the `final_value` part of `plot_curves()` and `write_summary()`.

## Hints

<details>
<summary>How to read tfevents directly</summary>

It's already in `load_run_series()` in `solution.py` (no need to fill it in).

```python
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
ea.Reload()
vals = [e.value for e in ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")]
```

`size_guidance={"scalars": 0}` means "read everything". The default samples and discards.
</details>

<details>
<summary>How to handle missing configs (pass criteria 5·6·7)</summary>

- If there are fewer than `last_k` values, average **only what's there**. Skip empty lists — `sum([])/len([])` is division by zero.
- Only put in `"Waq-Base"` when **both** `Waq` and `Base` are present. Same for `"Oracle-Waq"`.
- **Don't even put the key** for a missing config into the result dict. Filling with `0.0` draws a run that hasn't
  been trained yet as "performance 0", and the figure lies.
</details>

<details>
<summary>Mean of the last k</summary>

`vals[-last_k:]` gives only as many as there are even if the list is shorter than `last_k`. So no separate length check is needed.
But **empty lists** must be filtered out — `sum([]) / len([])` is division by zero.
</details>

## Next

- [`task03_interpret`](../task03_interpret/) — write what these numbers mean
- [`task04_diagnose`](../task04_diagnose/) — find the cause when the numbers differ from expectations
