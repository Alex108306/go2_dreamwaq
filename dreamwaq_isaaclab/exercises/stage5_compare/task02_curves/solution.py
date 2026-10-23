#!/usr/bin/env python3
"""Stage 5 · task02 — Extract 3-way comparison metrics from tfevents.  `L2 · BUILD`

If the instructor's distributed assets (tensorboard logs) are there, read them; otherwise demonstrate with synthetic data.
The finished version that also draws figures is in `dreamwaq_manager/scripts/compare_runs.py`.

    python solution.py                    # finds the distributed assets automatically if present
    python solution.py --logdir <path>    # if you extracted them elsewhere
"""
from __future__ import annotations

import argparse
from pathlib import Path

TRACK_TAG = "Episode_Reward/track_lin_vel_xy_exp"
TRACK_WEIGHT = 1.5  # track_lin_vel_xy_exp weight of the official Go2 recipe (IsaacLab UnitreeGo2RoughEnvCfg).
                   # The DreamWaQ recipe's (velocity_env_cfg.py) track_lin_vel_xy is 1.0 — a different term.

CONFIGS = ["Base", "Waq", "Oracle"]
EXPERIMENTS = {
    "Base": "BaseDwq-Official-Rough-PPO-v0",
    "Waq": "Waq-Official-Rough-PPO-v0",
    "Oracle": "OracleDwq-Official-Rough-PPO-v0",
}
DEFAULT_LOGDIR = Path(__file__).resolve().parents[3] / "dreamwaq_manager" / "logs" / "rsl_rl"


def summarize(series: dict[str, list[float]], weight: float = TRACK_WEIGHT,
              last_k: int = 300) -> dict[str, float]:
    """Compute each config's final underlying tracking and the two deltas.

    Args:
        series: {config name: list of track_lin_vel_xy_exp log values}. Missing configs are absent or empty lists.
        weight: reward weight. Divide by it to get back to underlying (0~1).
        last_k: number of last log entries to average for the final value. Default 300.

    Why 300: this metric fluctuates with sd ~0.03 per iteration. Averaging only the last 3 points
    makes the standard error of the mean 0.017, larger than the difference being measured (~0.01). In the distributed assets (seed42, 4000 it),
    rough's Waq-Base is +0.030 at last-3 and +0.0073 at last-400, **inflated 4x**, and
    flat is -0.012 at last-3 and +0.0007 at last-400, **the sign flips.**

    Returns:
        {"Base": .., "Waq": .., "Oracle": .., "Waq-Base": .., "Oracle-Waq": ..}
        Items with no data don't even get a key.
    """
    # ex:begin id=stage5-summary level=2 stage=stage5_compare inline_hints=2 task=Compute the underlying metric and the two deltas from the reward logs
    #   hint: underlying = log value / weight — the weight must be divided back out for runs to be comparable
    #   hint: the final value is the mean of the last last_k. Using only a few points lets the noise exceed the difference and flip the sign
    #   hint: if there are fewer than last_k values, average only what's there. Skip empty lists (division by zero)
    #   hint: only put in 'Waq-Base' when both Waq and Base are present
    #   hint: don't put missing configs into the result at all (filling with 0.0 makes the figure lie)
    out: dict[str, float] = {}
    for cfg in CONFIGS:
        vals = series.get(cfg) or []
        if not vals:
            continue
        tail = vals[-last_k:]
        out[cfg] = sum(tail) / len(tail) / weight

    if "Waq" in out and "Base" in out:
        out["Waq-Base"] = out["Waq"] - out["Base"]
    if "Oracle" in out and "Waq" in out:
        out["Oracle-Waq"] = out["Oracle"] - out["Waq"]
    return out
    # ex:end


def load_run_series(logdir: Path) -> dict[str, list[float]]:
    """Read the per-config track reward logs from the distributed assets. Empty dict if absent."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("[warn] no tensorboard — demonstrating with synthetic data")
        return {}

    series: dict[str, list[float]] = {}
    for cfg, exp in EXPERIMENTS.items():
        exp_dir = logdir / exp
        if not exp_dir.is_dir():
            continue
        events = sorted(exp_dir.glob("*/events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
        if not events:
            continue
        ea = event_accumulator.EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        if TRACK_TAG in ea.Tags().get("scalars", []):
            series[cfg] = [e.value for e in ea.Scalars(TRACK_TAG)]
    return series


def demo_series() -> dict[str, list[float]]:
    """Synthetic logs used when there are no distributed assets. Mimics the final values of the rough distributed runs."""
    return {
        "Base": [0.30 * TRACK_WEIGHT, 0.52 * TRACK_WEIGHT, 0.545 * TRACK_WEIGHT, 0.549 * TRACK_WEIGHT],
        "Waq": [0.30 * TRACK_WEIGHT, 0.52 * TRACK_WEIGHT, 0.551 * TRACK_WEIGHT, 0.557 * TRACK_WEIGHT],
        "Oracle": [0.32 * TRACK_WEIGHT, 0.55 * TRACK_WEIGHT, 0.576 * TRACK_WEIGHT, 0.582 * TRACK_WEIGHT],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    args = ap.parse_args()

    series = load_run_series(args.logdir) if args.logdir.is_dir() else {}
    if series:
        print(f"Read from distributed assets: {args.logdir}")
        print(f"  runs found: {', '.join(series)}")
    else:
        print("No distributed assets, so demonstrating with synthetic data (the numbers are just examples)")
        series = demo_series()

    result = summarize(series)

    print("\n=== rough final underlying tracking ===")
    for cfg in CONFIGS:
        v = result.get(cfg)
        print(f"  {cfg:8} {v:.3f}" if v is not None else f"  {cfg:8}   -")

    print("\n=== The two deltas ===")
    for key, label in (("Waq-Base", "pure contribution of CENet"), ("Oracle-Waq", "remaining gap")):
        v = result.get(key)
        print(f"  {key:12} {v:+.3f}   {label}" if v is not None else f"  {key:12}    -    {label}")

    if all(c in result for c in CONFIGS):
        ordered = result["Base"] < result["Waq"] < result["Oracle"]
        print(f"\nIs Base < Waq < Oracle? {'Yes — matches the paper ordering' if ordered else 'No — needs interpretation'}")

    print("\n[PASS] metric computation complete. Draw the figures with dreamwaq_manager/scripts/compare_runs.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
