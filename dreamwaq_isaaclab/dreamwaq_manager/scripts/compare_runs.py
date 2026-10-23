#!/usr/bin/env python3
"""Build Base / Waq / Oracle comparison figures and summary table (flat + rough).

Reads the tensorboard event files left by rsl_rl and produces course deliverables.

    figures/curves_flat.png          velocity-tracking curves (3 variants)
    figures/curves_rough.png         velocity-tracking curves (3 variants)
    figures/reward_{flat,rough}.png  total episode reward curves
    figures/terrain_level_rough.png  terrain curriculum progress
    figures/bar_final.png            final velocity-tracking bars
    figures/summary.csv              final values + Waq-Base / Oracle-Waq

The key metric `underlying` is the reward log with the weight divided back out.

    underlying = (Episode_Reward/track_lin_vel_xy_exp) / 1.5
               = mean exp(-||v_cmd - v||^2 / 0.25)   in [0, 1]   (1.0 = perfect tracking)

Without dividing out the weight, the value shifts whenever the reward design changes, making comparison impossible.

The two-line conclusion of the course comes from here.

    Waq - Base    = pure contribution of CENet   (same env, only the actor observation differs)
    Oracle - Waq  = remaining gap                (how far the estimate falls short of true privileged info)

Run with Isaac Lab's bundled python (tensorboard is required):

    cd dreamwaq_manager && ~/IsaacLab/_isaac_sim/python.sh scripts/compare_runs.py

Runs not yet trained only produce a warning and are skipped, so you can run this mid-training.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402


def _use_korean_font() -> None:
    """Titles use Korean, so look for a CJK font. If none is found, quietly keep the default font."""
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False  # Korean fonts break the Unicode minus sign


_use_korean_font()

TRACK_TAG = "Episode_Reward/track_lin_vel_xy_exp"
TERRAIN_TAG = "Curriculum/terrain_levels"
REWARD_TAG = "Train/mean_reward"
TRACK_WEIGHT = 1.5  # track_lin_vel_xy_exp weight of the official Go2 recipe (IsaacLab UnitreeGo2RoughEnvCfg).
                   # The DreamWaQ recipe's (velocity_env_cfg.py) track_lin_vel_xy is 1.0 — a different term.

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = REPO_ROOT / "dreamwaq_manager" / "logs" / "rsl_rl"
DEFAULT_OUTDIR = REPO_ROOT / "figures"

# Order = lower bound -> upper bound. Waq must be in the middle so both deltas are visible.
CONFIGS = ["Base", "Waq", "Oracle"]
COLORS = {"Base": "#888888", "Waq": "#1f77b4", "Oracle": "#2ca02c"}
OBS_DIM = {"Base": 45, "Waq": 64, "Oracle": 48}
EXPERIMENTS = {
    "flat": {
        "Base": "BaseDwq-Official-Flat-PPO-v0",
        "Waq": "Waq-Official-Flat-PPO-v0",
        "Oracle": "OracleDwq-Official-Flat-PPO-v0",
    },
    "rough": {
        "Base": "BaseDwq-Official-Rough-PPO-v0",
        "Waq": "Waq-Official-Rough-PPO-v0",
        "Oracle": "OracleDwq-Official-Rough-PPO-v0",
    },
}


def latest_event_file(logdir: Path, experiment: str) -> Path | None:
    exp_dir = logdir / experiment
    if not exp_dir.is_dir():
        return None
    runs = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        events = sorted(run.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
        if events:
            return events[-1]
    return None


def load_scalar(ev: Path, tag: str) -> tuple[list[int], list[float]]:
    ea = event_accumulator.EventAccumulator(str(ev), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def smooth(ys: list[float], n: int) -> list[float]:
    """Moving average. Prevents noisy curves from obscuring the 3-way comparison."""
    if n <= 1 or len(ys) < n:
        return ys
    out: list[float] = []
    win: deque = deque(maxlen=n)
    for y in ys:
        win.append(y)
        out.append(sum(win) / len(win))
    return out


def tail_window(n_points: int, last_k: int | None) -> int:
    """Tail length to average over. Default is the last 10% of the run (minimum 50 points)."""
    if last_k is not None:
        return max(1, last_k)
    return max(50, n_points // 10)


def final_value(ys: list[float], last_k: int | None = None) -> float | None:
    """Mean over the converged section.

    This metric fluctuates with sd ~0.03 per iteration. Averaging only the last few points
    makes the standard error larger than the difference being measured (~0.02), so **the sign flips.**
    In practice last-3 gives Waq-Base=+0.04 while last-100 gives -0.02.
    So the default is the last 10% of the run.
    """
    if not ys:
        return None
    tail = ys[-tail_window(len(ys), last_k):]
    return sum(tail) / len(tail)


def final_stats(ys: list[float], last_k: int | None = None) -> tuple[float, float, int] | None:
    """(mean, standard error of the mean, number of points used). Used to judge whether a difference is noise."""
    if not ys:
        return None
    n = tail_window(len(ys), last_k)
    tail = ys[-n:]
    m = sum(tail) / len(tail)
    if len(tail) < 2:
        return m, float("nan"), len(tail)
    var = sum((v - m) ** 2 for v in tail) / (len(tail) - 1)
    return m, (var / len(tail)) ** 0.5, len(tail)


def plot_curves(logdir: Path, terrain: str, outdir: Path, smooth_n: int,
                last_k: int | None = None,
                stats_out: dict[str, tuple[float, float, int]] | None = None) -> dict[str, float]:
    finals: dict[str, float] = {}
    fig = plt.figure(figsize=(7, 4.5))
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS[terrain][cfg])
        if ev is None:
            print(f"[warn] {terrain}/{cfg}: no run found ({EXPERIMENTS[terrain][cfg]}) — skipping")
            continue
        steps, vals = load_scalar(ev, TRACK_TAG)
        if not steps:
            print(f"[warn] {terrain}/{cfg}: tag {TRACK_TAG} missing — skipping")
            continue
        under = [v / TRACK_WEIGHT for v in vals]
        fv = final_value(under, last_k)
        if fv is not None:
            finals[cfg] = fv
        if stats_out is not None:
            s = final_stats(under, last_k)
            if s is not None:
                stats_out[cfg] = s
        plt.plot(steps, smooth(under, smooth_n), color=COLORS[cfg], lw=1.8,
                 label=f"{cfg} (obs {OBS_DIM[cfg]})  final={fv:.3f}")
    if not finals:  # Don't leave an empty figure before training
        plt.close(fig)
        print(f"[warn] curves_{terrain}: no data — skipping")
        return finals
    plt.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    plt.xlabel("iteration")
    plt.ylabel("underlying tracking  (1.0 = perfect)")
    plt.title(f"PPO {terrain.capitalize()} — Base vs Waq vs Oracle")
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / f"curves_{terrain}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")
    return finals


def plot_reward_curves(logdir: Path, terrain: str, outdir: Path, smooth_n: int) -> None:
    plt.figure(figsize=(7, 4.5))
    any_data = False
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS[terrain][cfg])
        if ev is None:
            continue
        steps, vals = load_scalar(ev, REWARD_TAG)
        if not steps:
            continue
        any_data = True
        plt.plot(steps, smooth(vals, smooth_n), color=COLORS[cfg], lw=1.8,
                 label=f"{cfg} (obs {OBS_DIM[cfg]})  final={final_value(vals):.2f}")
    if not any_data:
        plt.close()
        print(f"[warn] reward_{terrain}: no data — skipping")
        return
    plt.xlabel("iteration")
    plt.ylabel("mean episode reward (sum)")
    plt.title(f"PPO {terrain.capitalize()} — Total reward: Base vs Waq vs Oracle")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / f"reward_{terrain}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def plot_terrain_level(logdir: Path, outdir: Path, smooth_n: int) -> None:
    """Terrain curriculum. The better a policy walks, the harder the terrain it gets promoted to."""
    plt.figure(figsize=(7, 4.5))
    any_data = False
    for cfg in CONFIGS:
        ev = latest_event_file(logdir, EXPERIMENTS["rough"][cfg])
        if ev is None:
            continue
        steps, vals = load_scalar(ev, TERRAIN_TAG)
        if not steps:
            continue
        any_data = True
        plt.plot(steps, smooth(vals, smooth_n), color=COLORS[cfg], lw=1.8, label=cfg)
    if not any_data:
        plt.close()
        print("[warn] terrain_level: no data — skipping")
        return
    plt.xlabel("iteration")
    plt.ylabel("terrain level (curriculum)")
    plt.title("PPO Rough — Terrain curriculum promotion")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = outdir / "terrain_level_rough.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def plot_bars(finals: dict[str, dict[str, float]], outdir: Path) -> None:
    import numpy as np

    terrains = [t for t in ("flat", "rough") if finals.get(t)]
    if not terrains:
        print("[warn] bar_final: no final values — skipping")
        return
    x = np.arange(len(terrains))
    width = 0.25
    plt.figure(figsize=(7, 4.5))
    for i, cfg in enumerate(CONFIGS):
        ys = [finals[t].get(cfg, 0.0) for t in terrains]
        bars = plt.bar(x + (i - 1) * width, ys, width, color=COLORS[cfg], label=cfg)
        for b, y in zip(bars, ys):
            if y > 0:
                plt.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{y:.3f}",
                         ha="center", va="bottom", fontsize=8)
    plt.xticks(x, [t.capitalize() for t in terrains])
    plt.ylabel("final underlying tracking")
    plt.title("PPO final velocity tracking — Base vs Waq vs Oracle")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = outdir / "bar_final.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[ok] {out}")


def write_summary(finals: dict[str, dict[str, float]], outdir: Path,
                  stats: dict[str, dict[str, tuple[float, float, int]]] | None = None) -> None:
    """Write the final values and both deltas to CSV and also print them."""
    stats = stats or {}
    out = outdir / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["terrain", "Base", "Waq", "Oracle", "Waq-Base", "Oracle-Waq"])
        for terrain in ("flat", "rough"):
            f = finals.get(terrain, {})
            b, waq, o = f.get("Base"), f.get("Waq"), f.get("Oracle")
            w.writerow([
                terrain,
                "" if b is None else f"{b:.4f}",
                "" if waq is None else f"{waq:.4f}",
                "" if o is None else f"{o:.4f}",
                "" if (b is None or waq is None) else f"{waq - b:+.4f}",
                "" if (o is None or waq is None) else f"{o - waq:+.4f}",
            ])
    print(f"[ok] {out}")

    def cell(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "  -  "

    print("\n=== Final underlying tracking (mean over converged section) ===")
    print(f"{'terrain':8} {'Base':>7} {'Waq':>7} {'Oracle':>7} {'Waq-Base':>10} {'Oracle-Waq':>11}")
    for terrain in ("flat", "rough"):
        f = finals.get(terrain, {})
        b, waq, o = f.get("Base"), f.get("Waq"), f.get("Oracle")
        d1 = f"{waq - b:+.3f}" if (b is not None and waq is not None) else "   -  "
        d2 = f"{o - waq:+.3f}" if (o is not None and waq is not None) else "   -  "
        print(f"{terrain:8} {cell(b):>7} {cell(waq):>7} {cell(o):>7} {d1:>10} {d2:>11}")

    # Report the standard error too, so you can judge whether a difference is noise.
    for terrain in ("flat", "rough"):
        st = stats.get(terrain, {})
        if not st:
            continue
        print(f"\n--- {terrain} converged-section statistics (mean ± standard error of the mean) ---")
        for cfg in CONFIGS:
            if cfg in st:
                m, se, n = st[cfg]
                print(f"  {cfg:8} {m:.4f} ± {se:.4f}   (mean of last {n} iterations)")
        if "Waq" in st and "Base" in st:
            d = st["Waq"][0] - st["Base"][0]
            se = (st["Waq"][1] ** 2 + st["Base"][1] ** 2) ** 0.5
            verdict = "significant" if abs(d) > 2 * se else "indistinguishable from noise"
            print(f"  Waq-Base   {d:+.4f} ± {se:.4f}  → {verdict}")
        if "Oracle" in st and "Waq" in st:
            d = st["Oracle"][0] - st["Waq"][0]
            se = (st["Oracle"][1] ** 2 + st["Waq"][1] ** 2) ** 0.5
            verdict = "significant" if abs(d) > 2 * se else "indistinguishable from noise"
            print(f"  Oracle-Waq {d:+.4f} ± {se:.4f}  → {verdict}")
        print("  Note: single seed. The standard error only measures 'fluctuation within one run',"
              " not variation across seeds.")

    rough = finals.get("rough", {})
    if all(k in rough for k in CONFIGS):
        ordered = rough["Base"] < rough["Waq"] < rough["Oracle"]
        print(f"\nOn rough, is Base < Waq < Oracle? "
              f"{'Yes — matches the paper ordering' if ordered else 'No — needs interpretation'}")
        print("  Waq - Base   = pure contribution of CENet")
        print("  Oracle - Waq = remaining gap (estimate vs true privileged info)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Base/Waq/Oracle comparison figures and summary table")
    ap.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR,
                    help="rsl_rl log root (default: dreamwaq_manager/logs/rsl_rl)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help="output folder for figures/CSV (default: figures/)")
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window size for curves")
    ap.add_argument("--last-k", type=int, default=None,
                    help="number of last iterations to average for final values (default: 10%% of the run, min 50). "
                         "Small values are swayed by noise and can flip the sign")
    args = ap.parse_args()

    if not args.logdir.is_dir():
        raise SystemExit(f"Log folder not found: {args.logdir}\n  Run training first (run_full_pipeline.sh)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    finals: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    for terrain in ("flat", "rough"):
        stats[terrain] = {}
        finals[terrain] = plot_curves(args.logdir, terrain, args.outdir, args.smooth,
                                      args.last_k, stats[terrain])
        plot_reward_curves(args.logdir, terrain, args.outdir, args.smooth)
    plot_terrain_level(args.logdir, args.outdir, args.smooth)
    plot_bars(finals, args.outdir)
    write_summary(finals, args.outdir, stats)


if __name__ == "__main__":
    main()
