#!/usr/bin/env python3
"""Fast check — stage5-summary.

Uses only synthetic logs. Needs neither the distributed assets nor Isaac Sim.

    python check.py                 # check starter.py
    python check.py --solution      # check the finished version (to test the checker itself)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

W = 1.5  # TRACK_WEIGHT


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id stage5-summary")
    spec = importlib.util.spec_from_file_location(f"_s5_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    summarize = load_module(path).summarize

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def approx(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    # Logs built so the last 3 are 0.4 / 0.5 / 0.6 (underlying) respectively.
    flat3 = {
        "Base": [9.9, 0.3 * W, 0.4 * W, 0.5 * W],   # mean of the last 3 = 0.4
        "Waq": [9.9, 0.4 * W, 0.5 * W, 0.6 * W],    # = 0.5
        "Oracle": [9.9, 0.5 * W, 0.6 * W, 0.7 * W],  # = 0.6
    }

    try:
        r = summarize(flat3, last_k=3)
    except NotImplementedError:
        print("  TODO(stage5-summary) is still empty. Fill in starter.py.")
        return 1

    # --- 1. Divide the weight back out ---------------------------------------------------------
    check("1. underlying = log value / 1.5", approx(r.get("Base", 0), 0.4),
          f"Base={r.get('Base')} (expected 0.4). You may not have divided by the weight")
    check("1. all three configs are computed",
          approx(r.get("Waq", 0), 0.5) and approx(r.get("Oracle", 0), 0.6),
          f"Waq={r.get('Waq')}, Oracle={r.get('Oracle')}")

    # --- 2. It's the mean of the last last_k (not the last single point) ----------------------------
    spike = {"Base": [0.4 * W, 0.4 * W, 0.4 * W, 0.4 * W, 9.0 * W]}
    r2 = summarize(spike, last_k=3)
    check("2. the final value is the mean of the last last_k",
          approx(r2["Base"], (0.4 + 0.4 + 9.0) / 3),
          f"{r2['Base']:.4f} — you looked at only the last point or took the mean over everything")

    # --- 3. The two deltas -----------------------------------------------------------------
    check("3. Waq-Base = Waq − Base", approx(r["Waq-Base"], 0.1), f"{r.get('Waq-Base')}")
    check("3. Oracle-Waq = Oracle − Waq", approx(r["Oracle-Waq"], 0.1), f"{r.get('Oracle-Waq')}")

    # --- 4. The delta's sign carries the direction -----------------------------------------------------
    # CENet can actually be worse — this project's earlier run (logs/_archive_3000it_stairheavy)
    # actually had Waq-Base = -0.0166. Erasing the sign hides that fact.
    worse = {"Base": [0.6 * W] * 3, "Waq": [0.4 * W] * 3, "Oracle": [0.7 * W] * 3}
    r3 = summarize(worse, last_k=3)
    check("4. if Waq is worse than Base, Waq-Base is negative",
          r3["Waq-Base"] < 0, f"{r3['Waq-Base']:+.3f} — using abs() loses the direction")

    # --- 5. Don't fill missing runs with 0 ------------------------------------------------
    partial = {"Base": [0.4 * W] * 3}
    r4 = summarize(partial, last_k=3)
    check("5. missing configs have no key at all (not filled with 0.0)",
          "Waq" not in r4 and "Oracle" not in r4, f"{sorted(r4)}")
    check("5. no pair, no delta",
          "Waq-Base" not in r4 and "Oracle-Waq" not in r4, f"{sorted(r4)}")

    # --- 6. Empty lists are also treated as 'missing' ----------------------------------------------------
    r5 = summarize({"Base": [0.4 * W] * 3, "Waq": [], "Oracle": [0.6 * W] * 3}, last_k=3)
    check("6. empty lists count as missing",
          "Waq" not in r5 and "Base" in r5 and "Oracle" in r5, f"{sorted(r5)}")
    check("6. without Waq, neither delta exists",
          "Waq-Base" not in r5 and "Oracle-Waq" not in r5, f"{sorted(r5)}")

    # --- 7. Works even with fewer than last_k log entries ------------------------------------------
    r6 = summarize({"Base": [0.45 * W]}, last_k=3)
    check("7. computes with that value even with only 1 log entry", approx(r6["Base"], 0.45), f"{r6.get('Base')}")

    # --- 7b. The default last_k is large enough ------------------------------------------------
    # This metric fluctuates with sd~0.03 per iteration. With a small default, the noise exceeds the difference
    # (in the distributed assets' flat, last-3 gives -0.012 and last-400 gives +0.0007 — the sign flips).
    long_run = [0.9 * W] * 2700 + [0.1 * W] * 300   # first 2700 are 0.9, last 300 are 0.1
    r7 = summarize({"Base": long_run})
    check("7b. the default last_k is 100 or more (using only a few points gets flipped by noise)",
          r7["Base"] < 0.5, f"default mean={r7['Base']:.3f} — looking only at the last 300 points (0.1) it should be near 0.1")

    # --- 8. weight / last_k can be changed via arguments ------------------------------------------
    r7 = summarize({"Base": [1.0, 3.0]}, weight=1.0, last_k=1)
    check("8. the weight and last_k arguments are actually applied", approx(r7["Base"], 3.0),
          f"{r7.get('Base')} (expected 3.0) — you may have hard-coded a constant")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with task03_interpret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
