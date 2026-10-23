#!/usr/bin/env python3
"""Fast check — ppo-cfg.

Runs in pure Python without Isaac Sim. Done in 1 second.

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

DESIRED_KL = 0.01
LR0 = 1.0e-3


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id ppo-cfg")
    spec = importlib.util.spec_from_file_location(f"_cfg_{path.stem}", path)
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
    mod = load_module(path)
    adapt = mod.adapt_learning_rate

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def approx(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    try:
        got = adapt(0.05, LR0, DESIRED_KL)
    except NotImplementedError:
        print("  TODO(ppo-cfg) is still empty. Fill in starter.py.")
        return 1

    # --- 1. Lower it if KL is large ------------------------------------------------------
    check("1. if KL > desired*2 then lr / 1.5", approx(got, LR0 / 1.5),
          f"{LR0:.3e} → {got:.6e} (expected {LR0 / 1.5:.6e})")

    # --- 2. Raise it if KL is small ----------------------------------------------------
    got = adapt(0.001, LR0, DESIRED_KL)
    check("2. if KL < desired/2 then lr * 1.5", approx(got, LR0 * 1.5),
          f"{LR0:.3e} → {got:.6e} (expected {LR0 * 1.5:.6e})")

    # --- 3. Don't touch the dead zone ----------------------------------------------
    # Between desired/2 = 0.005 and desired*2 = 0.02. Also check the boundary values.
    for kl in (0.005, 0.008, 0.01, 0.015, 0.02):
        got = adapt(kl, LR0, DESIRED_KL)
        check(f"3. KL={kl} keeps lr", approx(got, LR0), f"{got:.6e}")

    # --- 4. Lower / upper bounds ------------------------------------------------------------
    check("4. doesn't go below the lower bound 1e-5", approx(adapt(0.5, 1e-5, DESIRED_KL), 1e-5),
          f"{adapt(0.5, 1e-5, DESIRED_KL):.6e}")
    check("4. doesn't go above the upper bound 1e-2", approx(adapt(0.0001, 1e-2, DESIRED_KL), 1e-2),
          f"{adapt(0.0001, 1e-2, DESIRED_KL):.6e}")

    # --- 5. Don't raise it if kl_mean == 0 -----------------------------------------
    # rsl_rl's `kl_mean > 0.0` condition. Without it, lr keeps growing even when KL is 0.
    check("5. doesn't raise lr if KL == 0", approx(adapt(0.0, LR0, DESIRED_KL), LR0),
          f"{adapt(0.0, LR0, DESIRED_KL):.6e}")

    # --- 6. Changing desired_kl moves the boundaries along with it ---------------------------------
    # With desired=0.1, KL=0.05 is in the dead zone (0.05~0.2), so it must stay the same.
    check("6. increasing desired_kl puts the same KL into the dead zone",
          approx(adapt(0.05, LR0, 0.1), LR0), f"{adapt(0.05, LR0, 0.1):.6e}")

    # --- 7. Repeated application follows the same trajectory as rsl_rl -----------------------------------
    trace = [0.05, 0.03, 0.02, 0.004, 0.002]
    lr, want = LR0, LR0
    for kl in trace:
        lr = adapt(kl, lr, DESIRED_KL)
        if kl > DESIRED_KL * 2.0:
            want = max(1e-5, want / 1.5)
        elif kl < DESIRED_KL / 2.0 and kl > 0.0:
            want = min(1e-2, want * 1.5)
    check("7. trajectory of successive applications matches the rsl_rl rule", approx(lr, want),
          f"{lr:.6e} vs {want:.6e}")

    # --- 8. Doesn't corrupt the original value (passed only via the return value) -------------------------------
    check("8. returns the value without side effects", isinstance(adapt(0.05, LR0, DESIRED_KL), float))

    # --- 9. log_std clamp (provided function — regression check) --------------------------------------
    import torch

    clamped = mod.clamp_log_std(torch.tensor([-99.0, 99.0, float("nan"), float("-inf")]))
    check("9. clamp_log_std bounds to [-5, 0] and removes NaN",
          bool(torch.isfinite(clamped).all())
          and float(clamped.min()) >= -5.0 and float(clamped.max()) <= 0.0,
          f"{clamped.tolist()}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m You've finished all the Stage 3 exercises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
