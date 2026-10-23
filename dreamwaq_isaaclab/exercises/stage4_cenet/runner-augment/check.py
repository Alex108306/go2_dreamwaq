#!/usr/bin/env python3
"""Fast check — runner-augment.

Runs without Isaac Sim. It detaches `_build_actor_obs` from the class and calls it with a fake self
(launching the whole runner would need an env, but this method only uses a few things from self).

    python check.py                 # check starter/dreamwaq_runner.py
    python check.py --solution      # check the finished version (to test the checker itself)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/dreamwaq_runner.py"
)
STARTER = HERE / "starter" / "dreamwaq_runner.py"

NUM_BASE, NUM_VEL, LATENT, NUM_SCAN, BATCH = 45, 3, 16, 187, 8
NORM_GAIN = 2.0  # fake normalization: only the base observation is scaled by this factor → you can see where it was applied


def load_build_actor_obs(path: Path):
    """Load the module and return the *unbound* `_build_actor_obs`."""
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id runner-augment")
    spec = importlib.util.spec_from_file_location(f"_runner_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.OnPolicyRunnerWaq._build_actor_obs


class FakeSelf:
    """A fake runner with only what `_build_actor_obs` actually touches."""

    def __init__(self, use_exteroception: bool):
        self.use_exteroception = use_exteroception
        self.num_base_obs = NUM_BASE
        # The flag read by the given NaN warning block. Setting it to True means the warning is never printed,
        # keeping the check output clean (the warning itself isn't graded in this exercise).
        self._nan_warned = True

    def _norm_base(self, base_obs):
        return base_obs * NORM_GAIN

    def _extract_extero(self, src_obs):
        return src_obs[:, NUM_BASE:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    build = load_build_actor_obs(path)

    torch.manual_seed(0)
    base = torch.randn(BATCH, NUM_BASE)
    vel = torch.randn(BATCH, NUM_VEL)
    ctx = torch.randn(BATCH, LATENT)
    src_flat = base.clone()
    src_rough = torch.cat([base, torch.randn(BATCH, NUM_SCAN)], dim=-1)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> bool:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)
        return cond

    def skip(name: str, why: str) -> None:
        """If an earlier check fails, later checks can't be judged — counting them as FAIL would be a misdiagnosis."""
        print(f"  [\033[1;33mSKIP\033[0m] {name}  — {why}")

    # --- 1. flat: 45 + 3 + 16 = 64 ----------------------------------------------
    try:
        out = build(FakeSelf(use_exteroception=False), base, vel, ctx, src_flat)
    except NotImplementedError:
        print("  TODO(runner-augment) is still empty. Fill in starter/dreamwaq_runner.py.")
        return 1
    check("1. on flat the actor observation is 64-dim (45+3+16)",
          tuple(out.shape) == (BATCH, NUM_BASE + NUM_VEL + LATENT),
          f"actual {tuple(out.shape)}")
    if tuple(out.shape) != (BATCH, NUM_BASE + NUM_VEL + LATENT):
        print("\n\033[1;31mThe dimensions differ, so the remaining checks can't proceed.\033[0m")
        return 1

    # --- 2. Order: [norm(base), vel, context] --------------------------------------
    # Checks 4·5 poke specific positions with fixed indices. If the order is wrong they misdiagnose "what went wrong"
    # (e.g. reporting an order bug as "nan_to_num was forgotten"), so if this fails they are skipped.
    order_ok = check("2. the first 45 are the normalized base observation",
                     torch.allclose(out[:, :NUM_BASE], base * NORM_GAIN, atol=1e-6),
                     "base wasn't normalized, or the order differs")
    order_ok &= check("2. the next 3 are vel_input (not normalized)",
                      torch.allclose(out[:, NUM_BASE:NUM_BASE + NUM_VEL], vel, atol=1e-6),
                      "normalization was applied to vel too, or context was concatenated first")
    order_ok &= check("2. the last 16 are context_vec (not normalized)",
                      torch.allclose(out[:, NUM_BASE + NUM_VEL:], ctx, atol=1e-6),
                      "normalization was applied to context, or its order was swapped with vel")

    # --- 3. rough: exteroception is appended at the very end ---------------------------------
    out_r = build(FakeSelf(use_exteroception=True), base, vel, ctx, src_rough)
    want_r = NUM_BASE + NUM_VEL + LATENT + NUM_SCAN
    check("3. on rough, height_scan (187) is appended to make 251 dims",
          tuple(out_r.shape) == (BATCH, want_r), f"actual {tuple(out_r.shape)}")
    if tuple(out_r.shape) == (BATCH, want_r):
        check("3. height_scan is appended as-is at the very end",
              torch.allclose(out_r[:, NUM_BASE + NUM_VEL + LATENT:], src_rough[:, NUM_BASE:], atol=1e-6))
        check("3. the first 64 are the same as on flat",
              torch.allclose(out_r[:, :NUM_BASE + NUM_VEL + LATENT], out, atol=1e-6))

    # --- 4. Don't let NaN / inf through -----------------------------------------
    # If CENet diverges, NaN goes straight into the actor and the whole policy is ruined.
    bad_vel = vel.clone()
    bad_vel[0, 0] = float("nan")
    bad_vel[1, 1] = float("inf")
    bad_vel[2, 2] = float("-inf")
    out_bad = build(FakeSelf(use_exteroception=False), base, bad_vel, ctx, src_flat)
    # Only this item is position-independent — it passes as long as it's finite, wherever it was applied.
    check("4. no NaN/inf remains", bool(torch.isfinite(out_bad).all()),
          "torch.nan_to_num is missing")

    blocked = "can't be judged because an earlier check (2. order) failed — this check pokes fixed positions"
    if order_ok:
        check("4. nan → 0.0", float(out_bad[0, NUM_BASE]) == 0.0, f"actual {float(out_bad[0, NUM_BASE])}")
        check("4. +inf → 10.0", float(out_bad[1, NUM_BASE + 1]) == 10.0, f"actual {float(out_bad[1, NUM_BASE + 1])}")
        check("4. -inf → -10.0", float(out_bad[2, NUM_BASE + 2]) == -10.0, f"actual {float(out_bad[2, NUM_BASE + 2])}")
    else:
        skip("4. nan → 0.0 / +inf → 10.0 / -inf → -10.0", blocked)

    # --- 5. Normal values are not clamped ---------------------------------------------
    # nan_to_num only touches NaN/inf. Cutting even finite large values distorts the learning signal.
    if order_ok:
        big_ctx = torch.full((BATCH, LATENT), 12345.0)
        out_big = build(FakeSelf(use_exteroception=False), base, vel, big_ctx, src_flat)
        check("5. finite large values pass through unchanged (not a clamp)",
              torch.allclose(out_big[:, NUM_BASE + NUM_VEL:], big_ctx, atol=1e-6),
              "you may have used torch.clamp — it should be nan_to_num")
    else:
        skip("5. finite large values pass through unchanged (not a clamp)", blocked)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m You've finished all the Stage 4 exercises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
