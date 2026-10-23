#!/usr/bin/env python3
"""Fast check — ppo-gae.  (no Isaac Sim needed, ~1 s)

    python check.py                 # starter.py
    python check.py --solution      # ../ppo.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE.parent / "ppo.py"
STARTER = HERE / "starter.py"


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with make_exercise.py --id ppo-gae")
    spec = importlib.util.spec_from_file_location(f"_ppo_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reference_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """The checker's own independent implementation — compared against the learner's code."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    acc = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nd = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nd - values[t]
        acc = delta + gamma * lam * nd * acc
        adv[t] = acc
    return adv + values, adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    gae = load(path).compute_gae

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    T, N = 8, 4
    torch.manual_seed(0)
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last_value = torch.randn(N)

    try:
        ret, adv = gae(rewards, values, dones, last_value)
    except NotImplementedError:
        print("  TODO(ppo-gae) is still empty. Fill in starter.py.")
        return 1

    check("1. returns two tensors of shape (T, N)", ret.shape == (T, N) and adv.shape == (T, N),
          f"returns={tuple(ret.shape)} adv={tuple(adv.shape)}")

    check("2. returns == advantages + values", torch.allclose(ret, adv + values, atol=1e-6))

    # Compare against the independent implementation
    ref_ret, ref_adv = reference_gae(rewards, values, dones, last_value)
    check("3. values match the independent implementation", torch.allclose(adv, ref_adv, atol=1e-5),
          f"max error {(adv - ref_adv).abs().max():.2e}")

    # With lam=0 it must equal the 1-step TD error exactly.
    _, adv0 = gae(rewards, values, dones, last_value, gamma=0.99, lam=0.0)
    nxt = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    td = rewards + 0.99 * nxt - values
    check("4. with lam=0 it equals the 1-step TD error", torch.allclose(adv0, td, atol=1e-5),
          f"max error {(adv0 - td).abs().max():.2e}")

    # gamma=lam=1, no dones -> advantage is the sum of future rewards - V(s_t)
    _, adv1 = gae(rewards, values, dones, last_value, gamma=1.0, lam=1.0)
    mc = torch.zeros_like(rewards)
    for t in range(T):
        mc[t] = rewards[t:].sum(0) + last_value - values[t]
    check("5. with gamma=lam=1 it equals Monte Carlo", torch.allclose(adv1, mc, atol=1e-4),
          f"max error {(adv1 - mc).abs().max():.2e}")

    # Does done cut the bootstrap — the key trap of this exercise
    d = torch.zeros(T, N)
    d[3] = 1.0
    _, adv_d = gae(rewards, values, d, last_value)
    _, ref_d = reference_gae(rewards, values, d, last_value)
    check("6. cuts the bootstrap at done steps", torch.allclose(adv_d, ref_d, atol=1e-5),
          f"max error {(adv_d - ref_d).abs().max():.2e}")

    # The advantage of the step right before done must change depending on done
    check("7. done actually changes the result", not torch.allclose(adv_d[3], adv[3], atol=1e-6),
          "done=1 but the value is unchanged — (1 - done) wasn't multiplied in")

    # The last step bootstraps with last_value
    other = last_value + 10.0
    _, adv_o = gae(rewards, values, dones, other)
    check("8. the last step uses last_value", not torch.allclose(adv_o[-1], adv[-1], atol=1e-6),
          "changing last_value leaves the last advantage unchanged")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with task02_ppo_update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
