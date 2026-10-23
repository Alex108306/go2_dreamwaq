#!/usr/bin/env python3
"""Fast check — reward-power-distribution.  (no Isaac Sim needed, ~1 s)

    python check.py                 # starter/rewards.py
    python check.py --solution      # finished version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fake_isaaclab import load_module  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/tasks/locomotion/mdp/rewards.py"
)
STARTER = HERE / "starter" / "rewards.py"

NUM_ENVS, NUM_JOINTS = 4, 12


class Cfg:
    def __init__(self, name="robot", joint_ids=None):
        self.name = name
        self.joint_ids = slice(None) if joint_ids is None else joint_ids


class FakeEnv:
    """Mimics only what rewards.py reads: env.scene[name].data.{applied_torque, joint_vel}"""

    def __init__(self, torque: torch.Tensor, joint_vel: torch.Tensor):
        data = type("D", (), {"applied_torque": torque, "joint_vel": joint_vel})()
        asset = type("A", (), {"data": data})()
        self.scene = type("S", (), {"__getitem__": lambda _s, _k: asset})()
        self.num_envs, self.device = torque.shape[0], "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    fn = load_module(path).power_distribution_l2

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    ones = torch.ones(NUM_ENVS, NUM_JOINTS)
    try:
        r = fn(FakeEnv(ones, ones), asset_cfg=Cfg())
    except NotImplementedError:
        print("  TODO(reward-power-distribution) is still empty. Fill in starter/rewards.py.")
        return 1

    check("1. returns a (num_envs,) scalar vector", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")

    # If all joints use the same power, variance 0 -> penalty 0. This is the very definition of the term.
    check("2. 0 when all joint powers are equal", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-8),
          f"{r[0].item():.6f}")

    # Even with large power, still 0 if 'uniform' — it penalizes imbalance, not the total.
    r = fn(FakeEnv(ones * 10.0, ones * 10.0), asset_cfg=Cfg())
    check("3. 0 if uniform, even with large power (looks at imbalance, not the total)",
          torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6), f"{r[0].item():.6f}")

    # Known value: if power per joint is [0..11], var (unbiased) = 13.0, squared = 169.0
    ramp = torch.arange(NUM_JOINTS, dtype=torch.float32).repeat(NUM_ENVS, 1)
    r = fn(FakeEnv(ramp, torch.ones_like(ramp)), asset_cfg=Cfg())
    expected = torch.var(ramp, dim=-1).square()
    check("4. exact value on a known distribution (var^2)", torch.allclose(r, expected, atol=1e-4),
          f"{r[0].item():.4f} vs {expected[0].item():.4f}")

    # Catch a missing square: returning only var differs from the expected value by a sqrt.
    check("5. the variance was squared once more",
          not torch.allclose(r, torch.var(ramp, dim=-1), atol=1e-4),
          "returned var as-is — torch.square is missing")

    # No absolute value: it must be the variance of the signed power.
    signed = ramp.clone(); signed[:, ::2] *= -1.0
    r_signed = fn(FakeEnv(signed, torch.ones_like(signed)), asset_cfg=Cfg())
    r_abs = fn(FakeEnv(signed.abs(), torch.ones_like(signed)), asset_cfg=Cfg())
    check("6. no absolute value used (variance of signed power)",
          not torch.allclose(r_signed, r_abs, atol=1e-4),
          "the sign was killed and the variance of absolute values was measured")

    r = fn(FakeEnv(ramp, torch.ones_like(ramp)), asset_cfg=Cfg(joint_ids=[0, 1, 2]))
    exp3 = torch.var(ramp[:, :3], dim=-1).square()
    check("7. selects joints via asset_cfg.joint_ids", torch.allclose(r, exp3, atol=1e-5),
          f"{r[0].item():.4f} vs {exp3[0].item():.4f}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
