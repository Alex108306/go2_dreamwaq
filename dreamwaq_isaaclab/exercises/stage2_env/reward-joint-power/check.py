#!/usr/bin/env python3
"""Fast check — reward-joint-power.  (no Isaac Sim needed, ~1 s)

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
    fn = load_module(path).joint_power_l1

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
        print("  TODO(reward-joint-power) is still empty. Fill in starter/rewards.py.")
        return 1

    check("1. returns a (num_envs,) scalar vector", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. torque=1, angular velocity=1, 12 joints → 12", torch.allclose(r, torch.full((NUM_ENVS,), 12.0)),
          f"{r[0].item():.4f}")

    # If torque is 0, power is 0 too (not quite "holding still isn't penalized"... but 0 by the formula)
    r = fn(FakeEnv(torch.zeros(NUM_ENVS, NUM_JOINTS), ones), asset_cfg=Cfg())
    check("3. 0 if torque is 0", torch.allclose(r, torch.zeros(NUM_ENVS)), f"{r[0].item():.4f}")

    # If angular velocity is 0, power is 0 — torque spent holding still isn't penalized by this term.
    r = fn(FakeEnv(ones * 5.0, torch.zeros(NUM_ENVS, NUM_JOINTS)), asset_cfg=Cfg())
    check("4. 0 if angular velocity is 0 (no penalty for holding torque)", torch.allclose(r, torch.zeros(NUM_ENVS)),
          f"{r[0].item():.4f}")

    # Consumption is the same regardless of sign — checks that absolute values were used.
    neg_t, neg_v = -ones * 2.0, -ones * 3.0
    r_pp = fn(FakeEnv(ones * 2.0, ones * 3.0), asset_cfg=Cfg())
    r_nn = fn(FakeEnv(neg_t, neg_v), asset_cfg=Cfg())
    r_np = fn(FakeEnv(neg_t, ones * 3.0), asset_cfg=Cfg())
    check("5. sign-independent (absolute values used)",
          torch.allclose(r_pp, r_nn) and torch.allclose(r_pp, r_np),
          f"(+,+)={r_pp[0]:.2f} (-,-)={r_nn[0]:.2f} (-,+)={r_np[0]:.2f}")

    # Does it respect joint_ids?
    r = fn(FakeEnv(ones, ones), asset_cfg=Cfg(joint_ids=[0, 1, 2]))
    check("6. selects joints via asset_cfg.joint_ids", torch.allclose(r, torch.full((NUM_ENVS,), 3.0)),
          f"{r[0].item():.4f} (only 3 joints were selected, so it should be 3.0)")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with reward-power-distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
