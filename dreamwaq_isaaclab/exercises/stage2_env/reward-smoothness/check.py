#!/usr/bin/env python3
"""Fast check — reward-smoothness.

Runs without Isaac Sim. `env` is replaced with a minimal stub that only has action_manager.

    python check.py                 # check starter/rewards.py
    python check.py --solution      # check the finished version
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

NUM_ENVS, NUM_ACTIONS = 8, 12


class FakeActionManager:
    def __init__(self, num_envs: int, num_actions: int):
        self.action = torch.zeros(num_envs, num_actions)
        self.prev_action = torch.zeros(num_envs, num_actions)


class FakeEnv:
    """Mimics only the bare minimum that rewards.py actually touches."""

    def __init__(self, num_envs: int = NUM_ENVS, num_actions: int = NUM_ACTIONS):
        self.num_envs = num_envs
        self.device = "cpu"
        self.action_manager = FakeActionManager(num_envs, num_actions)

    def step(self, new_action: torch.Tensor) -> None:
        """Advance one step: t-1 ← t, t ← new_action."""
        am = self.action_manager
        am.prev_action = am.action.clone()
        am.action = new_action.clone()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def fresh():
        env = FakeEnv()
        term = mod.action_smoothness_l2(cfg=None, env=env)
        term.prev_prev_action = torch.zeros(NUM_ENVS, NUM_ACTIONS)
        return env, term

    # --- 1. If the action stays 0, the jerk is 0 too --------------------------------------
    env, term = fresh()
    try:
        r = term(env)
    except NotImplementedError:
        print("  TODO(reward-smoothness) is still empty. Fill in starter/rewards.py.")
        return 1
    check("1. returns a (num_envs,) scalar vector", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. jerk 0 if the action is 0", torch.allclose(r, torch.zeros(NUM_ENVS)), f"{r.tolist()[:3]}")

    # --- 3. Constant-rate change (linear ramp) gives zero jerk -----------------------------------
    # If a_t = t*c, the second-order difference is exactly 0. "Smooth motion isn't penalized".
    env, term = fresh()
    c = torch.full((NUM_ENVS, NUM_ACTIONS), 0.3)
    for t in range(1, 5):
        env.step(c * t)
        r = term(env)
    check("3. jerk 0 for a linear ramp (constant-rate change)", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # --- 4. Is the value exact on a known sequence ---------------------------------------
    # a = 0, 0, 1 → jerk = 1 - 2*0 + 0 = 1 → sum over 12 joints of 1^2 = 12
    env, term = fresh()
    env.step(torch.zeros(NUM_ENVS, NUM_ACTIONS))  # t-1 ← 0, t ← 0
    term(env)  # prev_prev ← 0
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))  # t-1 ← 0, t ← 1
    r = term(env)
    check("4. a=(0,0,1) → sum of jerk^2 = 12", torch.allclose(r, torch.full((NUM_ENVS,), 12.0)),
          f"{r[0].item():.4f} (expected 12.0)")

    # --- 5. Is the t-2 buffer actually updated --------------------------------------
    # If the update is forgotten, prev_prev stays 0 and the value below changes.
    # a = 0, 1, 2 → jerk = 2 - 2*1 + 0 = 0  (0 since it's linear)
    env, term = fresh()
    env.step(torch.zeros(NUM_ENVS, NUM_ACTIONS))
    term(env)
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    env.step(torch.full((NUM_ENVS, NUM_ACTIONS), 2.0))
    r = term(env)
    check("5. t-2 buffer updated (a=0,1,2 → jerk 0)", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.4f} — without updating the buffer you get 48")

    # --- 6. Was it cloned without aliasing ----------------------------------
    env, term = fresh()
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    before = term.prev_prev_action.clone()
    env.action_manager.prev_action.mul_(99.0)  # shake the original
    check("6. prev_prev_action is not an alias (cloned)",
          torch.allclose(term.prev_prev_action, before),
          "changing prev_action also changed prev_prev_action — .clone() is missing")

    # --- 7. Does reset clear the buffer ------------------------------------------
    env, term = fresh()
    env.step(torch.ones(NUM_ENVS, NUM_ACTIONS))
    term(env)
    term.reset(env_ids=None)
    check("7. reset() resets the t-2 buffer to 0",
          torch.allclose(term.prev_prev_action, torch.zeros(NUM_ENVS, NUM_ACTIONS)))

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with reward-foot-clearance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
