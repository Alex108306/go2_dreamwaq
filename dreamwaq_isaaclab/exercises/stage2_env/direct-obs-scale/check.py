#!/usr/bin/env python3
"""Fast check — direct-obs-scale.  (no Isaac Sim needed, ~2 s)

Calls `DreamWaQEnv._get_observations` directly with a fake `self`. It checks a single method
in isolation without launching a real env.

    python check.py                 # starter/dreamwaq_env.py
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
    / "dreamwaq_direct/source/dreamwaq_direct/dreamwaq_direct/tasks/locomotion/dreamwaq_env.py"
)
STARTER = HERE / "starter" / "dreamwaq_env.py"

N, J = 4, 12
# Project defaults (dreamwaq_env_cfg.py)
S_LIN, S_ANG, S_POS, S_VEL = 2.0, 0.25, 1.0, 0.05


NUM_RAYS = 187


class Cfg:
    obs_scale_height = 5.0
    state_space = 235
    obs_scale_lin_vel = S_LIN
    obs_scale_ang_vel = S_ANG
    obs_scale_dof_pos = S_POS
    obs_scale_dof_vel = S_VEL
    obs_noise = False       # turning noise off allows deterministic checking
    system_delay = False
    include_lin_vel_in_obs = False


class FakeSelf:
    """Minimal object with only what _get_observations touches."""

    def __init__(self, lin, ang, grav, jpos, jvel, jdef, cmd, act):
        data = type("D", (), {
            "root_lin_vel_b": lin, "root_ang_vel_b": ang, "projected_gravity_b": grav,
            "joint_pos": jpos, "joint_vel": jvel, "default_joint_pos": jdef,
        })()
        self._robot = type("R", (), {"data": data})()
        self.cfg = Cfg()
        self._commands = cmd
        self._actions = act
        self._true_lin_vel_b = torch.zeros_like(lin)
        self._obs_history_buf = torch.zeros(N, 5, 45)
        # For the critic (privileged) observation — height scanner and push impulse
        hs = type("HD", (), {
            "pos_w": torch.zeros(N, 3),
            "ray_hits_w": torch.zeros(N, NUM_RAYS, 3),
        })()
        self._height_scanner = type("HS", (), {"data": hs})()
        self._disturb_force = torch.zeros(N, 3)
        self._command_vis = None      # GUI markers (absent in headless checks)
        self._heading_vis = None
        self.num_envs, self.device = N, "cpu"


def make(lin_v=0.0, ang_v=0.0, jvel_v=0.0, jpos_v=0.0):
    return FakeSelf(
        lin=torch.full((N, 3), lin_v),
        ang=torch.full((N, 3), ang_v),
        grav=torch.zeros(N, 3),
        jpos=torch.full((N, J), jpos_v),
        jvel=torch.full((N, J), jvel_v),
        jdef=torch.zeros(N, J),
        cmd=torch.zeros(N, 4),
        act=torch.zeros(N, J),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    get_obs = load_module(path, extra_paths=(SOLUTION.parent,)).DreamWaQEnv._get_observations

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def policy_of(obj) -> torch.Tensor:
        out = get_obs(obj)
        return out["policy"] if isinstance(out, dict) else out

    try:
        obs = policy_of(make())
    except NotImplementedError:
        print("  TODO(direct-obs-scale) is still empty. Fill in starter/dreamwaq_env.py.")
        return 1

    check("1. observation is 45-dimensional (Base/Waq)", obs.shape == (N, 45), f"shape={tuple(obs.shape)}")

    # Observation layout: ang_vel(3) gravity(3) commands(3) joint_pos(12) joint_vel(12) actions(12)
    ANG, GRAV, CMD, JPOS, JVEL = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 21), slice(21, 33)

    obs = policy_of(make(ang_v=4.0))
    check("2. obs_scale_ang_vel (0.25) applied to ang_vel",
          torch.allclose(obs[:, ANG], torch.full((N, 3), 4.0 * S_ANG), atol=1e-6),
          f"{obs[0, ANG].tolist()} (expected {4.0 * S_ANG})")

    obs = policy_of(make(jvel_v=20.0))
    check("3. obs_scale_dof_vel (0.05) applied to joint_vel",
          torch.allclose(obs[:, JVEL], torch.full((N, J), 20.0 * S_VEL), atol=1e-6),
          f"{obs[0, JVEL][0].item():.4f} (expected {20.0 * S_VEL})")

    obs = policy_of(make(jpos_v=0.5))
    check("4. obs_scale_dof_pos (1.0) applied to joint_pos",
          torch.allclose(obs[:, JPOS], torch.full((N, J), 0.5 * S_POS), atol=1e-6),
          f"{obs[0, JPOS][0].item():.4f}")

    # gravity is not scaled (1.0 in the original too)
    s = make()
    s._robot.data.projected_gravity_b = torch.full((N, 3), -1.0)
    obs = policy_of(s)
    check("5. gravity is not scaled", torch.allclose(obs[:, GRAV], torch.full((N, 3), -1.0), atol=1e-6),
          f"{obs[0, GRAV].tolist()}")

    # true_lin_vel must be the value 'before' scaling — important since it's CENet's training target
    s = make(lin_v=3.0)
    policy_of(s)
    check("6. _true_lin_vel_b keeps the raw value from before scaling",
          torch.allclose(s._true_lin_vel_b, torch.full((N, 3), 3.0), atol=1e-6),
          f"{s._true_lin_vel_b[0].tolist()} — storing the scaled value corrupts CENet's target")

    # --- Key: what happens without scaling ---
    # Real robot scale: joint_vel ~ ±20 rad/s, commands ~ ±1
    s = make(jvel_v=20.0)
    s._commands = torch.ones(N, 4)
    obs = policy_of(s)
    jvel_mag = obs[:, JVEL].abs().mean().item()
    cmd_mag = obs[:, CMD].abs().mean().item()
    check("7. after scaling, joint_vel doesn't overwhelm the commands (ratio < 3)",
          jvel_mag / max(cmd_mag, 1e-9) < 3.0,
          f"joint_vel/commands = {jvel_mag / max(cmd_mag, 1e-9):.1f}x — forgetting the scale makes it 20x")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
