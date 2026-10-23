#!/usr/bin/env python3
"""Fast check — reward-foot-clearance.

Runs without Isaac Sim. env / asset / height sensor are replaced with minimal stubs.

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

NUM_ENVS, NUM_FEET, NUM_RAYS = 4, 4, 187
DESIRED = 0.12


class Cfg:
    def __init__(self, name: str, body_ids=None):
        self.name = name
        self.body_ids = body_ids


class FakeAsset:
    def __init__(self, feet_pos_z: torch.Tensor, feet_vel_xy: torch.Tensor):
        n, f = feet_pos_z.shape
        self.data = type("D", (), {})()
        pos = torch.zeros(n, f, 3)
        pos[:, :, 2] = feet_pos_z
        vel = torch.zeros(n, f, 3)
        vel[:, :, 0] = feet_vel_xy[..., 0]
        vel[:, :, 1] = feet_vel_xy[..., 1]
        vel[:, :, 2] = 99.0  # mix in z velocity — it must not affect the answer
        self.data.body_pos_w = pos
        self.data.body_lin_vel_w = vel


class FakeSensor:
    def __init__(self, terrain_z: torch.Tensor, with_nan: bool = False):
        hits = torch.zeros(NUM_ENVS, NUM_RAYS, 3)
        hits[:, :, 2] = terrain_z[:, None]
        if with_nan:
            hits[:, 0, 2] = float("nan")
            hits[:, 1, 2] = float("inf")
        self.data = type("D", (), {"ray_hits_w": hits})()


class FakeScene:
    def __init__(self, asset, sensor):
        self._asset, self.sensors = asset, {"height_scanner": sensor}

    def __getitem__(self, _name):
        return self._asset


class FakeEnv:
    def __init__(self, asset, sensor):
        self.scene = FakeScene(asset, sensor)
        self.num_envs, self.device = NUM_ENVS, "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)
    fn = mod.foot_clearance_l2

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def call(feet_z, speed, terrain_z=None, use_sensor=True, with_nan=False):
        terrain = torch.zeros(NUM_ENVS) if terrain_z is None else terrain_z
        vel = torch.zeros(NUM_ENVS, NUM_FEET, 2)
        vel[..., 0] = speed  # only vx → lateral velocity = |vx|
        env = FakeEnv(FakeAsset(feet_z, vel), FakeSensor(terrain, with_nan))
        return fn(
            env,
            asset_cfg=Cfg("robot", body_ids=list(range(NUM_FEET))),
            height_sensor_cfg=Cfg("height_scanner") if use_sensor else None,
            desired_height=DESIRED,
        )

    flat = torch.full((NUM_ENVS, NUM_FEET), DESIRED)  # exactly the target height
    try:
        r = call(flat, speed=1.0)
    except NotImplementedError:
        print("  TODO(reward-foot-clearance) is still empty. Fill in starter/rewards.py.")
        return 1

    check("1. returns a (num_envs,) scalar vector", tuple(r.shape) == (NUM_ENVS,), f"shape={tuple(r.shape)}")
    check("2. penalty 0 when the feet are exactly at the target height", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # If a foot is stationary (lateral velocity 0), it isn't penalized even if the height is off — planted feet get a pass.
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=0.0)
    check("3. penalty 0 when lateral velocity is 0 (planted foot)", torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-6),
          f"{r[0].item():.6f}")

    # Known value: foot height 0, target 0.12, velocity 1.0, 4 feet → 4 * 0.12^2 * 1 = 0.0576
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    check("4. height 0 / velocity 1 / 4 feet → 4 x 0.12^2 = 0.0576",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.6f} (expected {4 * DESIRED**2:.4f})")

    # Linearly proportional to velocity.
    r1 = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    r2 = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=2.0)
    check("5. penalty is linearly proportional to lateral velocity", torch.allclose(r2, 2 * r1, atol=1e-6),
          f"{r1[0].item():.4f} → {r2[0].item():.4f}")

    # Key: when the terrain rises, it must be corrected to 'height above terrain'.
    # foot world z = 0.5, terrain = 0.38 → 0.12 above terrain = target → penalty 0.
    r = call(torch.full((NUM_ENVS, NUM_FEET), 0.5), speed=1.0,
             terrain_z=torch.full((NUM_ENVS,), 0.5 - DESIRED))
    check("6. subtracts terrain height to use 'height above terrain'",
          torch.allclose(r, torch.zeros(NUM_ENVS), atol=1e-5),
          f"{r[0].item():.6f} — using world z as-is doesn't give 0")

    # Using ray hits contaminated with NaN/inf as-is pollutes the result.
    r = call(flat, speed=1.0, with_nan=True)
    check("7. cleans NaN/inf in ray hits to 0", bool(torch.isfinite(r).all()),
          f"{r.tolist()} — torch.nan_to_num is missing")

    # z velocity must be ignored (the stub put in vz=99).
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0)
    check("8. doesn't include the z component in lateral velocity",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.4f} — vz=99 leaked in")

    # With height_sensor_cfg=None, world z is used without terrain correction (flat-only fallback).
    r = call(torch.zeros(NUM_ENVS, NUM_FEET), speed=1.0, use_sensor=False)
    check("9. falls back to world z if the sensor is None",
          torch.allclose(r, torch.full((NUM_ENVS,), 4 * DESIRED**2), atol=1e-6),
          f"{r[0].item():.4f}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
