#!/usr/bin/env python3
"""task05 — Sensors: contact and height scan  [L2 · BUILD]

The last and most important exercise of Stage 1. The two values read here are
the raw material for the **termination condition** and the **privileged observation** you'll build in Stage 2.

    ContactSensor  → is the base touching the ground? → termination condition
    RayCaster      → height of the terrain around the robot → Oracle's 187-dim privileged observation

A height scan isn't simply the z of the ray hit points. It must be **height relative to
the base height** so that it means the same thing wherever the robot is. You build that transform yourself.

    python starter.py
    python starter.py --viz kit
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task05 — Sensors")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    RayCaster,
    RayCasterCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets import UNITREE_GO2_CFG  # isort:skip

SPAWN_HEIGHT = 0.42
# Same scan grid as the project: 0.1 m spacing, 1.6 x 1.0 m → 17 x 11 = 187 rays
SCAN_RESOLUTION, SCAN_SIZE = 0.1, [1.6, 1.0]
RAY_START_HEIGHT = 20.0  # rays are cast downward from 20 m above the robot
BASE_HEIGHT_TARGET = 0.30  # DreamWaQ's target base height


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 2.0, 1.2], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = UNITREE_GO2_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
    robot_cfg.actuators["base_legs"].stiffness = (
        80.0  # it must be standing for the scan to be meaningful
    )
    robot_cfg.actuators["base_legs"].damping = 2.0
    robot = Articulation(cfg=robot_cfg)

    # --- Contact sensor: monitors every body ----------------------------------------
    contact = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Robot/.*", history_length=3, track_air_time=True
        )
    )

    # --- Ray sensor: attached to the base, sweeping downward ------------------------------------
    scanner = RayCaster(
        RayCasterCfg(
            prim_path="/World/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, RAY_START_HEIGHT)),
            ray_alignment="yaw",  # the grid follows the base yaw even when the robot turns
            pattern_cfg=patterns.GridPatternCfg(
                resolution=SCAN_RESOLUTION, size=SCAN_SIZE
            ),
            mesh_prim_paths=["/World/defaultGroundPlane"],
            debug_vis=True,
        )
    )

    sim.reset()
    default_q = robot.data.default_joint_pos.clone()
    print(
        f"[INFO] {scanner.num_rays} rays ({SCAN_SIZE[0]}m x {SCAN_SIZE[1]}m, {SCAN_RESOLUTION}m spacing)\n"
    )

    dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        contact.update(dt)
        scanner.update(dt)

    # ── 1. Contact: which bodies are touching the ground ────────────────────────────────
    # net_forces_w_history: (num_envs, history_length, num_sensors, 3)
    forces = contact.data.net_forces_w_history.torch
    force_mag = torch.linalg.norm(forces, dim=-1).max(dim=1)[
        0
    ]  # history max → (envs, sensors)

    base_idx, _ = contact.find_sensors("base")
    feet_idx, feet_names = contact.find_sensors(".*foot")
    base_force = force_mag[0, base_idx].max().item()
    feet_force = force_mag[0, feet_idx]

    print("  [Contact]")
    print(
        f"    base contact force {base_force:8.2f} N   ← the termination condition looks at this value (terminates if > 1 N)"
    )
    for name, f in zip(feet_names, feet_force.tolist()):
        print(f"    {name:10s} contact force {f:8.2f} N")

    # ── 2. height scan: height relative to the base ────────────────────────────────────
    ray_hits_z = scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays) — world-frame z
    base_z = robot.data.root_pos_w[:, 2]  # (num_envs,)

    # ── TODO(height-scan) ─ level L2 ─────────────────────────────
    # Convert the world z of the ray hits into 'height relative to the base'
    #   hint: the observation should tell "how much higher my base is than the target height"
    #   hint: the project's actual formula is base_z - hit_z - BASE_HEIGHT_TARGET (mdp.height_scan)
    #   hint: base_z is (num_envs,) and hit_z is (num_envs, num_rays). Get the broadcasting right
    #   hint: on flat ground with the base exactly at the target height, the result must be 0
    height_scan = base_z - ray_hits_z - BASE_HEIGHT_TARGET
    # For the pass criteria, see README.md in this exercise folder.
    # raise NotImplementedError("TODO(height-scan)")
    # ─────────────────────────────────────────────────────────────────────

    print("\n  [height scan]")
    print(f"    num rays        {height_scan.shape[1]}")
    print(f"    base height     {base_z[0].item():.3f} m")
    print(
        f"    hit point z     mean {ray_hits_z[0].mean().item():+.3f} m (nearly 0 since it's flat)"
    )
    print(
        f"    relative height mean {height_scan[0].mean().item():+.3f} m, "
        f"range [{height_scan[0].min().item():+.3f}, {height_scan[0].max().item():+.3f}]"
    )

    # --- Self-check ---------------------------------------------------------
    print()
    assert (
        scanner.num_rays == 187
    ), f"there must be 187 rays (17 x 11): {scanner.num_rays}"
    print(
        f"  [PASS] 187 rays — this is exactly Oracle's 187-dim privileged observation"
    )

    assert (
        feet_force.sum().item() > 1.0
    ), f"the feet aren't on the ground: {feet_force.tolist()}"
    print(f"  [PASS] the feet are on the ground (sum {feet_force.sum().item():.1f} N)")

    assert (
        base_force < 1.0
    ), f"the base is touching the ground ({base_force:.2f} N) — the termination condition would fire"
    print(f"  [PASS] the base is not touching the ground ({base_force:.2f} N < 1 N)")

    # Since it's flat, the z of every hit point must be near 0.
    assert (
        ray_hits_z[0].abs().max().item() < 0.05
    ), "hit point z is scattered even though it's flat"
    print(f"  [PASS] on flat ground every hit point z is near 0")

    # Key: relative height = base height - terrain height - target height.
    expected = base_z[0].item() - 0.0 - BASE_HEIGHT_TARGET
    assert (
        abs(height_scan[0].mean().item() - expected) < 0.02
    ), f"relative height differs from expected: {height_scan[0].mean().item():+.3f} vs {expected:+.3f}"
    print(
        f"  [PASS] relative height = base height - terrain height - {BASE_HEIGHT_TARGET}  ({expected:+.3f} m)"
    )

    print(
        "\n[OK] task05 passed — Stage 1 complete.\n"
        "    You now have all the ingredients of the observation:\n"
        "      base velocity/orientation, joint angles/velocities (task02)\n"
        "      action → joint target                         (task03)\n"
        "      terrain and env_origins                       (task04)\n"
        "      contact force, height scan                    (task05)\n"
        "    In Stage 2 you assemble these into 45/48/64-dim observations."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
