#!/usr/bin/env python3
"""task02 — Spawn Go2 and read its state  [L1 · FILL]

Put one robot into the scene and read its state while the simulation runs.
The observations you will build in Stage 2 are all combinations of the values read here.

    python starter.py
    python starter.py --viz kit    # view in the GUI
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task02 — Spawn Go2")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets import UNITREE_GO2_CFG  # isort:skip

# DreamWaQ's spawn height. The paper uses 0.34 m but this project uses 0.42 m —
# to clear the generated terrain's boxes (up to 0.1 m) together with the xy ±0.5 m reset randomization.
SPAWN_HEIGHT = 0.42


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # ── TODO(spawn-go2) ─ level L1 ─────────────────────────────
    # Copy UNITREE_GO2_CFG, set the prim path and spawn height, and create the Articulation
    #   hint: always copy the cfg with .copy(). Modifying the original affects other code in the same process
    #   hint: set prim_path to "/World/Robot"
    #   hint: init_state.pos is an (x, y, z) tuple. Put SPAWN_HEIGHT in z
    #   hint: create it with Articulation(cfg=...)
    go2_cfg = UNITREE_GO2_CFG.copy()
    go2_cfg.prim_path = "/World/Robot"
    go2_cfg.init_state.pos = (0.0, 0.0, SPAWN_HEIGHT)
    robot = Articulation(cfg=go2_cfg)
    # For the pass criteria, see README.md in this exercise folder.
    # raise NotImplementedError("TODO(spawn-go2)")
    # ─────────────────────────────────────────────────────────────────────

    # robot.data can't be read before reset(), because the physics scene doesn't exist yet.
    sim.reset()
    print("[INFO] Go2 spawned\n")

    # --- What is the robot made of ---------------------------------------------
    print(f"  {robot.num_joints} joints: {robot.joint_names}")
    print(f"  {robot.num_bodies} bodies: {robot.body_names}\n")

    # The default pose. The policy's actions are interpreted as 'offsets' from this value.
    default_q = robot.data.default_joint_pos[0]
    print(f"  Default joint pose (rad): {[f'{v:.2f}' for v in default_q.tolist()]}\n")

    # --- Run the simulation and read the state ---------------------------------
    # No actions are given, so the robot tries to hold the default pose but sinks under gravity.
    for step in range(args_cli.steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if step % 50 == 0:
            pos = robot.data.root_pos_w[0]
            lin = robot.data.root_lin_vel_b[0]
            grav = robot.data.projected_gravity_b[0]
            print(
                f"  step {step:3d} | height {pos[2]:.3f} m"
                f" | body-frame velocity ({lin[0]:+.2f}, {lin[1]:+.2f}, {lin[2]:+.2f})"
                f" | projected gravity z {grav[2]:+.3f}"
            )

    # --- Self-check ---------------------------------------------------------
    print()
    assert robot.num_joints == 12, f"Go2 must have 12 joints: {robot.num_joints}"
    print(f"  [PASS] 12 joints (4 legs × hip/thigh/calf 3)")

    assert "base" in robot.body_names, f"no 'base' body: {robot.body_names}"
    print(
        f"  [PASS] 'base' body exists — the termination condition looks at this body's contact"
    )

    feet = [b for b in robot.body_names if b.endswith("foot")]
    assert len(feet) == 4, f"there must be 4 feet: {feet}"
    print(f"  [PASS] 4 feet: {feet}")

    # It must not sink below the ground (if it did, the spawn height or ground setup is wrong).
    height = robot.data.root_pos_w[0, 2].item()
    assert (
        0.0 < height < SPAWN_HEIGHT + 0.1
    ), f"robot height is abnormal: {height:.3f} m"
    print(f"  [PASS] final height {height:.3f} m — above the ground")

    # Body-frame gravity z: -1 when standing upright, close to 0 when lying fully on its side.
    # 3 of the 45 observation dimensions in Stage 2 are exactly this projected_gravity_b.
    gz = robot.data.projected_gravity_b[0, 2].item()
    assert -1.05 <= gz <= 0.05, f"projected gravity z is out of range: {gz:.3f}"
    print(f"  [PASS] projected gravity z = {gz:.3f}  (upright=-1, lying on side=0)")

    print(
        "\n[OK] task02 passed.\n"
        f"    Since no joint targets were ever given, the robot sank under gravity\n"
        f"    (height {SPAWN_HEIGHT:.2f} → {height:.3f} m, projected gravity z {gz:+.3f}).\n"
        "    Making it stand is the next exercise, task03_pd_control."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
