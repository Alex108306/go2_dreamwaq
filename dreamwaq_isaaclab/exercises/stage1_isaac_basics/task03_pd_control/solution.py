#!/usr/bin/env python3
"""task03 — PD control and action scale  [L1 · FILL]

In task02 the robot sank, because no joint targets were ever given.
Here we give targets and see how those targets are computed from the policy's actions.

DreamWaQ's action interpretation (JointPositionActionCfg in velocity_env_cfg.py):

    joint_target = default_joint_pos + action * 0.25
                   └─ use_default_offset=True performs this addition
                                            └─ scale=0.25

The policy outputs "how far to deviate from the default pose", not "put this joint at X rad".

Two robots stand side by side. One has the soft gains this project uses (Kp=20),
the other stiff gains (Kp=80). Both receive the same command, action=0, i.e.
"hold the default pose". Why the results differ is the core of this exercise.

    python solution.py
    python solution.py --viz kit    # view the two side by side in the GUI
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task03 — PD control")
parser.add_argument("--steps", type=int, default=300)
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

SPAWN_HEIGHT = 0.42
ACTION_SCALE = 0.25  # JointPositionActionCfg(scale=0.25)
SOFT_KP, SOFT_KD = 20.0, 0.5  # DreamWaQ paper values — the values go2_base_cfg.py uses
STIFF_KP, STIFF_KD = 80.0, 2.0  # control group


def spawn(name: str, y: float, kp: float, kd: float) -> Articulation:
    cfg = UNITREE_GO2_CFG.copy()
    cfg.prim_path = f"/World/{name}"
    cfg.init_state.pos = (0.0, y, SPAWN_HEIGHT)
    cfg.actuators["base_legs"].stiffness = kp
    cfg.actuators["base_legs"].damping = kd
    return Articulation(cfg=cfg)


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.0, 0.0, 1.0], target=[0.0, 0.0, 0.3])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    soft = spawn("Soft", y=+0.6, kp=SOFT_KP, kd=SOFT_KD)
    stiff = spawn("Stiff", y=-0.6, kp=STIFF_KP, kd=STIFF_KD)

    sim.reset()
    print(f"[INFO] Soft Kp={SOFT_KP} Kd={SOFT_KD}  vs  Stiff Kp={STIFF_KP} Kd={STIFF_KD}")
    print(f"[INFO] Both action=0 — 'hold the default pose'\n")

    # ex:begin id=pd-target level=1 stage=stage1_isaac_basics task=compute the joint target from the action
    #   hint: joint_target = default_joint_pos + action * ACTION_SCALE
    #   hint: robot.data.default_joint_pos has shape (num_envs, 12)
    #   hint: if the action is 0, the target is the default pose itself
    def action_to_target(robot: Articulation, action: torch.Tensor) -> torch.Tensor:
        """Convert a policy action (±1) into a joint position target (rad)."""
        return robot.data.default_joint_pos + action * ACTION_SCALE
    # ex:end

    zero = torch.zeros_like(soft.data.default_joint_pos)
    target_soft = action_to_target(soft, zero)
    target_stiff = action_to_target(stiff, zero)

    for _ in range(args_cli.steps):
        soft.set_joint_position_target(target_soft)
        stiff.set_joint_position_target(target_stiff)
        soft.write_data_to_sim()
        stiff.write_data_to_sim()
        sim.step()
        soft.update(sim.get_physics_dt())
        stiff.update(sim.get_physics_dt())

    def report(robot: Articulation, target: torch.Tensor, label: str) -> tuple[float, float]:
        h = robot.data.root_pos_w[0, 2].item()
        err = (robot.data.joint_pos - target).abs().max().item()
        print(f"  {label:6s} height {h:.3f} m | max joint tracking error {err:.3f} rad")
        return h, err

    h_soft, e_soft = report(soft, target_soft, "Soft")
    h_stiff, e_stiff = report(stiff, target_stiff, "Stiff")
    print()

    # --- Self-check ---------------------------------------------------------
    # An action of ±1 moves a joint by only ±0.25 rad.
    one = torch.ones_like(zero)
    delta = (action_to_target(soft, one) - soft.data.default_joint_pos).abs().max().item()
    assert abs(delta - ACTION_SCALE) < 1e-6, f"wrong action scale: {delta:.4f}"
    print(f"  [PASS] action=+1 → joint target deviates from the default pose by only {delta:.2f} rad")

    assert abs(action_to_target(soft, zero) - soft.data.default_joint_pos).max() < 1e-9
    print(f"  [PASS] action=0 → target = default pose")

    # The stiff one tracks better. That's the definition of Kp.
    assert e_stiff < e_soft, f"raised Kp but tracking got worse ({e_stiff:.3f} >= {e_soft:.3f})"
    print(f"  [PASS] Kp↑ → tracking error↓  ({e_soft:.3f} → {e_stiff:.3f} rad)")

    assert h_stiff > h_soft, f"the stiff one is lower ({h_stiff:.3f} <= {h_soft:.3f})"
    print(f"  [PASS] Kp↑ → posture holding↑  (height {h_soft:.3f} → {h_stiff:.3f} m)")

    print(
        "\n[OK] task03 passed.\n"
        f"    Kp={SOFT_KP} can't even fully support its own weight and sags by {e_soft:.2f} rad.\n"
        "    DreamWaQ using these soft gains is not a mistake — the real Go2's\n"
        "    joints are like that, and the policy learns to make up the shortfall with its action at every step.\n"
        "    (Raising the gains makes it stand well in sim, but it diverges that much more when transferred to hardware.)\n"
        "    Next is task04_terrain."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
