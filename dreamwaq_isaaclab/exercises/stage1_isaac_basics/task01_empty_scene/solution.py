#!/usr/bin/env python3
"""task01 — Launch an empty scene  [L0 · READ]

Look at the 3-part structure every Isaac Lab script has.

    1) Launch the Omniverse app first with AppLauncher
    2) Only then can isaaclab.* be imported
    3) Create a SimulationContext, reset() it, then loop step()

This order cannot be changed. Isaac Sim's extensions are registered after the app starts,
so if you move the imports up you get ModuleNotFoundError. This project's
train.py / play.py all have the same shape.

    python solution.py              # headless by default
    python solution.py --viz kit    # view in a GUI window

Note: the `--headless` flag seen in older material is deprecated in this version.
Passing nothing is already headless; if you want a GUI, pass `--viz kit`.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task01 — Launch an empty scene")
parser.add_argument("--steps", type=int, default=200, help="number of simulation steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    # dt=1/200 is the physics step this project uses (same as dreamwaq_env_cfg.py).
    # The policy acts once every decimation=4 steps, so the control rate is 50 Hz.
    sim_cfg = SimulationCfg(dt=1.0 / 200.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # Ground and light. Without these the robot falls forever and the screen is pitch black.
    sim_utils.GroundPlaneCfg().func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # reset() actually builds the physics scene and switches it to the "playing" state.
    # Before this call, no asset's data can be read.
    sim.reset()
    print("[INFO] Scene ready")

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        sim.step()
        step += 1

    # --- Self-check ---------------------------------------------------------
    sim_dt = sim.get_physics_dt()
    assert abs(sim_dt - 1.0 / 200.0) < 1e-9, f"physics dt is not 1/200: {sim_dt}"
    print(f"  [PASS] physics dt = {sim_dt:.6f} s  ({1 / sim_dt:.0f} Hz)")

    assert step == args_cli.steps, f"failed to run {args_cli.steps} steps: {step}"
    print(f"  [PASS] {step}-step simulation complete = {step * sim_dt:.2f} s")

    print("\n[OK] task01 passed. Next is task02_spawn_go2.")


if __name__ == "__main__":
    main()
    simulation_app.close()
