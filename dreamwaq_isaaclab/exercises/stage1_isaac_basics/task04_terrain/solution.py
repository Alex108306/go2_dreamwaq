#!/usr/bin/env python3
"""task04 — Terrain generation and env_origins  [L1 · FILL]

So far we've played on an infinite plane. DreamWaQ's "implicit terrain imagination"
only becomes meaningful on bumpy terrain.

Isaac Lab has two kinds of terrain.

    terrain_type="plane"      a single infinite plane. Lightweight and good for debugging
    terrain_type="generator"  generates sub-terrain tiles in a num_rows × num_cols grid

In the generator grid, **the row is the difficulty**. The terrain curriculum moves a robot up to the next
row if it does well and down if it doesn't (`mdp.terrain_levels_vel`).

    python solution.py
    python solution.py --viz kit    # view the terrain in the GUI
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="task04 — Terrain")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    sim.set_camera_view(eye=[14.0, 14.0, 10.0], target=[0.0, 0.0, 0.0])

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # --- What's inside ROUGH_TERRAINS_CFG -----------------------------
    print("[INFO] sub-terrains of ROUGH_TERRAINS_CFG:")
    for name, sub in ROUGH_TERRAINS_CFG.sub_terrains.items():
        print(f"    {name:22s} proportion {sub.proportion:.2f}")
    print()

    # ex:begin id=terrain-cfg level=1 stage=stage1_isaac_basics task=Create a generated-terrain TerrainImporterCfg
    #   hint: set prim_path to "/World/ground". The height scanner references this path as its mesh
    #   hint: terrain_type is "generator"; for terrain_generator use a .copy() of ROUGH_TERRAINS_CFG
    #   hint: collision_group=-1 is needed so every env collides with this terrain (global group)
    #   hint: the grid size is set by terrain_generator.num_rows / num_cols. Here, 5 x 5
    terrain_gen = ROUGH_TERRAINS_CFG.copy()
    terrain_gen.num_rows = 5
    terrain_gen.num_cols = 5
    terrain_gen.curriculum = True

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen,
        collision_group=-1,
        max_init_terrain_level=4,
        debug_vis=False,
    )
    # ex:end

    # The TerrainImporter places one origin (env_origins) per env on the terrain.
    terrain_cfg.num_envs = args_cli.num_envs
    terrain_cfg.env_spacing = 3.0
    terrain = terrain_cfg.class_type(terrain_cfg)

    sim.reset()
    print(f"[INFO] Terrain generated ({terrain_gen.num_rows} x {terrain_gen.num_cols} grid)\n")

    origins = terrain.env_origins  # (num_envs, 3)
    print(f"  env_origins shape: {tuple(origins.shape)}")
    print(f"  x range [{origins[:, 0].min():+.2f}, {origins[:, 0].max():+.2f}] m")
    print(f"  y range [{origins[:, 1].min():+.2f}, {origins[:, 1].max():+.2f}] m")
    print(f"  z range [{origins[:, 2].min():+.2f}, {origins[:, 2].max():+.2f}] m  ← terrain height difference\n")

    # --- Self-check ---------------------------------------------------------
    assert tuple(origins.shape) == (args_cli.num_envs, 3), f"env_origins has an unexpected shape: {origins.shape}"
    print(f"  [PASS] one origin per env ({args_cli.num_envs})")

    # On a plane every origin's z is 0. On generated terrain each tile has a different height.
    assert origins[:, :2].abs().max() > 1.0, "origins are bunched at one point — the terrain grid wasn't built"
    print(f"  [PASS] origins are spread across the terrain grid")

    # Rationale for the 0.42 m robot spawn height: the terrain's boxes are up to 0.1 m.
    n_sub = len(ROUGH_TERRAINS_CFG.sub_terrains)
    assert n_sub >= 4, f"too few sub-terrains: {n_sub}"
    print(f"  [PASS] {n_sub} sub-terrain types — flat/slopes/stairs/blocks etc. are mixed")

    # Foundation of the curriculum: the row is the difficulty.
    assert terrain_gen.curriculum, "curriculum=True is required for rows to be ordered by difficulty"
    print(f"  [PASS] curriculum=True — rows become difficulty levels")

    print(
        "\n[OK] task04 passed.\n"
        f"    The {terrain_gen.num_rows} rows are difficulty levels. During training mdp.terrain_levels_vel\n"
        "    moves robots that walk well to higher rows and those that don't to lower rows.\n"
        "    Curriculum/terrain_levels in the training log is exactly the mean of this row index.\n"
        "    Next is task05_sensors — the last and most important exercise."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
