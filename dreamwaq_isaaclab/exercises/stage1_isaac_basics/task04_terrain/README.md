# task04 — Terrain generation and env_origins  `L1 · FILL`

## Goal

Leave the infinite plane. DreamWaQ's "implicit terrain imagination"
only becomes meaningful on bumpy terrain.

```
terrain_type="plane"      a single infinite plane. Lightweight and good for debugging
terrain_type="generator"  generates sub-terrain tiles in a num_rows x num_cols grid
```

**The grid's row is the difficulty.** This is the foundation of the terrain curriculum.

## What to fill in

`TODO(terrain-cfg)` in `starter.py` — 5–8 lines configuring a `TerrainImporterCfg`.

| Item | Value | Why |
|---|---|---|
| `prim_path` | `"/World/ground"` | The height scanner references this path as its mesh |
| `terrain_type` | `"generator"` | |
| `terrain_generator` | `ROUGH_TERRAINS_CFG.copy()` | The original must not be modified |
| `collision_group` | `-1` | Global group — every env collides with this terrain |
| `num_rows` / `num_cols` | `5` / `5` | |

## Run

```bash
python starter.py
python solution.py --viz kit   # view the terrain grid
```

## Pass criteria

1. `env_origins` has shape `(num_envs, 3)`
2. The origins are spread across the grid (not bunched at one point)
3. At least 4 sub-terrain types
4. `curriculum=True`

## What you get when you run it

Composition of `ROUGH_TERRAINS_CFG`:

| sub-terrain | Proportion |
|---|---:|
| `pyramid_stairs` / `pyramid_stairs_inv` | 0.20 / 0.20 |
| `boxes` | 0.20 |
| `random_rough` | 0.20 |
| `hf_pyramid_slope` / `_inv` | 0.10 / 0.10 |

The z range of `env_origins` spreads to `[-1.38, +0.71] m` — each tile has a different height.
**This is the rationale for the 0.42 m spawn height.** `boxes` are up to 0.1 m and
xy is jittered by ±0.5 m on reset, so spawning low gets the robot stuck in the terrain.

## Connection to the curriculum

During training `mdp.terrain_levels_vel` moves robots that walk well to higher rows and those that don't
to lower rows. **`Curriculum/terrain_levels`** in the training log is exactly the mean of this
row index — you'll read this curve in Stage 5.
