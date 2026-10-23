# task01 — Launch an empty scene  `L0 · READ`

## Goal

See with your own eyes the **3-part structure every Isaac Lab script has**. There's no code to write.

```
1) Launch the Omniverse app first with AppLauncher
2) Only then import isaaclab.*      ← this order cannot be changed
3) Create a SimulationContext, reset() → step()
```

## Run

```bash
python solution.py             # headless (default)
python solution.py --viz kit   # view in the GUI
```

## What to check

- The `import` statements are in the middle of the file, **below** `simulation_app = app_launcher.app`.
  Isaac Sim extensions are registered after the app starts, so moving them up gives `ModuleNotFoundError`.
  This project's `train.py` / `play.py` all have the same shape.
- `dt=1/200` — this project's physics step. The policy acts once every `decimation=4` steps,
  so **the control rate is 50 Hz**.
- `sim.reset()` actually builds the physics scene. Before it, no asset's `data` can be read.

## Pass criteria

When it runs to the end, 2 `[PASS]` lines and `[OK]` are printed.

## Try it yourself (optional)

Change `dt` to `1/50` and run it in the GUI. Can you see the physics getting rough?
That's why, in locomotion, a large physics step makes contacts penetrate or bounce.
