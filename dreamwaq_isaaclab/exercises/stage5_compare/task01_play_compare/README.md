# task01_play_compare — Look at the 3 checkpoints with your own eyes  `L0 · READ`

**Stage 5 · Comparing results**

## Goal

There's no code to write. **Watch.**

This course's first lesson: **don't just look at metrics — check the gait in video.** It really happens that the reward curve goes up
while the robot is crawling on its knees. You can't tell from numbers.

## Preparation

Extract the distributed assets under `dreamwaq_manager/logs/` (see the [Stage 5 README](../README.md)).
Each run folder already contains `videos/play/rl-video-step-0.mp4` — watch those first.

## 1. Watch the 3 distributed videos side by side

```bash
cd dreamwaq_manager/logs/rsl_rl
ls */*/videos/play/*.mp4
```

The 3 rough runs are this stage's main comparison axis (the 3 flat runs are included with the same structure).

| run (`experiment_name`) | actor observation | Play task id |
|---|---|---|
| `BaseDwq-Official-Rough-PPO-v0` | 45 — knows neither terrain nor velocity | `DreamWaQ-BaseDwq-Rough-PPO-Play-v0` |
| `Waq-Official-Rough-PPO-v0` | 64 — CENet estimates | `DreamWaQ-Waq-Official-Rough-PPO-Play-v0` |
| `OracleDwq-Official-Rough-PPO-v0` | 48 — just receives the actual velocity | `DreamWaQ-OracleDwq-Rough-PPO-Play-v0` |

## 2. Run them yourself

There's only one 20-second (400-step) video each. If you want to see more, run them yourself.
For `--load_run`, give the **timestamp folder name** under `experiment_name`.

```bash
cd dreamwaq_manager

# View on screen (viewport)
python scripts/rsl_rl/play.py --task=DreamWaQ-BaseDwq-Rough-PPO-Play-v0 \
    --num_envs=16 --track_agent \
    --load_run=2026-08-09_15-03-53_DreamWaQ-BaseDwq-Rough-PPO-v0_seed42_envs4096 \
    --checkpoint=model_3999.pt

# Record a new video (headless)
python scripts/rsl_rl/play.py --task=DreamWaQ-Waq-Official-Rough-PPO-Play-v0 \
    --headless --num_envs=32 --video --video_length=400 \
    --track_agent --track_env_index=0 \
    --load_run=2026-08-09_20-38-55_DreamWaQ-Waq-Official-Rough-PPO-v0_seed42_envs4096 \
    --checkpoint=model_3999.pt
```

Adding `--debug_vis` also shows the **commanded velocity arrow** and the **height scanner rays**
(the `*_PLAY` settings of these three tasks don't turn debug_vis on — play.py's flag does).
How much the command arrow and the actual direction of travel diverge is exactly the velocity tracking performance.
Adjust the camera with `--cam_distance` / `--cam_height`.

## 3. Compare against mid-training

With `save_interval=500` and 4000 iterations, there are **9** checkpoints
(`model_0` / `500` / `1000` / `1500` / `2000` / `2500` / `3000` / `3500` / `3999`).
Comparing early and final shows what improved.

```bash
python scripts/rsl_rl/play.py --task=... --checkpoint=model_500.pt  ...   # stumbles
python scripts/rsl_rl/play.py --task=... --checkpoint=model_3999.pt ...   # walks
```

## What to look at

Watch with a checklist in hand. Just watching ends with "they all look similar?".

| Aspect | Specifically |
|---|---|
| **Command tracking** | Does the arrow direction match the actual direction of travel? Especially for sidestep and turn-in-place commands |
| **Falling** | Does the base touch at rough-terrain boundaries or on stairs? |
| **Foot lifting** | Does it lift its feet rather than drag them? (what the `foot_clearance` reward aims for) |
| **Jitter** | Do the joints jitter at high frequency? (what the `smoothness` reward aims for). On hardware it burns out motors |
| **Posture** | Is the base height constant, and does it avoid walking in a crouch? |

## Open questions

- Base knows neither terrain nor velocity — **so why does it walk reasonably well?**
  (Hint: terrain information is indirectly present in the joint angle history. That's why Base isn't 0)
- Oracle is the upper bound yet isn't perfect (1.0). It's 0.582 on rough. **What's left?**
- Does Waq's gait resemble Base or Oracle more? Does the impression match the numbers (task02)?
- Peeking at the numbers: on rough Waq **tracks velocity better** than Base
  (0.5565 vs 0.5492) but **falls more often** (`base_contact` 0.123 vs 0.093,
  `mean_episode_length` 919 vs 936). Can you see that difference in the video?
  Is it walking more aggressively, or is it just unstable?

## Pass criteria

It's enough if, after watching the 3 videos, you can describe the differences using the checklist above. There's no `check.py`.

## Next

- [`task02_curves`](../task02_curves/) — confirm your impressions with numbers (L2)
