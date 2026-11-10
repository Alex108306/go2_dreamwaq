#!/usr/bin/env python3
"""Stage 3 · task04 — PPO cfg and this repo's real-world issues.  `L1 · FILL`

Wrap up here the two things task03 passed over as "what we don't have".

  1. desired_kl adaptive learning rate  <- blank
  2. log_std clamp                      <- a problem this repo actually ran into. Read and run it

    python starter.py
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# This repo's actual PPO cfg. If the values change, the output here follows.
_CFG_PATH = (
    Path(__file__).resolve().parents[3]
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/tasks/locomotion/config/go2"
    / "agents/rsl_rl_ppo_cfg.py"
)

# Must match the values used by the monkey patch in train.py.
LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.0


def adapt_learning_rate(
    kl_mean: float, learning_rate: float, desired_kl: float = 0.01
) -> float:
    """Adjust the learning rate based on KL (rsl_rl `PPO.update`'s adaptive schedule).

    Args:
        kl_mean: minibatch mean of the KL between the previous and current policy
        learning_rate: current learning rate
        desired_kl: target KL (0.01 in this repo)

    Returns:
        The updated learning rate. Left unchanged if between half and 2x the target.
    """
    # ── TODO(ppo-cfg) ─ level L1 ─────────────────────────────
    # Fill in the rule that raises or lowers the learning rate based on KL
    #   hint: if KL is more than 2x desired_kl it moved too much -> divide lr by 1.5
    #   hint: if KL is less than half of desired_kl it moved too little -> multiply lr by 1.5
    #   hint: leave it alone in between (half to 2x) — this 'dead zone' prevents oscillation
    #   hint: bound it with lower bound 1e-5 and upper bound 1e-2 (use max / min)
    #   hint: don't raise it if kl_mean is 0 (rsl_rl's kl_mean > 0.0 condition)
    # For the pass criteria, see README.md in this exercise folder.
    if kl_mean > 0:
        if kl_mean > desired_kl * 2:
            learning_rate = max(1e-5, learning_rate / 1.5)
        elif kl_mean < desired_kl / 2:
            learning_rate = min(1e-2, learning_rate * 1.5)
        else:
            learning_rate = learning_rate
        return learning_rate
    else:
        return learning_rate
    # raise NotImplementedError("TODO(ppo-cfg)")
    # ─────────────────────────────────────────────────────────────────────


def clamp_log_std(log_std: torch.Tensor) -> torch.Tensor:
    """Bound log_std to [-5, 0] and restore NaN/inf. (Same as the monkey patch in train.py)

    This function is given — no need to fill it in. See for yourself in main() below why it's needed.
    """
    log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=LOG_STD_MIN)
    return log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)


def _read_cfg_values() -> dict[str, str]:
    """Read the values of the algorithm block straight from the cfg file (without importing isaaclab)."""
    if not _CFG_PATH.exists():
        return {}
    text = _CFG_PATH.read_text(encoding="utf-8")
    block = re.search(r"algorithm = RslRlPpoAlgorithmCfg\((.*?)\n    \)", text, re.S)
    if not block:
        return {}
    return dict(re.findall(r"(\w+)\s*=\s*([^,\n]+),", block.group(1)))


def main() -> int:
    print("=" * 72)
    print(" 1. This repo's PPO cfg  (read directly from rsl_rl_ppo_cfg.py)")
    print("=" * 72)

    meaning = {
        "clip_param": "clips the surrogate's ratio to 1±eps. Paper §III-B",
        "value_loss_coef": "value loss weight",
        "entropy_coef": "entropy bonus. Maintains exploration",
        "num_learning_epochs": "how many times the same rollout is reused (the on-policy compromise)",
        "num_mini_batches": "number of minibatches per epoch",
        "learning_rate": "initial learning rate. If adaptive, it starts here and moves",
        "schedule": "if adaptive, lr is adjusted by desired_kl",
        "gamma": "discount factor",
        "lam": "GAE lambda",
        "desired_kl": "target KL. lr goes up and down relative to this value",
        "max_grad_norm": "gradient clipping",
        "use_clipped_value_loss": "also clips the critic prediction around the old value",
    }
    values = _read_cfg_values()
    if values:
        for k, v in values.items():
            print(f"  {k:<24} = {v.strip():<10}  {meaning.get(k, '')}")
    else:
        print("  (couldn't find the cfg file — you may be running outside the repo)")

    print(
        "\n  These values weren't picked arbitrarily; they match §III-B of the DreamWaQ paper."
    )

    print("\n" + "=" * 72)
    print(" 2. Adaptive learning rate — lr moves based on KL")
    print("=" * 72)

    desired_kl = 0.01
    lr = 1.0e-3
    # A KL trajectory commonly seen during training: moves a lot at first (policy changes rapidly), then gradually settles.
    kl_trace = [0.05, 0.03, 0.02, 0.012, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0]
    print(f"  desired_kl = {desired_kl},  starting lr = {lr:.2e}")
    print(f"  {'KL':>8}  {'verdict':<10} {'lr':>10}")
    for kl in kl_trace:
        new_lr = adapt_learning_rate(kl, lr, desired_kl)
        verdict = "lowered" if new_lr < lr else ("raised" if new_lr > lr else "kept")
        print(f"  {kl:8.4f}  {verdict:<10} {new_lr:10.3e}")
        lr = new_lr

    assert lr > 1e-3, "lr should rise as KL gets smaller"
    print(
        "\n  [PASS] Cautious when KL is large, bold when small — it sets its own step size."
    )

    print("\n" + "=" * 72)
    print(" 3. log_std clamp — a problem this repo actually ran into")
    print("=" * 72)
    print("""
  The actor's action distribution has a learnable log_std. But when the adaptive lr and
  a narrow observation distribution coincide, log_std drifts to extremes during training.
  Once it becomes ±inf, Normal sampling spits out NaN and the whole policy dies —
  and that's after hours of training.""")

    drift = torch.tensor([0.0, 0.0])
    print(f"\n  {'step':>6}  {'no clamp':>22}  {'with clamp':>22}")
    guarded = drift.clone()
    for step in range(1, 6):
        drift = drift - 2.0  # mimics a runaway gradient
        guarded = clamp_log_std(
            guarded - 2.0
        )  # same as the hook applied after every optimizer step
        print(
            f"  {step:6d}  log_std={drift[0]:8.2f} std={drift[0].exp():8.2e}"
            f"  log_std={guarded[0]:6.2f} std={guarded[0].exp():6.3f}"
        )

    nan_case = clamp_log_std(torch.tensor([float("-inf"), float("nan")]))
    print(f"\n  when -inf, nan come in -> {nan_case.tolist()}  (revived)")

    assert torch.isfinite(nan_case).all()
    assert float(guarded[0]) >= LOG_STD_MIN
    print(f"\n  [PASS] log_std stays within [{LOG_STD_MIN:.0f}, {LOG_STD_MAX:.0f}].")
    print("\n  Actual code: dreamwaq_manager/scripts/rsl_rl/train.py")
    print(
        "            _register_std_clamp_for_all_runners() — applied as a hook after the optimizer step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
