#!/usr/bin/env python3
"""Fast check — ppo-losses.  (no Isaac Sim needed, ~1 s)

    python check.py                 # starter.py
    python check.py --solution      # ../ppo.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE.parent / "ppo.py"
STARTER = HERE / "starter.py"

B = 256
CLIP = 0.2


def load(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with make_exercise.py --id ppo-losses")
    spec = importlib.util.spec_from_file_location(f"_ppo_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    losses = load(path).ppo_losses

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    torch.manual_seed(0)
    lp = torch.randn(B) * 0.1
    adv = torch.randn(B)
    val = torch.randn(B)
    ret = torch.randn(B)
    ent = torch.rand(B) + 0.5

    try:
        total, sur, vl, el = losses(lp, lp.clone(), adv, val, ret, ent)
    except NotImplementedError:
        print("  TODO(ppo-losses) is still empty. Fill in starter.py.")
        return 1

    check("1. all four values are scalars", all(x.dim() == 0 for x in (total, sur, vl, el)),
          f"{[tuple(x.shape) for x in (total, sur, vl, el)]}")

    # If the policy hasn't changed, ratio=1 -> surrogate = -mean(A)
    check("2. with ratio=1, surrogate = -mean(advantage)",
          torch.allclose(sur, -adv.mean(), atol=1e-5), f"{sur.item():.6f} vs {-adv.mean().item():.6f}")

    # value loss is MSE
    check("3. value_loss = mean((value - returns)^2)",
          torch.allclose(vl, (val - ret).pow(2).mean(), atol=1e-6), f"{vl.item():.6f}")

    # entropy enters the loss negatively
    check("4. entropy_loss = -mean(entropy)", torch.allclose(el, -ent.mean(), atol=1e-6),
          f"{el.item():.6f} vs {-ent.mean().item():.6f}")

    # total = surrogate + c_v * value + c_e * entropy
    t2, s2, v2, e2 = losses(lp, lp.clone(), adv, val, ret, ent,
                            value_loss_coef=2.0, entropy_coef=0.05)
    check("5. total = surrogate + c_v*value + c_e*entropy",
          torch.allclose(t2, s2 + 2.0 * v2 + 0.05 * e2, atol=1e-5),
          f"{t2.item():.6f} vs {(s2 + 2.0 * v2 + 0.05 * e2).item():.6f}")

    # --- Does the clipping actually kick in (the core of this exercise) ---
    # Make the ratio deviate a lot: log_prob - old = +1 -> ratio = e ~ 2.718
    old = torch.zeros(B)
    big = torch.ones(B)
    pos_adv = torch.ones(B)      # positive advantage
    neg_adv = -torch.ones(B)

    s_pos = losses(big, old, pos_adv, val, ret, ent, clip_param=CLIP)[1]
    # A>0, ratio>1+eps -> clipped, surrogate = -(1+eps)*A
    check("6. A>0, ratio above the upper bound gets clipped",
          torch.allclose(s_pos, torch.tensor(-(1.0 + CLIP)), atol=1e-5),
          f"{s_pos.item():.6f} (expected {-(1.0 + CLIP):.4f}) — without clipping it's -2.718")

    s_neg = losses(-big, old, neg_adv, val, ret, ent, clip_param=CLIP)[1]
    # A<0, ratio<1-eps -> clipped, surrogate = -(1-eps)*A = +(1-eps)
    check("7. A<0, ratio below the lower bound gets clipped",
          torch.allclose(s_neg, torch.tensor(1.0 - CLIP), atol=1e-5),
          f"{s_neg.item():.6f} (expected {1.0 - CLIP:.4f})")

    # If min() is omitted and only clip is used, this catches it:
    # with A>0 and the ratio deviating downward, the 'unclipped' side is smaller and must be used as-is.
    s_low = losses(-big, old, pos_adv, val, ret, ent, clip_param=CLIP)[1]
    ratio_low = torch.exp(-big - old)[0]
    check("8. A>0, ratio below the lower bound is not clipped (min was used)",
          torch.allclose(s_low, -ratio_low, atol=1e-5),
          f"{s_low.item():.6f} (expected {-ratio_low.item():.4f}) — using only clip gives {-(1 - CLIP):.4f}")

    # No clipping on the advantage
    s_a = losses(lp, lp.clone(), adv * 10.0, val, ret, ent)[1]
    check("9. the advantage is not clipped",
          torch.allclose(s_a, -(adv * 10.0).mean(), atol=1e-4), f"{s_a.item():.6f}")

    # Does the gradient flow
    lp_g = lp.clone().requires_grad_(True)
    val_g = val.clone().requires_grad_(True)
    losses(lp_g, lp.detach(), adv, val_g, ret, ent)[0].backward()
    check("10. gradient flows to log_prob and value",
          lp_g.grad is not None and val_g.grad is not None
          and lp_g.grad.abs().sum() > 0 and val_g.grad.abs().sum() > 0)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Now run real training with `python starter.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
