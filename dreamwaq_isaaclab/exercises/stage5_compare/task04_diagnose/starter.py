#!/usr/bin/env python3
"""Stage 5 · task04 — Judge whether the CENet latent died, and if so, **which side died first**.  `L2 · BUILD`

This project's CENet latent variable z (16-dim) **collapsed twice, and the two mechanisms were different.**

    (a) KL overpressure      The decoder still tries to read z (|W_z| is large) but the encoder freezes.
                             This happens when the KL term's pressure is stronger than the signal recon sends to z.
    (b) The decoder          v̂ is directly supervised by vel_loss, so it beats z in the reconstruction competition.
        drops z              Once the decoder starts reading only v̂, the recon gradient to z disappears,
                             and then z dies even under a weak KL.

Both look exactly the same from the outside — `|mu| → 0`, `sigma → 1`, `KL → 0`. **Looking only at the encoder, they can't be
told apart.** Even though the prescriptions are opposite: (a) needs β lowered, while for (b) lowering β doesn't help
(z has to be given work to do). So this exercise **also measures the decoder's first-layer weights.**

    column j of the decoder's first-layer weight W  =  how much the decoder reads from input j
    the input is [v̂(3), z(16)], so  |W_vel| = first 3 columns, |W_z| = last 16 columns

Neither Isaac Sim nor a simulator is needed. A checkpoint's `cenet_state_dict` is enough; it's done in
pure torch.

    python starter.py                    # if the assets are extracted, finds and scans the Waq runs
    python starter.py --run <path>       # give a run folder directly (can be given multiple times)
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

# CENet structure (must match algorithms/cenet.py)
INPUT_DIM = 225  # obs_history = 45 x H(5)
HIDDEN1, HIDDEN2 = 128, 64
NUM_VEL, NUM_LATENT = 3, 16
OUT_DIM = NUM_VEL + 2 * NUM_LATENT  # 35 = est_vel(3) + mu(16) + logvar(16)

# Two decision lines. Both are **reading aids, not truth** — why you mustn't conclude health from a threshold
# is written in the README's 'Don't conclude health from a threshold' section.
MU_ALIVE = 1e-2  # if |mu| is above this, the encoder is responding to its input
Z_USED = 1e-2  # if |W_z| / |W_vel| is above this, the decoder is still reading z

MECHANISMS = ("alive", "kl_overpressure", "decoder_dropped_z", "nan")
VERDICTS = {
    "alive": "alive — the encoder responds to its input",
    "kl_overpressure": (
        "collapsed (KL overpressure) — the decoder still wants z (|W_z| is alive) but the encoder's mu was pressed down. "
        "Lowering β is the prescription"
    ),
    "decoder_dropped_z": (
        "collapsed (the decoder dropped z) — |W_z| died to 0 and only the v̂ side grew. "
        "Lowering β won't revive it. z needs work to do"
    ),
    "nan": (
        "CENet weights are NaN — no point discussing the latent. The runner feeds the actor observation through "
        "nan_to_num, replacing it with 0, so the reward curve looks fine"
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS = REPO_ROOT / "dreamwaq_manager" / "logs"
# Only decide where to search. **Which run is healthy is not written here** — that's what this script
# is meant to judge, not a value known in advance. Missing folders are silently skipped.
SEARCH_ROOTS = [
    LOGS / "rsl_rl",
    LOGS / "_archive_4000it_collapsedwaq",
    LOGS / "_archive_4000it_clip1_norew",
    LOGS / "_archive_3000it_stairheavy" / "rsl_rl",
]


def build_encoder(cenet_state_dict: dict) -> nn.Sequential:
    """Revive just the encoder from a checkpoint's `cenet_state_dict`."""
    enc = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN1), nn.ELU(),
        nn.Linear(HIDDEN1, HIDDEN2), nn.ELU(),
        nn.Linear(HIDDEN2, OUT_DIM),
    )
    weights = {k[len("encoder."):]: v for k, v in cenet_state_dict.items() if k.startswith("encoder.")}
    if not weights:
        raise KeyError("no 'encoder.*' keys in state_dict — check that it's really a CENet checkpoint")
    enc.load_state_dict(weights)
    enc.eval()
    return enc


def decoder_input_weight(cenet_state_dict: dict) -> torch.Tensor:
    """Extract the decoder's **first Linear** weight `[hidden, 19]`.

    It's usually `decoder.0.weight`, but the index isn't hard-coded — if an experiment gate is inserted in front,
    the numbers shift by one. Pick the 2-dim `decoder.*.weight` with the smallest number.
    """
    found = [
        (int(m.group(1)), k)
        for k, v in cenet_state_dict.items()
        if (m := re.fullmatch(r"decoder\.(\d+)\.weight", k)) and v.ndim == 2
    ]
    if not found:
        raise KeyError("no 'decoder.<i>.weight' in state_dict — check that it's really a CENet checkpoint")
    return cenet_state_dict[min(found)[1]]


def diagnose_checkpoint(cenet_state_dict: dict, num_samples: int = 2048, seed: int = 0) -> dict:
    """Diagnose one checkpoint — encoder sensitivity + decoder usage + mechanism verdict.

    Args:
        cenet_state_dict: the checkpoint's `cenet_state_dict`.
        num_samples: number of probes to feed the encoder. More gives more stable statistics.
        seed: probe RNG seed. **The same seed must give the same numbers** — the verdict must be reproducible,
            and the global RNG state must not be touched either.

    Returns:
        {# --- Encoder: does the output move when the input changes ---------------------------------
         "mu_abs":      mean of |mu|                 pressed to 0 on collapse  (key metric)
         "sigma":       mean of exp(0.5*logvar)      pressed to 1 on collapse
         "kl":          KL(q||N(0,I)), 16-dim **sum**, sample mean, in nats. 0 on collapse
         "est_vel_std": mean std of the 3 estimated-velocity dims. On collapse it doesn't move even when the input changes
         # --- Decoder: what is it reading ----------------------------------------------
         "dec_w_vel":   mean column norm of the **first 3 columns** of the decoder's first-layer weight (how much it reads v̂)
         "dec_w_z":     mean column norm of the **last 16 columns** of the decoder's first-layer weight (how much it reads z)
         # --- Combine both to name the mechanism --------------------------------------------
         "mechanism":   one of MECHANISMS (string)}

    There are four `mechanism`s — `alive` / `kl_overpressure` / `decoder_dropped_z` / `nan`.
    The last exists because it actually happened: in `_archive_3000it_stairheavy`, the flat Waq run had
    **all 14 CENet tensors NaN** from it 500, yet the reward rose to 35. That's because the runner washes the actor
    observation with `nan_to_num(..., nan=0.0)` (`dreamwaq_runner.py:203`) — i.e. that run was
    "Base + 19 zeros" for all 3000 iterations. Without filtering NaN first, this run would be
    labeled `decoder_dropped_z`, explaining a mechanism that doesn't exist.

    Why N(0,1) probes rather than real observations: what's measured isn't "accuracy" but **input sensitivity**.
    Observations go in normalized, so N(0,1) is a plausible input distribution, and without a simulator
    the same yardstick can be applied to any checkpoint.

    Why weights rather than probes for the decoder: how much the decoder reads from input j is entirely contained in
    **the norm of column j** of the first-layer weight. If a column is 0, that input has no effect on anything downstream.
    """
    # ── TODO(stage5-collapse-probe) ─ level L2 ─────────────────────────────
    # Measure encoder sensitivity and decoder usage from one CENet checkpoint, and tell the two collapse mechanisms apart
    #   hint: draw random numbers from a torch.Generator made from seed, not the global torch.manual_seed — the verdict must not change someone else's RNG state
    #   hint: all numbers are Python floats; only mechanism is a string (returning tensors as-is gets caught by the checker)
    #   The remaining hints are folded in the 'Hints' section of README.md — try without them first.
    # For the pass criteria, see README.md in this exercise folder.
    raise NotImplementedError("TODO(stage5-collapse-probe)")
    # ─────────────────────────────────────────────────────────────────────


def probe_encoder(cenet_state_dict: dict, num_samples: int = 2048, seed: int = 0) -> dict[str, float]:
    """Pick out only the 4 encoder metrics. `PAPER.md` §4 cites the probe definition by these names."""
    m = diagnose_checkpoint(cenet_state_dict, num_samples, seed)
    return {k: m[k] for k in ("mu_abs", "sigma", "kl", "est_vel_std")}


def checkpoints(run_dir: Path) -> list[Path]:
    """Yield the run folder's `model_<iter>.pt` in iteration order.

    Filename sorting isn't used — sorting as strings puts `model_1000` before `model_500`.
    """
    found = [(int(m.group(1)), p) for p in run_dir.glob("model_*.pt")
             if (m := re.fullmatch(r"model_(\d+)", p.stem))]
    return [p for _, p in sorted(found)]


def probe_run(run_dir: Path, num_samples: int = 2048, seed: int = 0) -> list[tuple[int, dict]]:
    """Scan every checkpoint in the run folder. [(iteration, metrics), ...]"""
    trace = []
    for ckpt in checkpoints(run_dir):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "cenet_state_dict" not in blob:
            continue  # Base / Oracle runs have no CENet
        trace.append((int(blob.get("iter", -1)), diagnose_checkpoint(blob["cenet_state_dict"], num_samples, seed)))
    return trace


def verdict(trace: list[tuple[int, dict]]) -> str:
    """Judge by the last checkpoint.

    **Why the last one**: collapse happens during training, not right after initialization. The archived collapsed runs
    had a healthy |mu| of 3e-2 ~ 1e-1 at it 0. Had we judged by the first checkpoint, we'd have written that they were all
    healthy — the same kind of mistake as task02's 'averaging window' lesson.
    """
    if not trace:
        return "can't judge — no CENet checkpoints"
    return VERDICTS[trace[-1][1]["mechanism"]]


def find_runs(roots: list[Path] | None = None) -> list[Path]:
    """Collect run folders under `<root>/<experiment_name>/<timestamp>/` that likely have a CENet.

    Only looks at experiment names containing `Waq` (Base/Oracle checkpoints have no CENet).
    One most-recent timestamp per experiment.
    """
    out: list[Path] = []
    for root in roots or SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for exp in sorted(root.iterdir()):
            if not exp.is_dir() or "Waq" not in exp.name:
                continue
            stamps = sorted(p for p in exp.iterdir() if p.is_dir() and any(p.glob("model_*.pt")))
            if stamps:
                out.append(stamps[-1])
    return out


def label_for(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(LOGS))
    except ValueError:
        return str(run_dir)


def report(run_dir: Path, num_samples: int, seed: int) -> None:
    print(f"\n=== {label_for(run_dir)}")
    if not run_dir.is_dir():
        print("    [skipped] folder not found (see the 'Assets' section of README.md)")
        return
    trace = probe_run(run_dir, num_samples, seed)
    if not trace:
        print("    [skipped] no checkpoints with a `cenet_state_dict`")
        return
    print(f"    {'iter':>6} {'|mu|':>10} {'sigma':>8} {'KL(nats)':>10} "
          f"{'|W_vel|':>9} {'|W_z|':>9} {'W_z/W_vel':>10}  mechanism")
    for it, m in trace:
        ratio = m["dec_w_z"] / m["dec_w_vel"] if m["dec_w_vel"] else float("nan")
        print(f"    {it:>6} {m['mu_abs']:>10.3e} {m['sigma']:>8.4f} {m['kl']:>10.3e} "
              f"{m['dec_w_vel']:>9.4f} {m['dec_w_z']:>9.4f} {ratio:>10.2e}  {m['mechanism']}")
    print(f"    verdict: {verdict(trace)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CENet checkpoint forensics")
    ap.add_argument("--run", type=Path, action="append", default=None,
                    help="run folder. Can be given multiple times. If omitted, looks for Waq runs under logs/")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = args.run or find_runs()
    if not runs:
        print("No runs to check. Extract the training outputs under dreamwaq_manager/logs/ or")
        print("  specify one directly with --run <folder> (the 'Assets' section of README.md).")
        return 0

    for run in runs:
        report(run, args.num_samples, args.seed)

    print("\nWith |mu| alone, the two collapses look the same. What separates them is the decoder's |W_z| (PAPER.md §4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
