#!/usr/bin/env python3
"""Fast check — stage5-collapse-probe.

**Only synthetic checkpoints are graded.** Neither training outputs nor Isaac Sim are needed. The weights are hand-made,
so the correct answers are known analytically, and picking the wrong axis or the wrong probe distribution gets caught.

If real checkpoints exist, at the end it **only prints them — they aren't graded.** Which run is healthy
can't be known from file names (a run named healthy once collapsed later), and the moment you pick a threshold
to fit the runs, that threshold judges nothing. See the README.

    ~/IsaacLab/_isaac_sim/python.sh check.py              # check starter.py
    ~/IsaacLab/_isaac_sim/python.sh check.py --solution   # check the finished version
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

INPUT_DIM, HIDDEN1, HIDDEN2, OUT_DIM = 225, 128, 64, 35
LATENT_IN = 19  # decoder input = est_vel(3) + z(16)
BIG = 10.0  # bias that makes ELU act as the identity. The probability that 10 + N(0,1) is negative is practically 0.
E_ABS_NORMAL = math.sqrt(2.0 / math.pi)  # E|x|, x ~ N(0,1) = 0.7979


def load_module(path: Path):
    if not path.exists():
        raise SystemExit(
            f"File not found: {path}\n"
            "  generate it with python3 exercises/tools/make_exercise.py --id stage5-collapse-probe"
        )
    spec = importlib.util.spec_from_file_location(f"_s5d_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def synth(
    pick_row: int | None = None,
    logvar_bias: float = 0.0,
    dec_vel: float = 1.0,
    dec_z: float = 1.0,
    dec_cols: list[float] | None = None,
) -> dict:
    """Build a checkpoint whose correct answers are known analytically.

    Encoder: in the first two layers ELU acts as the identity thanks to the BIG bias, carrying only component 0
    of the input (h1[0] = BIG + x0, other units are 0). In the last layer, picking just one desired output row and
    cancelling the constant with `-BIG` makes that output exactly **x0 ~ N(0,1)**.
    With pick_row=None the last layer is all zeros — a 'dead' encoder that emits a constant whatever the input.

    Decoder: values are put only in row 0 of the first-layer weight, so **column norm = the absolute value of that value**.
    Give them per column with dec_cols, or the first 3 / last 16 columns at once with dec_vel/dec_z.
    """
    w0 = torch.zeros(HIDDEN1, INPUT_DIM); w0[0, 0] = 1.0
    w2 = torch.zeros(HIDDEN2, HIDDEN1); w2[0, 0] = 1.0
    w4 = torch.zeros(OUT_DIM, HIDDEN2)
    b4 = torch.zeros(OUT_DIM)
    b4[19:35] = logvar_bias
    if pick_row is not None:
        w4[pick_row, 0] = 1.0
        b4[pick_row] = -BIG

    cols = dec_cols if dec_cols is not None else [dec_vel] * 3 + [dec_z] * 16
    if len(cols) != LATENT_IN:
        raise AssertionError(f"dec_cols must have {LATENT_IN} entries")
    dec_w = torch.zeros(HIDDEN2, LATENT_IN)
    dec_w[0] = torch.tensor(cols)

    return {
        "encoder.0.weight": w0, "encoder.0.bias": torch.full((HIDDEN1,), BIG),
        "encoder.2.weight": w2, "encoder.2.bias": torch.zeros(HIDDEN2),
        "encoder.4.weight": w4, "encoder.4.bias": b4,
        "decoder.0.weight": dec_w, "decoder.0.bias": torch.zeros(HIDDEN2),
        # A real checkpoint also has the later layers mixed in — only the first Linear must be picked.
        "decoder.2.weight": torch.zeros(HIDDEN1, HIDDEN2), "decoder.2.bias": torch.zeros(HIDDEN1),
    }


def nan_checkpoint() -> dict:
    """A checkpoint where every tensor is NaN. Not a made-up situation — the flat Waq run of `_archive_3000it_stairheavy`
    was in exactly this state from it 500, and yet the reward curve looked fine."""
    return {k: torch.full_like(v, float("nan")) for k, v in synth().items()}


def alive_encoder(**kw) -> dict:
    """An encoder where all 16 mu dims follow the input — |mu| far exceeds the decision line."""
    sd = synth(**kw)
    sd["encoder.4.weight"][3:19, 0] = 1.0
    sd["encoder.4.bias"][3:19] = -BIG
    return sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path)
    probe = mod.diagnose_checkpoint

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def close(a: float, b: float, rel: float = 0.03) -> bool:
        return abs(a - b) <= rel * max(1.0, abs(b))

    try:
        r_mu = probe(synth(pick_row=3), num_samples=8192, seed=0)   # mu[0] = x0
    except NotImplementedError:
        print("  TODO(stage5-collapse-probe) is still empty. Fill in starter.py.")
        return 1

    # --- 1. Return contract -----------------------------------------------------------------
    nums = {"mu_abs", "sigma", "kl", "est_vel_std", "dec_w_vel", "dec_w_z"}
    check("1. returns all six numbers and mechanism", nums | {"mechanism"} <= set(r_mu),
          f"keys received: {sorted(r_mu)}")
    check("1. numbers are Python floats (not tensors)",
          all(isinstance(r_mu.get(k), float) for k in nums),
          f"{ {k: type(r_mu.get(k)).__name__ for k in sorted(nums)} }")
    check("1. mechanism is one of MECHANISMS", r_mu.get("mechanism") in mod.MECHANISMS,
          f"{r_mu.get('mechanism')!r} — possible values: {mod.MECHANISMS}")

    # --- 2. The encoder probe is standard normal ---------------------------------------------------
    # Only one of the 16 mu dims is x0 and the rest are 0, so  E|mu| = E|x0| / 16.
    # Feeding a uniform distribution or constant input doesn't give this value.
    check("2. probe input is N(0,1)", close(r_mu["mu_abs"], E_ABS_NORMAL / 16),
          f"|mu|={r_mu['mu_abs']:.5f} (expected {E_ABS_NORMAL / 16:.5f})")

    # --- 3. The KL is a 16-dim 'sum' --------------------------------------------------------
    # With logvar=0, mu[0]=x0:  KL = 0.5 * sum mu_j^2 = 0.5 * x0^2  ->  expected value 0.5.
    check("3. KL = -0.5*sum(1+logvar-mu^2-exp(logvar))", close(r_mu["kl"], 0.5),
          f"kl={r_mu['kl']:.4f} (expected 0.500). If you divided by 16 you'd get 0.031")
    # Encoder with only logvar raised to the constant 0.5:  KL = 0.5*16*(exp(.5) - 1 - .5) = 1.18977
    r_lv = probe(synth(logvar_bias=0.5), num_samples=4096, seed=0)
    check("3. the logvar term also goes into the KL", close(r_lv["kl"], 8 * (math.exp(0.5) - 1.5)),
          f"kl={r_lv['kl']:.4f} (expected {8 * (math.exp(0.5) - 1.5):.4f})")
    check("3. sigma = mean of exp(0.5*logvar)", close(r_lv["sigma"], math.exp(0.25)),
          f"sigma={r_lv['sigma']:.4f} (expected {math.exp(0.25):.4f}) — using logvar as-is gives 0.5")

    # --- 4. est_vel_std is the std over the 'sample axis' -------------------------------------------
    r_vel = probe(synth(pick_row=0), num_samples=8192, seed=0)  # est_vel[0] = x0
    check("4. est_vel_std = mean over 3 dims of the sample-axis std", close(r_vel["est_vel_std"], 1.0 / 3),
          f"{r_vel['est_vel_std']:.4f} (expected {1 / 3:.4f}) — taking the std over the dimension axis gives a different value")
    check("4. est_vel_std is 0 if the output is constant", r_mu["est_vel_std"] < 1e-6,
          f"{r_mu['est_vel_std']:.3e}")

    # --- 5. Decoder column norms -----------------------------------------------------------------
    # This is the new axis this exercise measures. Looking only at the encoder, the two collapses can't be told apart.
    r_w = probe(synth(dec_vel=2.0, dec_z=0.5), num_samples=64, seed=0)
    check("5. dec_w_vel / dec_w_z are means of column norms",
          close(r_w["dec_w_vel"], 2.0) and close(r_w["dec_w_z"], 0.5),
          f"|W_vel|={r_w['dec_w_vel']:.4f} (expected 2.0), |W_z|={r_w['dec_w_z']:.4f} (expected 0.5)")
    # Columns 0,1,2 are 1.0, only column 3 is 10.0, the other 15 are 0. A correct 3/16 split gives (1.0, 0.625).
    # A wrong 4/15 split gives (3.25, 0.0).
    r_split = probe(synth(dec_cols=[1.0, 1.0, 1.0, 10.0] + [0.0] * 15), num_samples=64, seed=0)
    check("5. the first 3 columns are v̂, the last 16 are z (caught if the boundary shifts by one)",
          close(r_split["dec_w_vel"], 1.0) and close(r_split["dec_w_z"], 10.0 / 16),
          f"|W_vel|={r_split['dec_w_vel']:.4f} (expected 1.0), "
          f"|W_z|={r_split['dec_w_z']:.4f} (expected {10 / 16:.4f}); a 4/15 split gives 3.25 / 0.0")

    # --- 6. Tell the two mechanisms apart --------------------------------------------------------------
    # The synthetic values are kept more than an order of magnitude away from the decision lines. The measured passing ranges are
    #   MU_ALIVE : 0 < x < 0.81      (the collapsed synthetic |mu| is exactly 0, so the lower side is infinite)
    #   Z_USED   : 4.30e-4 < x <= 0.435  (the narrowest side is 23x = 1.4 orders of magnitude from the default 1e-2)
    # — "independent of the threshold" is wrong; "more than an order of magnitude of margin" is the accurate phrasing.
    r_alive = probe(alive_encoder(), num_samples=2048, seed=0)
    check("6. alive if mu follows the input", r_alive["mechanism"] == "alive",
          f"{r_alive['mechanism']!r} (|mu|={r_alive['mu_abs']:.3e})")
    # The decoder reads z strongly (|W_z|/|W_vel| = 0.44) but the encoder's mu is dead = KL overpressure.
    r_kl = probe(synth(dec_vel=4.39, dec_z=1.91), num_samples=2048, seed=0)
    check("6. kl_overpressure if mu is dead but |W_z| is alive",
          r_kl["mechanism"] == "kl_overpressure",
          f"{r_kl['mechanism']!r} (|mu|={r_kl['mu_abs']:.3e}, |W_z|/|W_vel|="
          f"{r_kl['dec_w_z'] / r_kl['dec_w_vel']:.3f})")
    # The decoder reads only v̂ (|W_z|/|W_vel| = 4e-4) = the decoder dropped z.
    r_dec = probe(synth(dec_vel=10.24, dec_z=0.0044), num_samples=2048, seed=0)
    check("6. decoder_dropped_z if both mu and |W_z| are dead",
          r_dec["mechanism"] == "decoder_dropped_z",
          f"{r_dec['mechanism']!r} (|mu|={r_dec['mu_abs']:.3e}, |W_z|/|W_vel|="
          f"{r_dec['dec_w_z'] / r_dec['dec_w_vel']:.2e})")
    # NaN is False in every comparison, so unless filtered it tumbles down into the final else and
    # ends up explaining a mechanism that doesn't exist.
    r_nan = probe(nan_checkpoint(), num_samples=64, seed=0)
    check("6. labeled nan if the weights are NaN", r_nan["mechanism"] == "nan",
          f"{r_nan['mechanism']!r} — without filtering NaN first, it leaks out as decoder_dropped_z")
    # Verdict order: if mu is alive, it's not a collapse even if the decoder doesn't read z, because the actor receives
    # the context, not the decoder (|W_z| dying is a sign that it 'will die later').
    #
    # Of the two combinations below, only '|mu| alive + |W_z| alive' actually catches the order.
    # '|mu| alive + |W_z| dead' gives the same answer even if the ladder order is flipped (|W_z| branch before |mu|),
    # so it can't check the order — that's why both are checked.
    r_order = probe(alive_encoder(dec_vel=4.39, dec_z=1.91), num_samples=2048, seed=0)
    check("6. mu comes first in the verdict order (mu is checked first even if |W_z| is alive)",
          r_order["mechanism"] == "alive",
          f"{r_order['mechanism']!r} — putting the |W_z| branch before the |mu| branch gives kl_overpressure. "
          "Check |mu| first, and only when dead split the mechanism via |W_z|")
    r_order2 = probe(alive_encoder(dec_vel=10.0, dec_z=0.001), num_samples=2048, seed=0)
    check("6. alive if mu is alive even if |W_z| is dead (it's still only a sign)",
          r_order2["mechanism"] == "alive",
          f"{r_order2['mechanism']!r} — what the actor receives is the context, not the decoder output")

    # --- 7. Must be reproducible -------------------------------------------------------------
    sd = synth(pick_row=3)
    a = probe(sd, num_samples=64, seed=1)
    b = probe(sd, num_samples=64, seed=1)
    c = probe(sd, num_samples=64, seed=2)
    check("7. the same seed gives the same numbers", a == b, f"{a} vs {b}")
    check("7. a different seed uses different samples", a["mu_abs"] != c["mu_abs"],
          "you may have hard-coded a constant instead of using the seed argument")

    # Check not 'whether the global RNG was touched' but 'whether it was touched differently depending on the probe seed'.
    # build_encoder() consumes a fixed amount of the global RNG while creating nn.Linear, so a before/after
    # comparison would FAIL even the correct answer. If probe uses the global seed, the two states diverge.
    torch.manual_seed(1234)
    probe(sd, num_samples=64, seed=7)
    state_a = torch.random.get_rng_state()
    torch.manual_seed(1234)
    probe(sd, num_samples=64, seed=99)
    state_b = torch.random.get_rng_state()
    check("7. doesn't pollute the global RNG state", torch.equal(state_a, state_b),
          "torch.manual_seed / torch.randn was used globally — use a Generator")

    # --- 8. num_samples is the real sample count ---------------------------------------------------
    def spread(n: int) -> float:
        vals = [probe(sd, num_samples=n, seed=s)["mu_abs"] for s in range(5)]
        return max(vals) - min(vals)

    s_small, s_big = spread(32), spread(8192)
    check("8. increasing num_samples stabilizes the estimate", s_small > 3 * s_big,
          f"spread over 5 seeds: {s_small:.2e} at n=32, {s_big:.2e} at n=8192 — "
          "you may have ignored the sample count and used a constant")

    # --- 9. Real checkpoints: print only (not graded) ------------------------------------
    print("\n  --- Real checkpoints (prints only the last one if present. Not graded) ---")
    runs = mod.find_runs()
    if not runs:
        print("  [skipped] couldn't find a Waq run under logs/.")
        print("           Training outputs are gitignored and not in the repo — passing the synthetic checks above is enough.")
    for run in runs:
        ckpts = mod.checkpoints(run)
        blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        if "cenet_state_dict" not in blob:
            continue
        m = mod.diagnose_checkpoint(blob["cenet_state_dict"], num_samples=2048, seed=0)
        print(f"  {mod.label_for(run)}")
        print(f"    {ckpts[-1].name}: |mu|={m['mu_abs']:.3e}  |W_vel|={m['dec_w_vel']:.3f}  "
              f"|W_z|={m['dec_w_z']:.4f}  →  {m['mechanism']}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m You can diagnose your own training runs with the same probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
