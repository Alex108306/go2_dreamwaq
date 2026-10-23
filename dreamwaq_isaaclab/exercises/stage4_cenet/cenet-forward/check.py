#!/usr/bin/env python3
"""Fast check — cenet-forward.

Runs on pure torch tensors without Isaac Sim. Done in 1 second.

    python check.py                 # check starter/cenet.py
    python check.py --solution      # check the finished version (to test the checker itself)
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = (
    REPO_ROOT
    / "dreamwaq_manager/source/dreamwaq_manager/dreamwaq_manager/algorithms/cenet.py"
)
STARTER = HERE / "starter" / "cenet.py"

OBS_HISTORY, LATENT, NUM_VEL, NUM_OBS, BATCH = 225, 16, 3, 45, 64


def load_cenet(path: Path):
    """Import a CENet module from an arbitrary path without touching sys.path."""
    if not path.exists():
        raise SystemExit(f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id cenet-forward")
    spec = importlib.util.spec_from_file_location(f"_cenet_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CENet


def make_net(CENet, latent_dim1: int = NUM_VEL + LATENT * 2):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):  # CENet.__init__ prints its MLP structure
        return CENet(
            input_dim=OBS_HISTORY,
            latent_dim1=latent_dim1,
            latent_dim2=NUM_VEL + LATENT,
            output_dim=NUM_OBS,
            device="cpu",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    CENet = load_cenet(path)

    torch.manual_seed(0)
    obs_hist = torch.randn(BATCH, OBS_HISTORY)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    net = make_net(CENet)
    try:
        out = net.forward(obs_hist)
    except NotImplementedError:
        print("  TODO(cenet-forward) is still empty. Fill in starter/cenet.py.")
        return 1

    # --- 1. Return form -----------------------------------------------------------
    ok_tuple = isinstance(out, tuple) and len(out) == 5
    check("1. returns 5 values (est_next_obs, est_vel, mu, logvar, context_vec)", ok_tuple,
          f"received: {type(out).__name__} len={len(out) if hasattr(out, '__len__') else '?'}")
    if not ok_tuple:
        print("\n\033[1;31mThe return form differs, so the remaining checks can't proceed.\033[0m")
        return 1
    est_next_obs, est_vel, mu, logvar, context_vec = out

    # --- 2. shape ---------------------------------------------------------------
    shapes = {
        "est_next_obs (B,45)": (est_next_obs, (BATCH, NUM_OBS)),
        "est_vel (B,3)": (est_vel, (BATCH, NUM_VEL)),
        "mu (B,16)": (mu, (BATCH, LATENT)),
        "logvar (B,16)": (logvar, (BATCH, LATENT)),
        "context_vec (B,16)": (context_vec, (BATCH, LATENT)),
    }
    for label, (tensor, want) in shapes.items():
        check(f"2. {label}", tuple(tensor.shape) == want, f"actual {tuple(tensor.shape)}")

    # --- 3. Was the encoder output split exactly into 3 / 16 / 16 -------------------------
    # The encoder is deterministic, so run it directly to build the expected values.
    h = net.encoder(obs_hist)
    check("3. est_vel is the first 3 dims of the encoder output",
          torch.allclose(est_vel, h[:, :NUM_VEL], atol=1e-6))
    check("3. mu is the next 16 dims",
          torch.allclose(mu, h[:, NUM_VEL:NUM_VEL + LATENT], atol=1e-6))
    check("3. logvar is the last 16 dims",
          torch.allclose(logvar, h[:, NUM_VEL + LATENT:], atol=1e-6),
          "mu and logvar may be swapped")

    # --- 4. Is what went into the decoder [est_vel, context_vec] ----------------------
    # Attach a forward pre-hook to the decoder to capture its input.
    seen: dict[str, torch.Tensor] = {}
    net2 = make_net(CENet)
    net2.decoder.register_forward_pre_hook(lambda _mod, args: seen.update(latent=args[0]))
    _, est_vel2, mu2, _, ctx2 = net2.forward(obs_hist)
    latent = seen.get("latent")
    check("4. decoder input is 19-dim (3 + 16)",
          latent is not None and tuple(latent.shape) == (BATCH, NUM_VEL + LATENT),
          f"actual {None if latent is None else tuple(latent.shape)}")
    if latent is not None and tuple(latent.shape) == (BATCH, NUM_VEL + LATENT):
        check("4. first 3 of the decoder input = est_vel",
              torch.allclose(latent[:, :NUM_VEL], est_vel2, atol=1e-6))
        check("4. last 16 of the decoder input = context_vec (not mu)",
              torch.allclose(latent[:, NUM_VEL:], ctx2, atol=1e-6),
              "you may have concatenated mu as-is instead of the reparameterize result")
        check("4. mu was not fed into the decoder as-is",
              not torch.allclose(latent[:, NUM_VEL:], mu2, atol=1e-6))

    # --- 5. Does the reparameterization actually sample --------------------------------------
    # Feeding the same input twice, mu/logvar must be the same and only context_vec should differ.
    net3 = make_net(CENet)
    _, _, mu_a, lv_a, ctx_a = net3.forward(obs_hist)
    _, _, mu_b, lv_b, ctx_b = net3.forward(obs_hist)
    check("5. mu / logvar are deterministic for the same input",
          torch.allclose(mu_a, mu_b, atol=1e-6) and torch.allclose(lv_a, lv_b, atol=1e-6))
    check("5. context_vec differs on every call (sampling)",
          not torch.allclose(ctx_a, ctx_b, atol=1e-6),
          "you may have passed mu as-is without using reparameterize")

    # --- 6. AssertionError if the context parameters are odd ------------------------------
    net4 = make_net(CENet, latent_dim1=NUM_VEL + 33)  # 33 can't be split in half
    try:
        net4.forward(obs_hist)
        raised = False
    except AssertionError:
        raised = True
    except Exception:  # noqa: BLE001 — other exceptions don't count as a pass
        raised = False
    check("6. AssertionError if the context parameters are odd", raised)

    # --- 7. Gradient flows to both encoder and decoder ------------------------------
    net5 = make_net(CENet)
    est_next_obs5, est_vel5, _, _, _ = net5.forward(obs_hist)
    (est_next_obs5.sum() + est_vel5.sum()).backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net5.encoder.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net5.decoder.parameters())
    check("7. gradient flows to the encoder", enc_grad)
    check("7. gradient flows to the decoder", dec_grad)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with the cenet-loss exercise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
