#!/usr/bin/env python3
"""Fast check — vae-core.

Runs in pure torch without Isaac Sim. Done in 1 second.

    python check.py                 # check starter.py
    python check.py --solution      # check the finished version (to test the checker itself)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOLUTION = HERE / "solution.py"
STARTER = HERE / "starter.py"

INPUT_DIM, LATENT_DIM, BATCH = 8, 2, 64
HIDDEN_DIM = 32


def load_module(path: Path):
    """Since this is L3, the starter raises at module top level — the import itself may fail."""
    if not path.exists():
        raise SystemExit(
            f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id vae-core"
        )
    spec = importlib.util.spec_from_file_location(f"_vae_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--solution", action="store_true", help="check the finished version"
    )
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    try:
        mod = load_module(path)
    except NotImplementedError:
        print("  TODO(vae-core) is still empty. Write ToyVAE in starter.py.")
        return 1

    if not hasattr(mod, "ToyVAE"):
        print("  There's no ToyVAE class. Match the API in README.md exactly.")
        return 1
    ToyVAE = mod.ToyVAE

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def make(beta: float = 1.0):
        torch.manual_seed(0)
        return ToyVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, beta=beta)

    torch.manual_seed(0)
    x = torch.randn(BATCH, INPUT_DIM)

    # --- 1. encode → (mu, logvar) -----------------------------------------------
    net = make()
    mu, logvar = net.encode(x)
    check(
        "1. encode returns mu, logvar",
        tuple(mu.shape) == (BATCH, LATENT_DIM)
        and tuple(logvar.shape) == (BATCH, LATENT_DIM),
        f"mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}",
    )
    check(
        "1. mu and logvar are different values (the same half wasn't used twice)",
        not torch.allclose(mu, logvar, atol=1e-6),
    )

    # --- 2. reparameterize ------------------------------------------------------
    mu_f = torch.zeros(BATCH, LATENT_DIM, requires_grad=True)
    lv_tiny = torch.full((BATCH, LATENT_DIM), -30.0)  # std ~ 0 → z ~ mu
    z_tiny = net.reparameterize(mu_f, lv_tiny)
    check(
        "2. with very small logvar, z ≈ mu (std = exp(0.5*logvar))",
        torch.allclose(z_tiny, mu_f, atol=1e-4),
        f"max error {(z_tiny - mu_f).abs().max():.3e}",
    )

    lv0 = torch.zeros(BATCH, LATENT_DIM)
    z_a = net.reparameterize(mu_f, lv0)
    z_b = net.reparameterize(mu_f, lv0)
    check(
        "2. calling twice with the same arguments gives different samples",
        not torch.allclose(z_a, z_b, atol=1e-6),
        "random sampling is missing",
    )
    check(
        "2. the sample standard deviation is about 1 (logvar=0 → std=1)",
        0.8 < float(z_a.std().detach()) < 1.25,
        f"actual {float(z_a.std().detach()):.3f}",
    )

    z_a.sum().backward()
    check(
        "2. gradient flows to mu (reparameterization trick)",
        mu_f.grad is not None and float(mu_f.grad.abs().sum()) > 0,
    )

    # --- 3. decode / forward ----------------------------------------------------
    net = make()
    x_hat = net.decode(torch.randn(BATCH, LATENT_DIM))
    check(
        "3. decode: (B, latent) → (B, input)",
        tuple(x_hat.shape) == (BATCH, INPUT_DIM),
        f"actual {tuple(x_hat.shape)}",
    )

    out = net.forward(x)
    ok4 = isinstance(out, tuple) and len(out) == 4
    check(
        "3. forward returns 4 values (x_hat, mu, logvar, z)",
        ok4,
        f"len={len(out) if hasattr(out, '__len__') else '?'}",
    )
    if ok4:
        xh, m, lv, z = out
        check(
            "3. forward return shapes are correct",
            tuple(xh.shape) == (BATCH, INPUT_DIM)
            and tuple(m.shape) == (BATCH, LATENT_DIM)
            and tuple(lv.shape) == (BATCH, LATENT_DIM)
            and tuple(z.shape) == (BATCH, LATENT_DIM),
        )
        # Writing decode(mu) passes all the checks above (the shapes are the same). Whether sampling is
        # actually on the forward path only shows when the same x is fed in twice.
        xh2, _, _, _ = net.forward(x)
        check(
            "3. calling forward twice with the same x gives different x_hat",
            not torch.allclose(xh, xh2, atol=1e-6),
            "sampling is not on the forward path — you may have used decode(mu) instead of decode(z)",
        )

    # --- 4. ELBO ----------------------------------------------------------------
    net = make()
    total, recon, kl = net.loss(x)
    check(
        "4. loss returns (total, recon, kl)",
        all(torch.is_tensor(t) and t.dim() == 0 for t in (total, recon, kl)),
    )
    check(
        "4. total == recon + kl",
        torch.allclose(total, recon + kl, atol=1e-5),
        f"total={total.item():.6f} vs {recon.item() + kl.item():.6f}",
    )

    # If q == p then KL == 0. Intercept encode to force mu=0, logvar=0.
    net = make()
    zeros = torch.zeros(BATCH, LATENT_DIM)
    net.encode = lambda _x: (
        zeros.clone().requires_grad_(True),
        zeros.clone().requires_grad_(True),
    )
    _, _, kl0 = net.loss(x)
    check(
        "4. kl == 0 when mu=0, logvar=0", abs(kl0.item()) < 1e-6, f"kl={kl0.item():.6f}"
    )

    def kl_with(
        mu_val: float, lv_val: float, beta: float = 1.0, latent: int = LATENT_DIM
    ) -> float:
        torch.manual_seed(0)
        n = ToyVAE(input_dim=INPUT_DIM, latent_dim=latent, beta=beta)
        m = torch.full((BATCH, latent), mu_val, requires_grad=True)
        lv = torch.full((BATCH, latent), lv_val, requires_grad=True)
        n.encode = lambda _x: (m, lv)
        return n.loss(x)[2].item()

    check(
        "4. kl increases as mu moves away from 0",
        kl_with(1.0, 0.0) > kl_with(0.0, 0.0) + 1e-6,
    )
    check(
        "4. kl increases as logvar moves away from 0",
        kl_with(0.0, 1.5) > kl_with(0.0, 0.0) + 1e-6,
    )
    # recon is nn.MSELoss() = per-dimension mean. KL must use the same axis so the ratio of the two terms
    # isn't swayed by the number of dimensions. How to tell: keep the per-dim value fixed and only increase the latent dims —
    # a mean stays the same, a sum grows proportionally.
    check(
        "4. kl is 'averaged' over latent dims (mu=1, logvar=0 → 0.5)",
        abs(kl_with(1.0, 0.0) - 0.5) < 1e-4,
        f"expected 0.5 but got {kl_with(1.0, 0.0):.4f} — a sum would give {LATENT_DIM * 0.5}",
    )
    check(
        "4. kl stays the same when latent dims go 2 → 4 (a sum would double)",
        abs(kl_with(1.0, 0.0) - kl_with(1.0, 0.0, latent=LATENT_DIM * 2)) < 1e-4,
        f"latent {LATENT_DIM} → {kl_with(1.0, 0.0):.4f}, "
        f"latent {LATENT_DIM * 2} → {kl_with(1.0, 0.0, latent=LATENT_DIM * 2):.4f}",
    )

    # --- 5. beta applies only to KL --------------------------------------------------------
    k1, k2 = kl_with(1.0, 0.5, beta=1.0), kl_with(1.0, 0.5, beta=2.0)
    check("5. beta 2x → kl exactly 2x", abs(k2 - 2 * k1) < 1e-4, f"{k1:.6f} → {k2:.6f}")

    # --- 6. Perfect reconstruction gives recon == 0 -------------------------------------------------
    net = make()
    net.decode = lambda _z: x
    _, recon0, _ = net.loss(x)
    check(
        "6. recon == 0 if x_hat == x",
        abs(recon0.item()) < 1e-6,
        f"recon={recon0.item():.6f}",
    )

    # --- 7. It actually learns --------------------------------------------------------
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    data = mod.make_toy_batch(256, seed=1)
    first = None
    for step in range(200):
        t, _, _ = net.loss(data)
        opt.zero_grad()
        t.backward()
        opt.step()
        if step == 0:
            first = t.item()
    check(
        "7. loss decreases with 200 steps of training",
        t.item() < first,
        f"{first:.4f} → {t.item():.4f}",
    )

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with the cenet-forward exercise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
