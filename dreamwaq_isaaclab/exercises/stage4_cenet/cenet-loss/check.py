#!/usr/bin/env python3
"""Fast check — cenet-loss.

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
        raise SystemExit(f"File not found: {path}\n  generate it with exercises/tools/make_exercise.py --id cenet-loss")
    spec = importlib.util.spec_from_file_location(f"_cenet_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CENet


def make_net(CENet, beta: float = 1.0, latent: int = LATENT, num_obs: int = NUM_OBS):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):  # CENet.__init__ prints its MLP structure
        return CENet(
            num_learning_epochs=1,
            num_mini_batches=1,
            input_dim=OBS_HISTORY,
            latent_dim1=NUM_VEL + latent * 2,
            latent_dim2=NUM_VEL + latent,
            output_dim=num_obs,
            beta=beta,
            device="cpu",
        )


def leaf(t: torch.Tensor) -> torch.Tensor:
    """A differentiable stand-in: `update()` calls backward(), so fakes need a grad path."""
    return t.clone().detach().requires_grad_(True)


class Batch:
    """Mimics one minibatch consumed by update()."""

    def __init__(self, obs_hist, true_vel, true_onext):
        self._data = [(obs_hist, true_vel, true_onext)]

    def __call__(self, *_args, **_kwargs):
        return iter(self._data)


class FakeStorage:
    def __init__(self, batch):
        self.mini_batch_generator = batch
        self.cleared = False

    def clear(self):
        self.cleared = True


def run_update(net, obs_hist, true_vel, true_onext):
    """Run CENet.update() on one synthetic batch and get (total, vel, recon, kl)."""
    net.storage = FakeStorage(Batch(obs_hist, true_vel, true_onext))
    net.train_mode()
    return net.update()


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    CENet = load_cenet(path)

    torch.manual_seed(0)
    obs_hist = torch.randn(BATCH, OBS_HISTORY)
    true_vel = torch.randn(BATCH, NUM_VEL)
    true_onext = torch.randn(BATCH, NUM_OBS)

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # --- 1. Is it reduced to a scalar -------------------------------------------------
    net = make_net(CENet)
    try:
        total, vel, recon, kl = run_update(net, obs_hist, true_vel, true_onext)
    except NotImplementedError:
        print("  TODO(cenet-loss) is still empty. Fill in starter/cenet.py.")
        return 1
    except RuntimeError as exc:
        # Common mistake: leaving `kl_loss = klds.mean(1)` gives shape (B,), so total_loss.backward()
        # inside update() blows up. Left as-is, the checker dies without printing a single FAIL line.
        check("1. the four losses are returned as scalars", False)
        print()
        if "scalar outputs" in str(exc):
            print("\033[1;31mThe loss isn't a scalar — you may have missed the batch mean (`.mean(0)`).\033[0m")
            print("  `klds.mean(1)` is still (B,). After the latent-dim mean, one more batch mean is needed.")
            print("  `total_loss.backward()` only works on scalars, so it blows up here.")
        else:
            print("\033[1;31mupdate() died with a RuntimeError.\033[0m")
        print(f"  torch message: {exc}")
        return 1
    check("1. the four losses are returned as scalars", all(isinstance(x, float) for x in (total, vel, recon, kl)))

    # --- 2. Perfect prediction gives zero MSE terms -------------------------------------------
    # To force est == true, intercept forward and return the ground truth as-is.
    net = make_net(CENet)
    mu0 = torch.zeros(BATCH, LATENT)
    net.forward = lambda _oh: (leaf(true_onext), leaf(true_vel), leaf(mu0), leaf(mu0), leaf(mu0))
    _, vel_exact, recon_exact, kl_exact = run_update(net, obs_hist, true_vel, true_onext)
    check("2. vel_loss == 0 if est == true", approx(vel_exact, 0.0), f"vel_loss={vel_exact:.6f}")
    check("2. recon_loss == 0 if est == true", approx(recon_exact, 0.0), f"recon_loss={recon_exact:.6f}")

    # --- 3. KL == 0 if q == p ------------------------------------------------
    check("3. kl_loss == 0 when mu=0, logvar=0", approx(kl_exact, 0.0), f"kl_loss={kl_exact:.6f}")

    # --- 4·5. Look at the properties of the KL with a deterministic fake forward -------------------------
    # The real forward draws randn_like inside reparameterize, so results differ per call.
    # To look at the loss function alone, forward must be replaced with fixed values.
    def losses_with(mu_val: float, logvar_val: float, beta: float = 1.0, latent: int = LATENT):
        n = make_net(CENet, beta=beta, latent=latent)
        mu = torch.full((BATCH, latent), mu_val)
        lv = torch.full((BATCH, latent), logvar_val)
        est_onext, est_vel = true_onext + 0.5, true_vel + 0.3  # deliberately wrong predictions
        n.forward = lambda _oh: (leaf(est_onext), leaf(est_vel), leaf(mu), leaf(lv), leaf(mu))
        return run_update(n, obs_hist, true_vel, true_onext)

    kl_base = losses_with(0.0, 0.0)[3]
    kl_mu = losses_with(1.0, 0.0)[3]
    kl_lv = losses_with(0.0, 1.5)[3]
    check("4. kl increases as mu moves away from 0", kl_mu > kl_base + 1e-6, f"{kl_mu:.6f} <= {kl_base:.6f}")
    check("4. kl increases as logvar moves away from 0", kl_lv > kl_base + 1e-6, f"{kl_lv:.6f} <= {kl_base:.6f}")

    # --- 4b. Reduction axis — the core of this exercise -------------------------------------------
    # Looking only at the inequalities, both sum and mean pass. What actually breaks training is "recon and KL
    # being reduced over different axes". nn.MSELoss() is a per-dimension mean, so the KL must also be a per-dimension
    # mean so the ratio of the two terms isn't swayed by the number of dimensions.
    #
    # How to tell: keep the per-dim value fixed and only increase the number of dims. With a mean the value stays the same;
    # with a sum it grows in proportion to the number of dims.
    def recon_with_dims(num_obs: int) -> float:
        n = make_net(CENet, num_obs=num_obs)
        t_onext = torch.zeros(BATCH, num_obs)
        e_onext = t_onext + 1.0  # error of 1 per element → recon == 1.0 if it's a per-dim mean
        mu = torch.zeros(BATCH, LATENT)
        n.forward = lambda _oh: (leaf(e_onext), leaf(true_vel), leaf(mu), leaf(mu), leaf(mu))
        return run_update(n, obs_hist, true_vel, t_onext)[2]

    kl_l16 = losses_with(1.0, 0.0, latent=LATENT)[3]
    kl_l32 = losses_with(1.0, 0.0, latent=LATENT * 2)[3]
    recon_d45 = recon_with_dims(NUM_OBS)
    recon_d90 = recon_with_dims(NUM_OBS * 2)

    kl_is_mean = approx(kl_l16, kl_l32, 1e-4)
    recon_is_mean = approx(recon_d45, recon_d90, 1e-4)
    check("4. kl is 'averaged' over latent dims (dim=1) (mu=1, logvar=0 → 0.5)",
          approx(kl_l16, 0.5, 1e-4),
          f"expected 0.5 but got {kl_l16:.4f} — a sum would give {LATENT * 0.5}")
    check("4. kl stays the same when latent dims go 16 → 32 (a sum would double)",
          kl_is_mean, f"latent 16 → {kl_l16:.4f}, latent 32 → {kl_l32:.4f}")
    check("4. recon and kl have the same reduction axis (both per-dim means)",
          kl_is_mean and recon_is_mean,
          f"recon 45dim {recon_d45:.4f} / 90dim {recon_d90:.4f}, "
          f"kl 16dim {kl_l16:.4f} / 32dim {kl_l32:.4f}")

    # --- 5. beta applies only to KL, linearly --------------------------------------------
    b1 = losses_with(1.0, 0.5, beta=1.0)
    b2 = losses_with(1.0, 0.5, beta=2.0)
    check("5. beta 2x → kl exactly 2x", approx(b2[3], 2 * b1[3], 1e-4),
          f"beta=1 → {b1[3]:.6f}, beta=2 → {b2[3]:.6f}")
    check("5. changing beta leaves vel/recon unchanged",
          approx(b1[1], b2[1], 1e-4) and approx(b1[2], b2[2], 1e-4),
          f"vel {b1[1]:.6f}/{b2[1]:.6f}, recon {b1[2]:.6f}/{b2[2]:.6f}")

    # --- 6. Is the total the sum of the three terms (checked with the real forward result run at the top) --------
    check("6. total == vel + recon + kl", approx(total, vel + recon + kl, 1e-4),
          f"total={total:.6f} vs sum={vel + recon + kl:.6f}")
    # Measuring only at β=1 leaves a hole. An implementation that multiplies β only on the `total_loss` side and not on the
    # returned `kl_loss` (or vice versa) has equal values at β=1 and just passes. Check 5 doesn't catch it either —
    # check 5 only looks at whether `kl_loss` is proportional to β. So measure once more with β≠1.
    check("6. total == vel + recon + kl even at beta≠1 (β wasn't multiplied separately in two places)",
          approx(b2[0], b2[1] + b2[2] + b2[3], 1e-4),
          f"at beta=2 total={b2[0]:.6f} vs vel+recon+kl={b2[1] + b2[2] + b2[3]:.6f}")

    # --- 7. Does gradient flow to both encoder and decoder --------------------------
    net = make_net(CENet)
    run_update(net, obs_hist, true_vel, true_onext)
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoder.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.decoder.parameters())
    check("7. gradient flows to the encoder", enc_grad)
    check("7. gradient flows to the decoder", dec_grad)

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m Continue with the runner-augment exercise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
