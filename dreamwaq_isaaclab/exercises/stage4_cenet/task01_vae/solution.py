#!/usr/bin/env python3
"""Stage 4 · task01 — Plain VAE (toy data).

CENet is a VAE. First build just the VAE — no terrain, no robot, no Isaac Sim.
The encode / reparameterize / decode / ELBO written here become CENet as-is in the next exercises (cenet-forward,
cenet-loss).

    python solution.py                # train briefly on toy data and see whether the loss decreases
    python solution.py --beta 10.0    # apply strong regularization and see posterior collapse with your own eyes
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

INPUT_DIM, HIDDEN_DIM, LATENT_DIM = 8, 32, 2


def make_toy_batch(batch: int, input_dim: int = INPUT_DIM, seed: int | None = None) -> torch.Tensor:
    """Toy observations: a 2-dim latent structure spread out into input_dim dimensions.

    The true degrees of freedom are only 2, but it's observed in 8 dims — a situation made for the VAE to uncover.
    In DreamWaQ terms, this corresponds to "the true state of the terrain is summarized by a few numbers, but the robot only sees a 225-dim history".
    """
    gen = torch.Generator().manual_seed(0)  # the spreading matrix must always be the same
    mixer = torch.randn(LATENT_DIM, input_dim, generator=gen)
    if seed is not None:
        torch.manual_seed(seed)
    z_true = torch.randn(batch, LATENT_DIM)
    return z_true @ mixer + 0.05 * torch.randn(batch, input_dim)


# ex:begin id=vae-core level=3 stage=stage4_cenet task=Implement the whole VAE for toy data — encode / reparameterize / decode / ELBO
#   hint: the encoder maps the input to 2*latent_dim; split the first half as mu and the second half as logvar
#   hint: reparameterize is z = mu + eps * std, std = exp(0.5 * logvar), eps ~ N(0, I). Gradient must flow to mu
#   hint: KL(q||p) = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)) — then 'average' over latent dims and also over the batch. recon is an MSE mean, so this matches the axes
#   hint: the kl returned by loss already has self.beta multiplied in (total = recon + kl). Same convention as CENet
#   hint: the exact required API is written in README.md in this folder
class ToyVAE(nn.Module):
    """Input -> latent distribution -> sample -> reconstruction. Same skeleton as CENet."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # The encoder outputs mu and logvar at once — that's why the output is 2 * latent_dim.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.split(self.latent_dim, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Writing the sample as mu + eps*std leaves randomness only in eps, so gradient flows to mu/logvar.
        # Drawing directly from z ~ N(mu, sigma^2) isn't differentiable — this is the reparameterization trick.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, x):
        x_hat, mu, logvar, _ = self.forward(x)

        recon_loss = nn.MSELoss()(x_hat, x)

        # recon is nn.MSELoss() — a mean per input dimension. The KL must be reduced over the same axis
        # (mean per latent dim) so the ratio of the two terms isn't swayed by the number of dims. Same convention as CENet.
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = klds.mean(-1).mean() * self.beta

        return recon_loss + kl_loss, recon_loss, kl_loss
# ex:end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight. Compare 1.0 and 10.0")
    args = ap.parse_args()

    torch.manual_seed(0)
    net = ToyVAE(beta=args.beta)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    print(f"beta = {args.beta}")

    data = make_toy_batch(512, seed=1)
    first = last = None
    for step in range(300):
        total, recon, kl = net.loss(data)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step == 0:
            first = total.item()
        if step % 60 == 0:
            print(f"  step {step:3d}  total={total.item():8.4f}  recon={recon.item():8.4f}  kl={kl.item():7.4f}")
        last = total.item()

    print(f"\nfirst loss {first:.4f} → last loss {last:.4f}")
    assert last < first, "training didn't reduce the loss"
    print("[PASS] the toy VAE trains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
