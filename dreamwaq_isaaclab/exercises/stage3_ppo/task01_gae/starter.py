"""Minimal PPO implementation — a standalone script that runs without Isaac Sim.  [Stage 3 solution]

Before reading rsl_rl's PPO, write the skeleton of the algorithm yourself. Where the pieces built here
correspond to in `rsl_rl/algorithms/ppo.py` is compared in `task03_rsl_rl_map/`.

    python ppo.py            # train on the toy environment and check that the reward rises
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# toy_env.py lives in stage3_ppo/. This file can be run from either there (solution) or taskNN/ (starter),
# so look in both places.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent):
    if (_cand / "toy_env.py").exists():
        sys.path.insert(0, str(_cand))
        break

from toy_env import ACT_DIM, OBS_DIM, VelocityTrackingEnv, standing_still_reward  # noqa: E402


class ActorCritic(nn.Module):
    """Same structure as DreamWaQ: actor and critic don't share parameters.

    The real project uses 512-256-128, but this is a toy problem so it's reduced to 64-64.
    The action distribution is a Gaussian with a state-independent learnable log_std —
    also the same as rsl_rl and DreamWaQ.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        # init_noise_std = 1.0 (rsl_rl default and the DreamWaQ value)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: (T, N)  reward at each step
        values:  (T, N)  state value V(s_t) as seen by the critic
        dones:   (T, N)  1.0 if the episode ended at that step
        last_value: (N,) V(s_T) of the last observation — bootstraps the truncated rollout
        gamma: discount factor
        lam: GAE lambda. 0 gives 1-step TD, 1 gives Monte Carlo

    Returns:
        (returns, advantages), both (T, N)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    # ── TODO(ppo-gae) ─ level L2 ─────────────────────────────
    # Compute GAE advantages by accumulating from back to front
    #   hint: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)  <- TD error
    #   hint: A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}        <- recursion
    #   hint: sweep the time axis in reverse (reversed) so A_{t+1} is already known
    #   hint: at t = T-1, V(s_{t+1}) is the last_value passed in as an argument
    #   hint: at done steps there's no continuation to the next state, so cut it with (1 - done)
    #   hint: returns = advantages + values (the critic's regression target)
    # For the pass criteria, see README.md in this exercise folder.
    raise NotImplementedError("TODO(ppo-gae)")
    # ─────────────────────────────────────────────────────────────────────

    return returns, advantages


def ppo_losses(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_param: float = 0.2,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO's three loss terms and their total.

    Args:
        log_prob:     (B,) log probability the current policy gives that action
        old_log_prob: (B,) log probability the policy gave when collecting the rollout (constant)
        advantages:   (B,) normalized advantages
        value:        (B,) current critic prediction
        returns:      (B,) regression target built with GAE
        entropy:      (B,) entropy of the current policy

    Returns:
        (total, surrogate_loss, value_loss, entropy_loss)
    """
    ratio = torch.exp(log_prob - old_log_prob)
    surrogate = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages,
    ).mean()
    value_loss = (value - returns).pow(2).mean()
    entropy_loss = -entropy.mean()
    total = surrogate + value_loss_coef * value_loss + entropy_coef * entropy_loss

    return total, surrogate, value_loss, entropy_loss


def train(
    iterations: int = 200,
    num_envs: int = 256,
    steps_per_env: int = 32,
    epochs: int = 5,
    minibatches: int = 4,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[ActorCritic, list[float]]:
    """PPO training loop. Same skeleton as rsl_rl's OnPolicyRunner.learn()."""
    env = VelocityTrackingEnv(num_envs=num_envs, device=device)
    model = ActorCritic().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    obs = env.reset()
    history: list[float] = []

    for it in range(iterations):
        # --- 1. Collect rollouts (on-policy) ---
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
        for _ in range(steps_per_env):
            action, log_prob, value = model.act(obs)
            next_obs, reward, done = env.step(action)
            buf_obs.append(obs); buf_act.append(action); buf_logp.append(log_prob)
            buf_rew.append(reward); buf_done.append(done.float()); buf_val.append(value)
            obs = next_obs

        with torch.no_grad():
            last_value = model.value(obs)

        b_obs = torch.stack(buf_obs); b_act = torch.stack(buf_act)
        b_logp = torch.stack(buf_logp); b_rew = torch.stack(buf_rew)
        b_done = torch.stack(buf_done); b_val = torch.stack(buf_val)

        # --- 2. Advantages ---
        returns, advantages = compute_gae(b_rew, b_val, b_done, last_value)

        # Normalize advantages after flattening (same as rsl_rl)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        f_obs, f_act = flat(b_obs), flat(b_act)
        f_logp, f_ret = flat(b_logp), flat(returns)
        f_adv = flat(advantages)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        # --- 3. Minibatch updates over several epochs ---
        batch = f_obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(batch, device=device)
            for mb in perm.chunk(minibatches):
                dist = model.distribution(f_obs[mb])
                log_prob = dist.log_prob(f_act[mb]).sum(-1)
                total, *_ = ppo_losses(
                    log_prob, f_logp[mb], f_adv[mb],
                    model.value(f_obs[mb]), f_ret[mb], dist.entropy().sum(-1),
                )
                optim.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

        mean_reward = b_rew.mean().item()
        history.append(mean_reward)
        if verbose and (it % 20 == 0 or it == iterations - 1):
            print(f"  iter {it:4d} | mean reward {mean_reward:.4f} | std {model.log_std.exp().mean():.3f}")

    return model, history


if __name__ == "__main__":
    baseline = standing_still_reward()
    print(f"Standing-still reward (floor) : {baseline:.4f}\n")
    _, history = train()
    final = sum(history[-10:]) / 10
    print(f"\nFinal mean reward : {final:.4f}  (floor {baseline:.4f}, perfect tracking 1.0)")
    print("Training succeeded" if final > baseline + 0.2 else "Training failed — didn't beat the floor")
