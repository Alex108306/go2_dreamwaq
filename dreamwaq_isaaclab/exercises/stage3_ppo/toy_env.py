"""Shared Stage 3 toy environment — commanded velocity tracking of a 1D point mass.  [provided code · read only]

A vectorized environment that runs in 1 second without Isaac Sim. Only the essence of locomotion is kept.

    state        v        current velocity
    command      v_cmd    velocity to track (redrawn from U[-1, 1] every episode)
    action       a        acceleration (policy output, clipped to [-1, 1])
    transition   v <- v + (a * action_scale - drag * v) * dt
    reward       exp{-4 (v_cmd - v)^2}          <- same form as Table I of the DreamWaQ paper
    observation  [v, v_cmd, a_prev]             (3-dim)

Correspondence with DreamWaQ:

    | here            | DreamWaQ                          |
    |-----------------|-----------------------------------|
    | v               | base_lin_vel                      |
    | v_cmd           | velocity_commands                 |
    | a               | joint target offset (action * 0.25) |
    | exp{-4 e^2}     | track_lin_vel_xy_exp (std=sqrt(0.25)) |
    | drag            | ground friction / gravity         |

PPO behaves in this environment exactly as it does in locomotion. Understand the algorithm here first,
then go back to the real robot in Stage 4.
"""
from __future__ import annotations

import torch

OBS_DIM = 3
ACT_DIM = 1


class VelocityTrackingEnv:
    """Vectorized 1D velocity-tracking environment. Uses only torch, no gym dependency."""

    def __init__(
        self,
        num_envs: int = 64,
        episode_length: int = 64,
        dt: float = 0.05,
        action_scale: float = 4.0,
        drag: float = 1.0,
        tracking_sigma_sq: float = 0.25,  # exp(-e^2 / 0.25) = exp(-4 e^2)
        device: str = "cpu",
        seed: int | None = 0,
    ):
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.dt = dt
        self.action_scale = action_scale
        self.drag = drag
        self.tracking_sigma_sq = tracking_sigma_sq
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)

        self.vel = torch.zeros(num_envs, device=device)
        self.cmd = torch.zeros(num_envs, device=device)
        self.prev_action = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset()

    # ------------------------------------------------------------------

    def _resample_cmd(self, mask: torch.Tensor) -> None:
        n = int(mask.sum())
        if n:
            self.cmd[mask] = torch.rand(n, device=self.device) * 2.0 - 1.0

    def _obs(self) -> torch.Tensor:
        return torch.stack([self.vel, self.cmd, self.prev_action], dim=1)

    def reset(self) -> torch.Tensor:
        self.vel.zero_()
        self.prev_action.zero_()
        self.step_count.zero_()
        self._resample_cmd(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._obs()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one step. Returns (obs, reward, done)."""
        a = action.squeeze(-1).clamp(-1.0, 1.0)
        self.vel = self.vel + (a * self.action_scale - self.drag * self.vel) * self.dt
        self.prev_action = a

        error = self.cmd - self.vel
        reward = torch.exp(-(error**2) / self.tracking_sigma_sq)

        self.step_count += 1
        done = self.step_count >= self.episode_length
        if done.any():
            self.vel[done] = 0.0
            self.prev_action[done] = 0.0
            self.step_count[done] = 0
            self._resample_cmd(done)

        return self._obs(), reward, done


def perfect_tracking_reward() -> float:
    """Per-step reward with perfect tracking = 1.0. Training results are compared against this value."""
    return 1.0


def standing_still_reward(num_samples: int = 100_000, seed: int = 0) -> float:
    """Expected reward when doing nothing and staying at v=0.

    It's the floor for judging whether training 'really' worked. If the policy can't beat this value,
    it hasn't learned anything. (Same idea as the failure in DreamWaQ where 'standing still is optimal'.)
    """
    g = torch.Generator().manual_seed(seed)
    cmd = torch.rand(num_samples, generator=g) * 2.0 - 1.0
    return torch.exp(-(cmd**2) / 0.25).mean().item()


if __name__ == "__main__":
    env = VelocityTrackingEnv(num_envs=8)
    obs = env.reset()
    print(f"observation shape: {tuple(obs.shape)}  (v, v_cmd, a_prev)")
    print(f"Perfect tracking reward : {perfect_tracking_reward():.4f}")
    print(f"Standing-still reward   : {standing_still_reward():.4f}   <- training fails if it can't beat this")

    total = torch.zeros(env.num_envs)
    for _ in range(env.episode_length):
        _, r, _ = env.step(torch.rand(env.num_envs, 1) * 2 - 1)
        total += r
    print(f"Random policy mean      : {(total / env.episode_length).mean():.4f}")
