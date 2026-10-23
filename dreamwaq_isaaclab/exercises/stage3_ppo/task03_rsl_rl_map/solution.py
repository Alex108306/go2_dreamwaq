#!/usr/bin/env python3
"""Stage 3 · task03 — Where in rsl_rl is the PPO I wrote.  `L0 · READ`

There's no code to write. When run, it pulls **the actual source of the installed rsl_rl** and shows it side by side with our `ppo.py`.
It only makes sense to read after writing it yourself, so finish task01·task02 first.

    python solution.py
"""
from __future__ import annotations

import inspect
import sys
import textwrap

C_HEAD = "\033[1;36m"
C_OURS = "\033[1;33m"
C_RSL = "\033[1;32m"
C_OFF = "\033[0m"


def rule(title: str) -> None:
    print(f"\n{C_HEAD}{'─' * 78}\n {title}\n{'─' * 78}{C_OFF}")


def snippet(src: str, start: str, end: str | None = None, max_lines: int = 24) -> str:
    """Cut the source from the start anchor up to just before the end anchor.

    It searches by anchor strings rather than line numbers, so it usually keeps up even if the rsl_rl version differs a bit.
    """
    lines = src.splitlines()
    try:
        i = next(n for n, ln in enumerate(lines) if start in ln)
    except StopIteration:
        return f"  (couldn't find '{start}' in this rsl_rl version — open the source directly)"
    j = len(lines)
    if end is not None:
        for n in range(i + 1, len(lines)):
            if end in lines[n]:
                j = n
                break
    return textwrap.dedent("\n".join(lines[i:min(j, i + max_lines)]))


def show(label: str, color: str, body: str) -> None:
    print(f"\n{color}[{label}]{C_OFF}")
    for ln in body.splitlines():
        print(f"    {ln}")


def main() -> int:
    try:
        import rsl_rl
        from rsl_rl.algorithms import PPO
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        print("Couldn't import rsl_rl. Run with Isaac Sim's bundled python:")
        print("  ~/IsaacLab/_isaac_sim/python.sh solution.py")
        return 1

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    import ppo as ours  # noqa: PLC0415  — our minimal implementation

    print(f"rsl_rl location    : {rsl_rl.__file__}")
    print(f"our implementation : {ours.__file__}")

    ppo_src = inspect.getsource(PPO)

    # ── 1. GAE ────────────────────────────────────────────────────────────────
    rule("1. GAE — compute_gae()  ↔  PPO.compute_returns()")
    show("our ppo.py · compute_gae", C_OURS,
         snippet(inspect.getsource(ours.compute_gae), "last_advantage = torch.zeros_like", "returns = advantages"))
    show("rsl_rl · PPO.compute_returns", C_RSL,
         snippet(ppo_src, "advantage = 0", "# Compute the advantages"))
    print("""
    Same recursion. Only the names differ.

      ours                    rsl_rl
      ----                    ------
      last_advantage          advantage
      not_done                next_is_not_terminal
      next_value              next_values
      returns = adv + values  st.returns[step] = advantage + st.values[step]

    One difference: rsl_rl finishes advantage normalization inside compute_returns
    (we did it in train() after flattening). The formula is the same.""")

    # ── 2. Losses ────────────────────────────────────────────────────────────────
    rule("2. Losses — ppo_losses()  ↔  inside PPO.update()")
    show("our ppo.py · ppo_losses", C_OURS,
         snippet(inspect.getsource(ours.ppo_losses), "ratio = torch.exp", "# ex:end"))
    show("rsl_rl · PPO.update (surrogate + value)", C_RSL,
         snippet(ppo_src, "# Surrogate loss", "# Symmetry loss", max_lines=18))
    print(f"""
    {C_OURS}The sign conventions look different but it's the same formula.{C_OFF}

      ours    : -min( ratio*A , clip(ratio)*A )
      rsl_rl  :  max( -A*ratio , -A*clip(ratio) )

    -min(x, y) == max(-x, -y), so they're fully equivalent. Both "turn the objective to maximize
    into a minimization problem by negating it".

    {C_OURS}Two real differences.{C_OFF}

      1. clipped value loss — if use_clipped_value_loss is True, rsl_rl
         also limits the critic prediction to ±clip_param around the old value.
         Our cfg has use_clipped_value_loss=True, so it actually takes this branch.
      2. entropy sign — rsl_rl subtracts it in the loss (- entropy_coef * entropy),
         we made it negative beforehand and added it (entropy_loss = -entropy.mean()). Same result.""")

    # ── 3. Adaptive learning rate ────────────────────────────────────────────────────────
    rule("3. What we don't have — desired_kl adaptive learning rate")
    show("rsl_rl · PPO.update (adaptive lr)", C_RSL,
         snippet(ppo_src, "if self.desired_kl is not None", "# Surrogate loss", max_lines=20))
    print("""
    Our toy PPO fixed lr at 1e-3. rsl_rl measures the KL between the previous and current policy
    every minibatch and adjusts the learning rate itself.

      KL more than 2x the target  -> moved too much   -> divide lr by 1.5
      KL less than half the target -> moved too little -> multiply lr by 1.5

    This is the topic of the next exercise (task04).""")

    # ── 4. Training loop ────────────────────────────────────────────────────────────
    rule("4. Training loop — train()  ↔  OnPolicyRunner.learn()")
    print(f"""
    {C_OURS}our train(){C_OFF}                         {C_RSL}rsl_rl{C_OFF}
    ------------------------------      ------------------------------
    for it in range(iterations)         OnPolicyRunner.learn()
      1. collect rollouts                 alg.act() / alg.process_env_step()
      2. compute_gae(...)                alg.compute_returns(obs)
      3. epoch x minibatch updates        alg.update()
      history.append(mean_reward)        logging / wandb / tensorboard

    OnPolicyRunner.learn is {len(inspect.getsource(OnPolicyRunner.learn).splitlines())} lines.
    Most of it is logging, checkpointing and multi-GPU handling; the algorithm itself is just the three lines above.

    This project's OnPolicyRunnerWaq overrides exactly this learn() to
    insert CENet right in the middle of the rollout loop (see Stage 4).""")

    # ── 5. Correspondence table ───────────────────────────────────────────────────────────────
    rule("5. At a glance")
    print("""
    ppo.py (ours)          rsl_rl                                   file
    ---------------------  ---------------------------------------  ----------------------
    ActorCritic            MLPModel actor / critic (separate instances) modules/
    log_std parameter      GaussianDistributionCfg(std_type="log")   isaaclab_rl cfg
    compute_gae            PPO.compute_returns                       algorithms/ppo.py
    ppo_losses             inside PPO.update                          algorithms/ppo.py
    train() loop           OnPolicyRunner.learn                      runners/on_policy_runner.py
    buf_* lists            RolloutStorage                            storage/rollout_storage.py
    (none)                 desired_kl adaptive lr                     algorithms/ppo.py
    (none)                 clipped value loss                        algorithms/ppo.py

    [PASS] comparison complete. Continue with task04_ppo_cfg.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
