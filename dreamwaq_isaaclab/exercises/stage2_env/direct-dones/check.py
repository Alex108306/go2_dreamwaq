#!/usr/bin/env python3
"""Fast check — direct-dones.  (no Isaac Sim needed, ~2 s)

Calls `DreamWaQEnv._get_dones` directly with a fake `self`.

    python check.py                 # starter/dreamwaq_env.py
    python check.py --solution      # finished version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fake_isaaclab import load_module  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_DIR = REPO_ROOT / "dreamwaq_direct/source/dreamwaq_direct/dreamwaq_direct/tasks/locomotion"
SOLUTION = SRC_DIR / "dreamwaq_env.py"
STARTER = HERE / "starter" / "dreamwaq_env.py"

N = 4                 # envs
HIST = 3              # contact sensor history_length
BODIES = 5            # say, 1 base + 4 feet
BASE_IDS = [0]        # _termination_contact_ids — base only
MAX_EP = 1000
THRESHOLD = 1.0


class Cfg:
    termination_contact_force = THRESHOLD


class FakeSelf:
    """Minimal object with only what _get_dones touches."""

    def __init__(self, forces: torch.Tensor, ep_len: torch.Tensor):
        data = type("D", (), {"net_forces_w_history": type("W", (), {"torch": forces})()})()
        self._contact_sensor = type("CS", (), {"data": data})()
        self._termination_contact_ids = BASE_IDS
        self.cfg = Cfg()
        self.episode_length_buf = ep_len
        self.max_episode_length = MAX_EP
        self._last_episode_dones = torch.zeros(N, 2, dtype=torch.bool)


def make(force_at=None, ep_len=None):
    """force_at: {(env, hist, body): force magnitude} — applies force only where specified."""
    forces = torch.zeros(N, HIST, BODIES, 3)
    for (e, h, b), mag in (force_at or {}).items():
        forces[e, h, b, 2] = mag  # force in the z direction
    ep = torch.zeros(N, dtype=torch.long) if ep_len is None else ep_len
    return FakeSelf(forces, ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", action="store_true", help="check the finished version")
    args = ap.parse_args()

    path = SOLUTION if args.solution else STARTER
    print(f"Checking: {path.relative_to(REPO_ROOT)}\n")
    mod = load_module(path, extra_paths=(str(SRC_DIR),))
    get_dones = mod.DreamWaQEnv._get_dones

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "\033[1;32mPASS\033[0m" if cond else "\033[1;31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    # --- 1. If nothing happens, both are False ------------------------------------------
    try:
        contact, timeout = get_dones(make())
    except NotImplementedError:
        print("  TODO(direct-dones) is still empty. Fill in starter/dreamwaq_env.py.")
        return 1

    check("1. returns two tensors of shape (num_envs,)",
          tuple(contact.shape) == (N,) and tuple(timeout.shape) == (N,),
          f"contact {tuple(contact.shape)}, timeout {tuple(timeout.shape)}")
    check("1. with no force and no timeout, both are False",
          not contact.any() and not timeout.any())

    # --- 2. Timeout ---------------------------------------------------------------
    ep = torch.tensor([0, MAX_EP - 2, MAX_EP - 1, MAX_EP])
    _, timeout = get_dones(make(ep_len=ep))
    check("2. only envs with episode_length >= max-1 time out",
          timeout.tolist() == [False, False, True, True], f"{timeout.tolist()}")

    # --- 3. Base contact ---------------------------------------------------------------
    contact, _ = get_dones(make({(1, 0, 0): 5.0}))
    check("3. only envs with a force above the threshold on the base terminate",
          contact.tolist() == [False, True, False, False], f"{contact.tolist()}")

    # --- 4. Threshold boundary --------------------------------------------------------------
    # The original uses > 1.0. Exactly 1.0 is not a termination.
    contact, _ = get_dones(make({(0, 0, 0): THRESHOLD}))
    check("4. exactly at the threshold is not a termination (> not >=)",
          not contact[0], f"{contact.tolist()}")
    contact, _ = get_dones(make({(0, 0, 0): THRESHOLD + 0.01}))
    check("4. exceeding the threshold even slightly terminates", bool(contact[0]))

    # --- 5. history is judged by its max -------------------------------------------------
    # Case where it hit hard in only one of the 3 frames. Using the mean misses it.
    contact, _ = get_dones(make({(2, 1, 0): 9.0}))
    check("5. hitting in just one of the 3 history frames terminates (max, not mean)",
          bool(contact[2]), f"{contact.tolist()}")

    # --- 6. Bodies other than the base are ignored --------------------------------------------------
    # Large forces on the feet (bodies 1–4) are normal walking. Terminating on them means it can't walk.
    contact, _ = get_dones(make({(0, 0, 1): 500.0, (1, 2, 4): 500.0}))
    check("6. foot contact is not a termination (filtered with _termination_contact_ids)",
          not contact.any(), f"{contact.tolist()}")

    # --- 7. Force is measured as vector magnitude ---------------------------------------------------------
    # It must be the norm, not per component. (0.8, 0.8, 0.8) has each component below 1 but a norm of 1.39.
    forces = torch.zeros(N, HIST, BODIES, 3)
    forces[3, 0, 0] = torch.tensor([0.8, 0.8, 0.8])
    contact, _ = get_dones(FakeSelf(forces, torch.zeros(N, dtype=torch.long)))
    check("7. force is measured as the vector norm, not per component",
          bool(contact[3]), f"norm={float(torch.linalg.norm(forces[3, 0, 0])):.3f} but got {contact.tolist()}")

    # --- 8. The two are independent -------------------------------------------------------------
    ep = torch.tensor([MAX_EP, 0, 0, 0])
    contact, timeout = get_dones(make({(1, 0, 0): 5.0}, ep_len=ep))
    check("8. timeout and base_contact don't affect each other",
          timeout.tolist() == [True, False, False, False]
          and contact.tolist() == [False, True, False, False],
          f"timeout {timeout.tolist()}, contact {contact.tolist()}")

    print()
    if failures:
        print(f"\033[1;31m{len(failures)} failed\033[0m: {', '.join(failures)}")
        return 1
    print("\033[1;32mAll passed.\033[0m You've finished all the Stage 2 exercises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
