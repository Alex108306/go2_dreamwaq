"""Go2 DreamWaQ environment configurations and Gymnasium registration.

Lecture scope: **PPO only**, three observation variants — Base / Oracle / Waq.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DreamWaQ-Manager-Go2-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_base_cfg:Go2BaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Base-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_base_cfg:Go2BaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Oracle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_oracle_cfg:Go2OracleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Oracle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_oracle_cfg:Go2OracleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Waq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_waq_cfg:Go2WaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Waq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_waq_cfg:Go2WaqEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)

##
# The actual comparison axis of the Stage 1 course — 3 variants on top of the official env recipe, with only the actor observation altered.
#
# With the original DreamWaQ recipe (±1 m/s x 3-axis push every 1 s, DreamWaQ reward set), every algorithm
# plateaus at underlying 0.16~0.26 and fails to walk. Having confirmed that the cause is the env design, not CENet,
# the whole matrix was unified onto the official recipe (report.qmd §1).
#
#   Base   actor 45 (blind)                        lower bound
#   Oracle actor 48 (+ true lin_vel)               upper bound on velocity information
#   Waq    actor 45 -> 64 (+ CENet est_vel+context) DreamWaQ proposal
#
# All three critics are privileged (full official observation + height_scan for rough).
##

# Flat Oracle used to point at the stock UnitreeGo2FlatEnvCfg; it now needs a local subclass
# so that the DreamWaQ Table-I rewards apply to this arm too (PAPER.md §2).
for _terrain, _envcls in (("Flat", "OracleOfficialFlatEnvCfg"), ("Rough", "OracleOfficialRoughEnvCfg")):
    for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"DreamWaQ-OracleDwq-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OracleDwqHparams{_terrain}PPORunnerCfg"
                ),
            },
        )

for _terrain, _envcls in (("Flat", "WaqOfficialFlatEnvCfg"), ("Rough", "WaqOfficialRoughEnvCfg")):
    for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
        # Base: blind actor 45 + privileged critic, same hparams as Waq (Waq - Base = pure CENet contribution)
        gym.register(
            id=f"DreamWaQ-BaseDwq-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BaseDwqHparams{_terrain}PPORunnerCfg"
                ),
            },
        )
        # Waq: same env, CENet runner
        gym.register(
            id=f"DreamWaQ-Waq-Official-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqOfficial{_terrain}PPORunnerCfg"
                ),
            },
        )


##
# The oracle the paper actually talks about — a policy that sees the terrain height map (rough only).
#
# Paper Fig.3 caption: "The oracle policy has access to the height map measurement of the robot's
# surroundings", §III-B: "despite walking without exteroception, DreamWaQ performs almost as well
# as the oracle policy that has direct access to the surrounding terrain's height map."
#
# OracleDwq above is a *velocity* oracle, an axis the paper makes no claim about. This arm is needed
# to test the paper's claim "does the blind Waq approach a policy that sees the terrain?".
# Flat has no terrain information, so only rough is registered.
##
for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"DreamWaQ-TerrainOracle-Rough-PPO{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.go2_waq_official_cfg:TerrainOracleOfficialRoughEnvCfg{_envsfx}"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TerrainOracleRoughPPORunnerCfg"
            ),
        },
    )
