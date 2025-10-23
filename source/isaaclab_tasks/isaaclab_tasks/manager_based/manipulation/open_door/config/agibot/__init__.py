import gymnasium as gym

##
# Register Gym environments.
##

##
# Agibot Right Arm: open task, with RmpFlow
##
gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rmp_abs_env_cfg:RmpFlowAgibotOpenDoorEnvCfg",
    },
    disable_env_checker=True,
)
