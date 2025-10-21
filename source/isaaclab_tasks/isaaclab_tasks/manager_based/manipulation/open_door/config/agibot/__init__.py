import gymnasium as gym

##
# Register Gym environments.
##

##
# Agibot Right Arm: open task, with RmpFlow
##
gym.register(
    id="Isaac-Open-Door-Agibot-Right-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.open_door_rmp_rel_env_cfg:RmpFlowAgibotOpenDoorEnvCfg",
    },
    disable_env_checker=True,
)
