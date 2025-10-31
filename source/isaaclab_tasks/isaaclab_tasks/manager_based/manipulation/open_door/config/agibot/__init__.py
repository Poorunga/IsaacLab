import gymnasium as gym
import os

from . import agents

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
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_env_cfg:AgibotOpenDoorEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Sensors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_sensors_env_cfg:AgibotOpenDoorSensorsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_visuomotor_env_cfg:AgibotOpenDoorVisuomotorEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-Cosmos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_visuomotor_cosmos_env_cfg:AgibotOpenDoorVisuomotorCosmosEnvCfg",
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-Fisheye-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_visuomotor_fisheye_env_cfg:AgibotOpenDoorVisuomotorFisheyeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-Pointcloud-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_visuomotor_pointcloud_env_cfg:AgibotOpenDoorVisuomotorPointcloudEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-Annotators-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.opendoor_rmp_abs_visuomotor_annotators_env_cfg:AgibotOpenDoorVisuomotorAnnotatorsEnvCfg",
    },
    disable_env_checker=True,
)
