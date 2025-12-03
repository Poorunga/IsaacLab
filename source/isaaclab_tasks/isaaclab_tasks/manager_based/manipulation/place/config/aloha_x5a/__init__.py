# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Aloha_x5a Left Arm: place upright mug task, with RmpFlow
##
gym.register(
    id="Aloha-X5A-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_orange_env_cfg:AlohaX5aPlaceOrangeEnvCfg",
    },
    disable_env_checker=True,
)
