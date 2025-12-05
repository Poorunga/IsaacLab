# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Aloha_x5a Right Arm: place and place orange
##
gym.register(
    id="AlohaX5a-Pnp-Orange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_orange_env_cfg:AlohaX5aPlaceOrangeEnvCfg",
    },
    disable_env_checker=True,
)
