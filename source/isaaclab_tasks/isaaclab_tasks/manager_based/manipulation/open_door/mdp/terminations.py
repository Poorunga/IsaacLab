from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def door_opened(env: ManagerBasedRLEnv, door_cfg: SceneEntityCfg = SceneEntityCfg("door"), threshold: float = 0.42) -> torch.Tensor:
    """
    Termination condition: door considered opened when its joint_0 absolute position exceeds `threshold`.

    Returns:
        torch.Tensor of shape (num_envs,) with dtype=torch.bool on the same device as the door joint tensor.
    """
    door: Articulation = env.scene[door_cfg.name]
    # door joint position (shape: [num_envs])
    door_joint_ids, _ = door.find_joints(["joint_0"])
    door_pos = door.data.joint_pos[:, door_joint_ids[0]]

    # compute boolean mask of opened doors and ensure correct dtype/device
    opened_mask = (door_pos.abs() > float(threshold))
    return torch.as_tensor(opened_mask, dtype=torch.bool, device=door_pos.device)
