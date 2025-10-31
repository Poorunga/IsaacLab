from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerData
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_handle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the door handle."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    handle_tf_data: FrameTransformerData = env.scene["handle_frame"].data

    return handle_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def handle_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Check if an object is grasped by the specified robot.
    Support both surface gripper and parallel gripper.
    If contact_grasp sensor is found, check if the contact force is greater than force_threshold.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.body_link_pos_w[:, object_cfg.body_ids[0]]
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if "contact_grasp" in env.scene.keys() and env.scene["contact_grasp"] is not None:
        contact_force_grasp = env.scene["contact_grasp"].data.net_forces_w  # shape:(N, 2, 3) for two fingers
        contact_force_norm = torch.linalg.vector_norm(
            contact_force_grasp, dim=2
        )  # shape:(N, 2) - force magnitude per finger
        both_fingers_force_ok = torch.all(
            contact_force_norm > force_threshold, dim=1
        )  # both fingers must exceed threshold
        grasped = torch.logical_and(pose_diff < diff_threshold, both_fingers_force_ok)
    elif (
        f"contact_grasp_{object_cfg.name}" in env.scene.keys()
        and env.scene[f"contact_grasp_{object_cfg.name}"] is not None
    ):
        contact_force_object = env.scene[
            f"contact_grasp_{object_cfg.name}"
        ].data.net_forces_w  # shape:(N, 2, 3) for two fingers
        contact_force_norm = torch.linalg.vector_norm(
            contact_force_object, dim=2
        )  # shape:(N, 2) - force magnitude per finger
        both_fingers_force_ok = torch.all(
            contact_force_norm > force_threshold, dim=1
        )  # both fingers must exceed threshold
        grasped = torch.logical_and(pose_diff < diff_threshold, both_fingers_force_ok)
    else:
        grasped = (pose_diff < diff_threshold).clone().detach()

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            grasped = torch.logical_and(
                grasped,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                > env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return grasped
