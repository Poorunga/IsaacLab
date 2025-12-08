# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.config.aloha_x5a.aloha_x5a_cfg import (
    ALOHA_X5A_CFG,
)
from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


MY_ASSETS_PATH = os.getenv("MY_ASSETS_PATH", "missing_assets_dir")


@configclass
class KitchenRoomSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/KitchenRoom",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0, 0), rot=(0.70711, 0.0, 0.0, -0.70711)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{MY_ASSETS_PATH}/Environments/Kitchen_Room/KitchenRoom_RSS.usd",
        ),
        collision_group=-1,
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.70, 0.75),
            pos_y=(-0.5, -0.4),
            pos_z=(1.0, 1.1),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("orange"),
            },
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("plate"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("orange")},
    )

    success = DoneTerm(
        func=place_mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("orange"),
            "object_b_cfg": SceneEntityCfg("plate"),
            "xy_threshold": 0.10,
            "height_diff": 0.0,
            "height_threshold": 0.03,
        },
    )


@configclass
class PlaceOrangeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: KitchenRoomSceneCfg = KitchenRoomSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # set viewer to see the whole scene
        self.viewer.eye = (-0.6, -1.7, 2.4)
        self.viewer.lookat = (2.9, 2.3, -1.0)


"""
Env to Replay Sim2Lab Demonstrations with JointSpaceAction
"""


class AlohaX5aPlaceOrangeEnvCfg(PlaceOrangeEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set aloha_x5a as robot
        self.scene.robot = ALOHA_X5A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (aloha_x5a)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_joint[1-8]"],
            body_name="right_gripper_center",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0)
            ),
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_joint[7-8]"],
            open_command_expr={
                "right_joint[7-8]": 0.4,
            },
            close_command_expr={
                "right_joint[7-8]": 0.0,
            },
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["right_joint7", "right_joint8"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.3

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "right_gripper_center"

        self.scene.orange = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Orange",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.73, -0.46, 0.95), rot=(1.0, 0, 0, 0.0)
            ),
            spawn=UsdFileCfg(
                usd_path=f"{MY_ASSETS_PATH}/Environments/Kitchen_Room/half_orange.usd",
            ),
        )

        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.72, -0.15, 0.95)),
            spawn=UsdFileCfg(
                usd_path=f"{MY_ASSETS_PATH}/Environments/Kitchen_Room/plate.usd",
            ),
        )

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm/right_gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
                ),
            ],
        )

        self.scene.right_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm/right_link6/right_cam",
            update_period=0.1,
            height=400,
            width=400,
            data_types=["rgb"],
            spawn=sim_utils.FisheyeCameraCfg(
                projection_type="fisheyePolynomial",
                focal_length=0.1,
                f_stop=10.0,
                clipping_range=(0.1, 1000.0),
                horizontal_aperture=10.0,
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.03947, 0.0, 0.11296),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0
