from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg, CameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import isaaclab.sim as sim_utils

import isaaclab_tasks.manager_based.manipulation.open_door.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.open_door.open_door_env_cfg import OpenDoorEnvCfg
from isaaclab_tasks.manager_based.manipulation.open_door.config.agibot.agibot_a2d_cfg import AGIBOT_A2D_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_RIGHT_ARM_RMPFLOW_CFG  # isort: skip


class RmpFlowAgibotOpenDoorEnvCfg(OpenDoorEnvCfg):
    """Configuration for Agibot Open Door Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Agibot as robot
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_center",
            controller=AGIBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=False,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
            open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
            close_command_expr={"right_hand_joint1": 0.0, "right_.*_Support_Joint": 0.0},
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["right_hand_joint1", "right_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2

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
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=[1.0, 0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # Setup robot head, left arm and right arm cameras
        CAMERA_CFG = TiledCameraCfg(
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        self.scene.head_camera = CAMERA_CFG.copy()
        self.scene.head_camera.prim_path = "{ENV_REGEX_NS}/Robot/link_pitch_head/head_camera"
        self.scene.head_camera.offset = TiledCameraCfg.OffsetCfg(
            pos=(-0.355, 0.181, 0.0),
            rot=(-0.0779, -0.4443, 0.7329, 0.5092),
            convention="world"
        )
        self.scene.left_camera = CAMERA_CFG.copy()
        self.scene.left_camera.prim_path = "{ENV_REGEX_NS}/Robot/left_base_link/left_camera"
        self.scene.left_camera.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.085, 0.097),
            rot=(-0.4815, 0.1081, 0.8177, -0.2962),
            convention="world"
        )
        self.scene.right_camera = CAMERA_CFG.copy()
        self.scene.right_camera.prim_path = "{ENV_REGEX_NS}/Robot/right_base_link/right_camera"
        self.scene.right_camera.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.084, 0.091),
            rot=(0.7038, -0.1129, -0.6355, 0.2966),
            convention="world"
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0
