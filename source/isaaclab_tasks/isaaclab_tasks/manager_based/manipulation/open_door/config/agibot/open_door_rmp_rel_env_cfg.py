import os
from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.open_door.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_LEFT_ARM_RMPFLOW_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
MY_ASSETS_PATH = os.getenv("MY_ASSETS_PATH", "missing_assets_dir")

AGIBOT_A2D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agibot/A2D/A2D_physics.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Body joints
            "joint_lift_body": 0.1995,
            "joint_body_pitch": 0.6025,
            # Head joints
            "joint_head_yaw": 0.0,
            "joint_head_pitch": 0.6708,
            # Left arm joints
            "left_arm_joint1": -1.0817,
            "left_arm_joint2": 0.5907,
            "left_arm_joint3": 0.3442,
            "left_arm_joint4": -1.2819,
            "left_arm_joint5": 0.6928,
            "left_arm_joint6": 1.4725,
            "left_arm_joint7": -0.1599,
            # Right arm joints
            "right_arm_joint1": 1.0817,
            "right_arm_joint2": -0.5907,
            "right_arm_joint3": -0.3442,
            "right_arm_joint4": 1.2819,
            "right_arm_joint5": -0.6928,
            "right_arm_joint6": -0.7,
            "right_arm_joint7": 0.0,
            # Left gripper joints
            "left_Right_1_Joint": 0.0,
            "left_hand_joint1": 0.994,
            "left_Right_0_Joint": 0.0,
            "left_Left_0_Joint": 0.0,
            "left_Right_Support_Joint": 0.994,
            "left_Left_Support_Joint": 0.994,
            "left_Right_RevoluteJoint": 0.0,
            "left_Left_RevoluteJoint": 0.0,
            # Right gripper joints
            "right_Right_1_Joint": 0.0,
            "right_hand_joint1": 0.994,
            "right_Right_0_Joint": 0.0,
            "right_Left_0_Joint": 0.0,
            "right_Right_Support_Joint": 0.994,
            "right_Left_Support_Joint": 0.994,
            "right_Right_RevoluteJoint": 0.0,
            "right_Left_RevoluteJoint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # init pos of the articulation for teleop
    ),
    actuators={
        # Body lift and torso actuators
        "body": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift_body", "joint_body_pitch"],
            effort_limit_sim=10000.0,
            velocity_limit_sim=2.61,
            stiffness=10000000.0,
            damping=200.0,
        ),
        # Head actuators
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_yaw", "joint_head_pitch"],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Left arm actuator
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-7]"],
            effort_limit_sim={
                "left_arm_joint1": 2000.0,
                "left_arm_joint[2-7]": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={"left_arm_joint1": 10000000.0, "left_arm_joint[2-7]": 20000.0},
            damping={"left_arm_joint1": 0.0, "left_arm_joint[2-7]": 0.0},
        ),
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-7]"],
            effort_limit_sim={
                "right_arm_joint1": 2000.0,
                "right_arm_joint[2-7]": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={"right_arm_joint1": 10000000.0, "right_arm_joint[2-7]": 20000.0},
            damping={"right_arm_joint1": 0.0, "right_arm_joint[2-7]": 0.0},
        ),
        # "left_Right_2_Joint" is excluded from Articulation.
        # "left_hand_joint1" is the driver joint, and "left_Right_1_Joint" is the mimic joint.
        # "left_.*_Support_Joint" driver joint can be set optionally, to disable the driver, set stiffness and damping to 0.0 below
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_hand_joint1", "left_.*_Support_Joint"],
            effort_limit_sim={"left_hand_joint1": 10.0, "left_.*_Support_Joint": 1.0},
            velocity_limit_sim=2.0,
            stiffness={"left_hand_joint1": 20.0, "left_.*_Support_Joint": 2.0},
            damping={"left_hand_joint1": 0.10, "left_.*_Support_Joint": 0.01},
        ),
        # set PD to zero for passive joints in close-loop gripper
        "left_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["left_.*_(0|1)_Joint", "left_.*_RevoluteJoint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # "right_Right_2_Joint" is excluded from Articulation.
        # "right_hand_joint1" is the driver joint, and "right_Right_1_Joint" is the mimic joint.
        # "right_.*_Support_Joint" driver joint can be set optionally, to disable the driver, set stiffness and damping to 0.0 below
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_joint1", "right_.*_Support_Joint"],
            effort_limit_sim={"right_hand_joint1": 100.0, "right_.*_Support_Joint": 100.0},
            velocity_limit_sim=10.0,
            stiffness={"right_hand_joint1": 20.0, "right_.*_Support_Joint": 2.0},
            damping={"right_hand_joint1": 0.10, "right_.*_Support_Joint": 0.01},
        ),
        # set PD to zero for passive joints in close-loop gripper
        "right_gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_.*_(0|1)_Joint", "right_.*_RevoluteJoint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


##
# Scene definition
##
@configclass
class SceneCfg(InteractiveSceneCfg):
    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{MY_ASSETS_PATH}/Objects/Door/door_9288/door_9288.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.3, 0.0, 0.9),
            rot=(1.0, 0.0, 0.0, -0.05),
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["joint_0"],
                effort_limit_sim=100.0,
                damping=80.0,
                stiffness=400.0,
            ),
            "handle": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit_sim=100.0,
                damping=10,
                stiffness=0.0,
            ),
        },
    )

    # Frame definitions for the door.
    handle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Door/base/base",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HandleFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Door/base/link_2",
                name="handle_frame",
                offset=OffsetCfg(
                    pos=[0.05, 0.0, 0.08], #  -y,-z,-x
                    rot=[0.7071, 0.0, -0.7071, 0.0], # end effector aligned
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        door_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["joint_0"])},
        )
        door_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["joint_0"])},
        )
        rel_ee_handle_distance = ObsTerm(func=mdp.rel_ee_handle_distance)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##
@configclass
class OpenDoorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the open door environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=128, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625


class RmpFlowAgibotOpenDoorEnvCfg(OpenDoorEnvCfg):
    """Configuration for Agibot Open Door Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Agibot as robot
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="gripper_center",
            controller=AGIBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[0.7071, 0.0, -0.7071, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
            open_command_expr={"left_hand_joint1": 0.994, "left_.*_Support_Joint": 0.994},
            close_command_expr={"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0},
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["left_hand_joint1", "left_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=[0.7071, 0.0, -0.7071, 0.0],
                    ),
                ),
            ],
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0
