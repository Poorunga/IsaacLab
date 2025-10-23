import os
from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.open_door.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
MY_ASSETS_PATH = os.getenv("MY_ASSETS_PATH", "missing_assets_dir")


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
            mass_props=MassPropertiesCfg(mass=10.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.2, -0.5, 0.95),
            rot=(1.0, 0.0, 0.0, -0.05),
            joint_pos={"joint_0": 0.0, "joint_2": 0.3},
        ),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["joint_0"],
                effort_limit_sim=0.0,
                damping=5.0,
                stiffness=0.0,
            ),
            "handle": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit_sim=87.0,
                damping=80.0,
                stiffness=400.0,
            ),
        },
    )

    # Frame definitions for the door.
    handle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Door/base",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HandleFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Door/link_2",
                name="handle_frame",
                offset=OffsetCfg(
                    pos=(0.05, 0.0, 0.08),  # -y,-z,-x
                    rot=[0.0, 0.0, -1.0, 0.0]
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
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
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (5.23, 5.25),
            "dynamic_friction_range": (5.23, 5.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    door_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names="link_2"),
            "static_friction_range": (5.23, 5.25),
            "dynamic_friction_range": (5.45, 5.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


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
    scene: SceneCfg = SceneCfg(num_envs=128, env_spacing=2.5)
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
        self.viewer.eye = (-2.0, -2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
