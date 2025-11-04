import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.stack.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.open_door.mdp as open_door_mdp

from . import opendoor_rmp_abs_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # actions = ObsTerm(func=mdp.last_action)
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        # eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # gripper_pos = ObsTerm(func=mdp.gripper_pos)
        # rel_ee_handle_distance = ObsTerm(func=open_door_mdp.rel_ee_handle_distance)

        persr_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "rgb", "normalize": False}
        )
        persr_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "semantic_segmentation", "normalize": True}
        )
        persr_cam_normals = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "normals", "normalize": True}
        )
        persr_cam_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "depth", "normalize": True}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # grasp = ObsTerm(
        #     func=open_door_mdp.handle_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("door", body_names="link_2"),
        #         "diff_threshold": 0.1,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class AgibotOpenDoorVisuomotorCosmosEnvCfg(opendoor_rmp_abs_env_cfg.AgibotOpenDoorEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # import carb
        # from isaacsim.core.utils.carb import set_carb_setting

        # carb_setting = carb.settings.get_settings()
        # set_carb_setting(carb_setting, "/rtx/domeLight/upperLowerStrategy", 4)

        SEMANTIC_MAPPING = {
            "class:door": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (204, 110, 248, 255),
            "class:UNLABELLED": (150, 150, 150, 255),
            "class:BACKGROUND": (200, 200, 200, 255),
        }

        # Set cameras
        self.scene.persr_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/persr_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals", "depth"],
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.2764, 1.76641, 1.67427),
                rot=(0.88373, 0.02408, 0.06949, -0.46219),
                convention="world"
            )
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["persr_cam"]
