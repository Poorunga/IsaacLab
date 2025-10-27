import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import opendoor_rmp_abs_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        persr_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "rgb", "normalize": False}
        )
        head_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("head_cam"), "data_type": "rgb", "normalize": False}
        )
        left_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_cam"), "data_type": "rgb", "normalize": False}
        )
        right_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class AgibotOpenDoorVisuomotorFisheyeEnvCfg(opendoor_rmp_abs_env_cfg.AgibotOpenDoorEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set cameras
        CAMERA_CFG = TiledCameraCfg(
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        self.scene.persr_cam = CAMERA_CFG.copy()
        self.scene.persr_cam.prim_path = "{ENV_REGEX_NS}/Robot/link_pitch_head/persr_cam"
        self.scene.persr_cam.offset = TiledCameraCfg.OffsetCfg(
            pos=(2.0, 0.85, 2.35),
            rot=(0.1637, -0.3696, 0.6854, 0.6055),
            convention="world"
        )
        self.scene.head_cam = CAMERA_CFG.copy()
        self.scene.head_cam.prim_path = "{ENV_REGEX_NS}/Robot/link_pitch_head/head_cam"
        self.scene.head_cam.offset = TiledCameraCfg.OffsetCfg(
            pos=(-0.355, 0.181, 0.0),
            rot=(-0.0779, -0.4443, 0.7329, 0.5092),
            convention="world"
        )

        FISHEYE_CAMERA_CFG = TiledCameraCfg(
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
        )
        self.scene.left_cam = FISHEYE_CAMERA_CFG.copy()
        self.scene.left_cam.prim_path = "{ENV_REGEX_NS}/Robot/left_base_link/left_cam"
        self.scene.left_cam.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.085, 0.097),
            rot=(-0.4815, 0.1081, 0.8177, -0.2962),
            convention="world"
        )
        self.scene.right_cam = FISHEYE_CAMERA_CFG.copy()
        self.scene.right_cam.prim_path = "{ENV_REGEX_NS}/Robot/right_base_link/right_cam"
        self.scene.right_cam.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.084, 0.091),
            rot=(0.7038, -0.1129, -0.6355, 0.2966),
            convention="world"
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["persr_cam", "head_cam", "left_cam", "right_cam"]
