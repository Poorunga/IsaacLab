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
        persr_cam_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("persr_cam"), "data_type": "distance_to_image_plane", "normalize": True}
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
class AgibotOpenDoorVisuomotorPointcloudEnvCfg(opendoor_rmp_abs_env_cfg.AgibotOpenDoorEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set cameras
        CAMERA_CFG = TiledCameraCfg(
            update_period=0.0,
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

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["persr_cam"]
