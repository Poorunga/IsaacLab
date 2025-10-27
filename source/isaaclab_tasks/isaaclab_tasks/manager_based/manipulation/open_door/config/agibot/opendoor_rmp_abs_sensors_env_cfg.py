from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG

from . import opendoor_rmp_abs_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

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
class AgibotOpenDoorSensorsEnvCfg(opendoor_rmp_abs_env_cfg.AgibotOpenDoorEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # setup contact sensors
        self.scene.contact_forces_RL = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_Left_Pad_Link",
            update_period=0.1,
            history_length=6,
            debug_vis=True,
        )
        self.scene.contact_forces_RR = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_Right_Pad_Link",
            update_period=0.1,
            history_length=6,
            debug_vis=True,
        )
        self.scene.contact_forces_Base = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.1,
            history_length=6,
            debug_vis=True,
        )

        # setup imu sensor
        self.scene.imu_RL = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/right_Left_Pad_Link", debug_vis=True)
        self.scene.imu_RR = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/right_Right_Pad_Link", debug_vis=True)
