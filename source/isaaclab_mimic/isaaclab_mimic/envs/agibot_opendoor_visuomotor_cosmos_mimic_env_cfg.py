from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.open_door.config.agibot.opendoor_rmp_abs_visuomotor_cosmos_env_cfg import (
    AgibotOpenDoorVisuomotorCosmosEnvCfg
)


@configclass
class RmpFlowAgibotOpenDoorVisuomotorCosmosMimicEnvCfg(AgibotOpenDoorVisuomotorCosmosEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Agibot Open Door env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        self.datagen_config.name = "demo_src_place_opendoor_visuomotor_cosmos_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the stack task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="door",
                # End of final subtask does not need to be detected
                subtask_term_signal="grasp",
                # No time offsets for the final subtask
                subtask_term_offset_range=(2, 10),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                # selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.01,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=15,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="door",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                # selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.01,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=15,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["agibot"] = subtask_configs
