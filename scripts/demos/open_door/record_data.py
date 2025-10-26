import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demonstrations for A2D open door environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Visuomotor-Cosmos-v0",
    help="Name of the task."
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=20,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Third-party imports
import gymnasium as gym
import os
import time
import torch

# Omniverse logger
import omni.log

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from state_machine import OpenDoorSm


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
        )
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    # env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(env: gym.Env, success_step_count: int) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    return success_step_count


def run_simulation_loop(env: gym.Env, success_term: object | None) -> int:
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False

    # Reset before starting
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    open_sm = OpenDoorSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    with torch.inference_mode():
        while simulation_app.is_running():
            dones = env.step(actions)[-2]
            if dones.any():
                success_step_count = handle_reset(env, success_step_count)
                should_reset_recording_instance = False
                open_sm.reset_idx()

            # observations
            # -- end-effector frame
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            # -- handle frame
            handle_frame_tf: FrameTransformer = env.unwrapped.scene["handle_frame"]
            handle_position = handle_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            handle_orientation = handle_frame_tf.data.target_quat_w[..., 0, :].clone()

            # advance state machine
            actions = open_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([handle_position, handle_orientation], dim=-1),
            )

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    env.step(actions)
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count)
                should_reset_recording_instance = False
                open_sm.reset_idx()

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

    return current_recorded_demo_count


def main() -> None:
    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, success_term)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
