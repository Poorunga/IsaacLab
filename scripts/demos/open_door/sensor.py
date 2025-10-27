import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Open door state machine for open door environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Learn-Open-Door-Agibot-Right-Arm-RmpFlow-Sensors-v0",
    help="Name of the task."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

# Omniverse logger
import omni.log

from isaaclab.sensors import FrameTransformer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.open_door.config.agibot.opendoor_rmp_abs_env_cfg import (
    OpenDoorEnvCfg
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from state_machine import OpenDoorSm


def main():
    # parse configuration
    env_cfg: OpenDoorEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0
    # create state machine
    open_sm = OpenDoorSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            # reset state machine
            if dones.any():
                open_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                env.reset()

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

            # print information from the sensors
            scene = env.unwrapped.scene
            # Contact sensors
            # print("-------------------------------")
            # print(scene["contact_forces_RL"])
            # print("Received force matrix of: ", scene["contact_forces_RL"].data.force_matrix_w)
            # print("Received contact force of: ", scene["contact_forces_RL"].data.net_forces_w)
            # print("-------------------------------")
            # print(scene["contact_forces_RR"])
            # print("Received force matrix of: ", scene["contact_forces_RR"].data.force_matrix_w)
            # print("Received contact force of: ", scene["contact_forces_RR"].data.net_forces_w)
            # print("-------------------------------")
            # print(scene["contact_forces_Base"])
            # print("Received force matrix of: ", scene["contact_forces_Base"].data.force_matrix_w)
            # print("Received contact force of: ", scene["contact_forces_Base"].data.net_forces_w)

            # IMU sensors
            print("-------------------------------")
            print(scene["imu_RL"])
            print("Received linear velocity: ", scene["imu_RL"].data.lin_vel_b)
            print("Received angular velocity: ", scene["imu_RL"].data.ang_vel_b)
            print("Received linear acceleration: ", scene["imu_RL"].data.lin_acc_b)
            print("Received angular acceleration: ", scene["imu_RL"].data.ang_acc_b)
            print("-------------------------------")
            print(scene["imu_RR"])
            print("Received linear velocity: ", scene["imu_RR"].data.lin_vel_b)
            print("Received angular velocity: ", scene["imu_RR"].data.ang_vel_b)
            print("Received linear acceleration: ", scene["imu_RR"].data.lin_acc_b)
            print("Received angular acceleration: ", scene["imu_RR"].data.ang_acc_b)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
