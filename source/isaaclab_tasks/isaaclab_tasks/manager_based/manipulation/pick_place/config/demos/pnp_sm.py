import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pick and place state machine for aloha pick orange environments."
)
parser.add_argument("--task", type=str, help="Name of the task.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
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

import omni.log
from state_machine import PickAndPlaceSm

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors import FrameTransformer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.pick_place.config.aloha_x5a.place_orange_env_cfg import (
    PlaceOrangeEnvCfg,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # parse configuration
    env_cfg: PlaceOrangeEnvCfg = parse_env_cfg(
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
    desired_orientation = torch.zeros(
        (env.unwrapped.num_envs, 4), device=env.unwrapped.device
    )
    desired_orientation[:] = torch.tensor(
        [0.96363, 0.0, 0.26724, 0.0], device=desired_orientation.device
    )
    # create state machine
    pnp_sm = PickAndPlaceSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=0.015,
    )

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            dones = env.step(actions)[-2]
            # reset state machine
            if dones.any():
                pnp_sm.reset_idx()
                env.reset()

            # observations
            # -- end-effector frame
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_tf.data.target_pos_w[..., 0, :].clone()
                - env.unwrapped.scene.env_origins
            )
            tcp_rest_orientation = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            # -- orange frame
            object_data: RigidObjectData = env.unwrapped.scene["orange"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[
                ..., :3
            ]
            # -- plate frame
            desired_position2 = (
                env.unwrapped.scene["plate"].data.root_pos_w
                - env.unwrapped.scene.env_origins
            ).clone()
            desired_position2[:, 2] += 0.15

            # advance state machine
            actions = pnp_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                torch.cat([desired_position2, desired_orientation], dim=-1),
            )

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
