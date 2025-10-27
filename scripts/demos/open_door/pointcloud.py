import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test open door environment.")
parser.add_argument("--num_envs", type=int, default=11, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import transform_points, unproject_depth
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

from isaaclab_tasks.manager_based.manipulation.open_door.config.agibot.opendoor_rmp_abs_visuomotor_pointcloud_env_cfg import (
    AgibotOpenDoorVisuomotorPointcloudEnvCfg
)


def main():
    """Main function."""
    # parse the arguments
    env_cfg = AgibotOpenDoorVisuomotorPointcloudEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Set marker
    cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
    cfg.markers["hit"].radius = 0.002
    pc_markers = VisualizationMarkers(cfg)

    env.reset()
    camera = env.scene["persr_cam"]

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            dones = env.step(joint_efforts)[-2]

            camera_index = 0
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                depth=camera.data.output["distance_to_image_plane"][camera_index],
                position=camera.data.pos_w[camera_index],
                orientation=camera.data.quat_w_ros[camera_index],
                device=args_cli.device,
            )
            pc_markers.visualize(translations=pointcloud)

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
