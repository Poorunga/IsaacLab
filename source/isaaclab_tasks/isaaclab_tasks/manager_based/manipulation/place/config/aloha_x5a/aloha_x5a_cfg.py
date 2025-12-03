import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MY_ASSETS_PATH = os.getenv("MY_ASSETS_PATH", "missing_assets_dir")

ALOHA_X5A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MY_ASSETS_PATH}/Robot/aloha_x5a_usd/aloha_x5a.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "right_joint7": 0.04,
            "right_joint8": 0.04,
        },
        pos=(0.2, -0.5, 0.0),  # init pos of the articulation for teleop
        rot=(0.7071, 0.0, 0.0, 0.7071),
    ),
    actuators={
        # Right arm actuator
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[1-6]"],
            effort_limit_sim={
                "right_joint1": 2000.0,
                "right_joint[2-6]": 1000.0,
            },
            velocity_limit_sim=1.57,
            stiffness={"right_joint1": 10000000.0, "right_joint[2-6]": 20000.0},
            damping={"right_joint1": 0.0, "right_joint[2-7]": 0.0},
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["right_joint[7-8]"],
            effort_limit_sim={"right_joint[7-8]": 100.0},
            velocity_limit_sim=10.0,
            stiffness={"right_joint[7-8]": 20.0},
            damping={"right_joint[7-8]": 0.10},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
