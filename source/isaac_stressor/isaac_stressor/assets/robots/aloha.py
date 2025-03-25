# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Aloha robots.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os
import isaac_stressor

BASE_DIR = os.path.dirname(os.path.abspath(isaac_stressor.__file__))
ROBOT_USD = "aloloha_v03_cameras2.usd"

##
# Configuration
##

# Available strings: 
# LINKS = ['fl_castor_wheel', 'fr_castor_wheel', 'left_wheel', 'right_wheel', 'rl_castor_wheel', 'rr_castor_wheel', 'fl_joint1', 'fr_joint1', 'lr_joint1', 'rr_joint1', 'fl_wheel', 'fr_wheel', 'rl_wheel', 'rr_wheel', 'fl_joint2', 'fr_joint2', 'lr_joint2', 'rr_joint2', 'fl_joint3', 'fr_joint3', 'lr_joint3', 'rr_joint3', 'fl_joint4', 'fr_joint4', 'lr_joint4', 'rr_joint4', 'fl_joint5', 'fr_joint5', 'lr_joint5', 'rr_joint5', 'fl_joint6', 'fr_joint6', 'lr_joint6', 'rr_joint6', 'fl_joint7', 'fl_joint8', 'fr_joint7', 'fr_joint8', 'lr_joint7', 'lr_joint8', 'rr_joint7', 'rr_joint8']

ALOHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{BASE_DIR}/assets/robots/usd/{ROBOT_USD}",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=12, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*": 0.0},
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["[\w]+_wheel"],
            effort_limit=40,
            velocity_limit=10000.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "fl_viper_arm": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint[1-6]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "fl_gripper": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint[7-8]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr_viper_arm": ImplicitActuatorCfg(
            joint_names_expr=["fr_joint[1-6]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr_gripper": ImplicitActuatorCfg(
            joint_names_expr=["fr_joint[7-8]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "lr_viper_arm": ImplicitActuatorCfg(
            joint_names_expr=["lr_joint[1-6]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "lr_gripper": ImplicitActuatorCfg(
            joint_names_expr=["lr_joint[7-8]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "rr_viper_arm": ImplicitActuatorCfg(
            joint_names_expr=["rr_joint[1-6]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "rr_gripper": ImplicitActuatorCfg(
            joint_names_expr=["rr_joint[7-8]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)
