# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]

def rel_ee_drawer_yaw(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data
    _, _, cabinet_yaw = math_utils.euler_xyz_from_quat(ee_tf_data.target_quat_w[..., 0, :])
    _, _, ee_yaw = math_utils.euler_xyz_from_quat(cabinet_tf_data.target_quat_w[..., 0, :])
    return cabinet_yaw - ee_yaw


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat

def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot_data: ArticulationData = env.scene[robot_cfg.name].data
    finger_joint_1 = robot_data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot_data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def gripper_effort(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot_data: ArticulationData = env.scene[robot_cfg.name].data
    finger_joint_1 = robot_data.applied_torque[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot_data.applied_torque[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)

def gripper_vel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot_data: ArticulationData = env.scene[robot_cfg.name].data
    finger_joint_1 = robot_data.joint_vel[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot_data.joint_vel[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)