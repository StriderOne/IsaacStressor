# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Scene definition
##

@configclass
class RackSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    # robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    # ee_frame: FrameTransformerCfg = MISSING

    # Cameras
    # wrist_cam: CameraCfg = MISSING
    # table_cam: CameraCfg = MISSING

    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/homa/Downloads/rack/rack.usd"),
    )

    can1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Can1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.09257, 0, 0.92], rot=[0.725086, 0.0, 0.0, -0.6886584]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/homa/IsaacStressor/assets/can1/can1.usd"),
    )

    can2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Can2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.086, 0, 0.92], rot=[0.6575462, 0.0, 0.0, -0.7534142]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/homa/IsaacStressor/assets/can2/can2.usd"),
    )
    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/homa/IsaacStressor/assets/background/background.usd"),
    )
    robot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -1.0, 0.8], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"/home/homa/IsaacStressor/source/isaac_stressor/isaac_stressor/assets/robots/usd/arx5_description/usd/arx5_.usd"),
    )

    # Frame definitions for the cabinet.
    # cabinet_frame = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
    #     debug_vis=False,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
    #             name="drawer_handle_top",
    #             offset=OffsetCfg(
    #                 pos=(0.305, 0.0, 0.01),
    #                 rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
    #             ),
    #         ),
    #     ],
    # )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    pass


@configclass
class EventCfg:
    """Configuration for events."""

    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##


@configclass
class RackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: RackSceneCfg = RackSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
