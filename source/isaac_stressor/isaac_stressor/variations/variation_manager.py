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
import robomimic
from isaac_stressor.tasks.manager_based.cabinet import mdp
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
import math
import yaml
import importlib
import math
from typing import Dict, Any
import copy

class VariationManager:
    '''
        Takes config as input and can raturn randomized config, manage all other variations
    '''

    def __init__(self):

        self.origin = None

        self.randomize_cabinet_texture = EventTerm(
            func=mdp.randomize_visual_texture_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cabinet"),
                "texture_paths": [
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
                ],
                "event_name": "object_panel_visual_texture",
                "texture_rotation": (math.pi / 2, math.pi / 2),
            },
        )

        self.cabinet_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cabinet", body_names=["drawer_top"]),
                "mass_distribution_params": (0.1, 4.1),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        self.cabinet_physics_material = EventTerm(
           func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cabinet"),
                "static_friction_range": (0.5, 1.5),
                "dynamic_friction_range": (0.5, 1.5),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )

    def generate_config(self, original_env_cfg, variation):
        count = -1
        for field_name, event_term in self.__dict__.items():
            count += 1
            if count != variation:
                continue
            env_cfg = copy.deepcopy(original_env_cfg)
        
            # Set attribute on env_cfg.events
            if event_term is not None:
                setattr(env_cfg.events, field_name, event_term)
            return env_cfg
     