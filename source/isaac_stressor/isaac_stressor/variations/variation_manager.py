from dataclasses import MISSING

from isaaclab.utils import configclass
import math
from typing import Dict, Any, List
from copy import deepcopy
from collections.abc import Callable, Sequence
from isaaclab.managers import EventTermCfg as EventTerm

@configclass
class VariationCfg:
    name: str = MISSING 
    func: Callable = MISSING
    mode: str = MISSING
    params: dict = MISSING
    min_step_count_between_reset: int = 720
    enable: bool = True

# @configclass
# class VariationManagerEntitiesCfg:
#     '''
#         Config for objects to be configurated by variation manager
#     '''
#     from isaaclab.managers import SceneEntityCfg
#     manipulation_object: SceneEntityCfg = MISSING
#     receiver_object: SceneEntityCfg = MISSING
#     table_object: SceneEntityCfg = MISSING
#     light_object: SceneEntityCfg = MISSING
#     background_object: SceneEntityCfg = MISSING
#     camera_object: List[SceneEntityCfg] | SceneEntityCfg = MISSING

@configclass
class VariationManagerCfg:
    '''
        Config for the variation manager
    '''

    # entities: VariationManagerEntitiesCfg = MISSING

    seed: int | None = None

    variations: object = MISSING


class VariationManager:
    '''
        Takes config as input and can return randomized config, manage all other variations
    '''

    def __init__(self, cfg: VariationManagerCfg):
        self.cfg = deepcopy(cfg)
        self.variations: Sequence[VariationCfg] = []
        self.init_variations()

    def generate_env_config(self, original_env_cfg, variation):
        env_cfg = deepcopy(original_env_cfg)
        if variation < len(self.variations):
            event_term = self.create_event_term_from_variation(self.variations[variation])
            setattr(env_cfg.events, self.variations[variation].name, event_term)
        return env_cfg
    
    def create_event_term_from_variation(self, variation: VariationCfg) -> EventTerm: 
        return EventTerm(func=variation.func, mode=variation.mode, params=variation.params)
            
    def init_variations(self):
        for key, variation in vars(self.cfg.variations).items():
            if variation.enable:
                self.variations.append(variation)