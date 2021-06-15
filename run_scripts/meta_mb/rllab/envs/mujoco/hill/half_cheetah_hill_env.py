import numpy as np

from meta_mb.rllab.envs.mujoco.hill.hill_env import HillEnv
from meta_mb.rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from meta_mb.rllab.misc.overrides import overrides
import meta_mb.rllab.envs.mujoco.hill.terrain as terrain
from meta_mb.rllab.spaces import Box

class HalfCheetahHillEnv(HillEnv):

    MODEL_CLASS = HalfCheetahEnv
    
    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(hfield, Box(np.array([-3.0, -1.5]), np.array([0.0, -0.5])))