import numpy as np

from meta_mb.rllab.envs.mujoco.hill.hill_env import HillEnv
from meta_mb.rllab.envs.mujoco.swimmer3d_env import Swimmer3DEnv
from meta_mb.rllab.misc.overrides import overrides
import meta_mb.rllab.envs.mujoco.hill.terrain as terrain
from meta_mb.rllab.spaces import Box

class Swimmer3DHillEnv(HillEnv):

    MODEL_CLASS = Swimmer3DEnv
    
    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(hfield, Box(np.array([-3.0, -1.5]), np.array([0.0, -0.5])))