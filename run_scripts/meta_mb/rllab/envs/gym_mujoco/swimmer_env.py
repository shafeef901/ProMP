from meta_mb.rllab.envs.base import Step
from meta_mb.rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from meta_mb.rllab.core.serializable import Serializable
from meta_mb.rllab.misc import logger
from meta_mb.rllab.misc import autoargs


class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, target_velocity=None, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.target_velocity = target_velocity
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat
            ])

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.model.data.qpos[0]
        self.forward_dynamics(action)
        xposafter = self.model.data.qpos[0]
        velocity = (xposafter - xposbefore) / self.dt
        if self.target_velocity:
            reward_fwd = np.abs(velocity - self.target_velocity)
        else:
            reward_fwd = velocity
        reward_ctrl = - ctrl_cost_coeff * np.square(action).sum()
        reward = reward_fwd + reward_ctrl
        ob = self.get_current_obs()
        done = False
        self.time_step += 1
        if self.max_path_length and self.time_step > self.max_path_length:
            done = True
        return ob, float(reward), done, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    @overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)
