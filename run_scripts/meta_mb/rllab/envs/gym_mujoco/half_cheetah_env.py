import numpy as np

from meta_mb.rllab.core.serializable import Serializable
from meta_mb.rllab.envs.base import Step
from meta_mb.rllab.envs.gym_mujoco.mujoco_env import MujocoEnv
from meta_mb.rllab.misc import logger
from meta_mb.rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, target_velocity=None, **kwargs):
        super(HalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.target_velocity = target_velocity
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        xposbefore = self.model.data.qpos[0]
        self.forward_dynamics(action)
        xposafter = self.model.data.qpos[0]
        ob = self.get_current_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        velocity = (xposafter - xposbefore) / self.dt
        if self.target_velocity:
            reward_run = np.abs(velocity - self.target_velocity)
        else:
            reward_run = velocity
        reward = reward_ctrl + reward_run
        done = False

        self.time_step += 1
        if self.max_path_length and self.time_step > self.max_path_length:
            done = True

        # clip reward in case mujoco sim goes crazy
        reward = np.minimum(np.maximum(-1000, reward), 1000)

        return ob, float(reward), done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
