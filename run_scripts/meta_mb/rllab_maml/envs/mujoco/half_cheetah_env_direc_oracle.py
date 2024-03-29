import numpy as np

from meta_mb.rllab_maml.core.serializable import Serializable
from meta_mb.rllab_maml.envs.base import Step
from meta_mb.rllab_maml.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.rllab_maml.misc import logger
from meta_mb.rllab_maml.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnvDirecOracle(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        super(HalfCheetahEnvDirecOracle, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goals(self, num_goals):
        # for fwd/bwd env, goal direc is backwards if < 1.0, forwards if > 1.0
        return np.random.uniform(0.0, 2.0, (num_goals, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        if reset_args is None:
            direcs = [-1,1]
            self.goal_direction = np.random.choice(direcs)
        else:
            self.goal_direction = -1.0 if reset_args < 1.0 else 1.0
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def get_current_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])
        return np.r_[obs, np.array([self.goal_direction])]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = self.goal_direction * -1 * self.get_body_comvel("torso")[0]
        #run_cost = 2.*np.abs(self.get_body_comvel("torso")[0] - 0.1)
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        #path["observations"][-1][-3] - path["observations"][0][-3]
        progs = [
            path["observations"][-1][-4] - path["observations"][0][-4]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))
