from meta_mb.rllab.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.rllab_maml.core.serializable import Serializable
import numpy as np

from meta_mb.rllab_maml.envs.base import Step
from meta_mb.rllab_maml.misc.overrides import overrides
from meta_mb.rllab_maml.misc import logger


class AntEnvRand(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, goal= None, *args, **kwargs):
        self._goal_vel = goal
        #if goal is not None:
        #    self._goal_vel *= 100
        super(AntEnvRand, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def sample_goals(self, num_goals):
        return np.random.uniform(0.0, 3.0, (num_goals, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_vel = reset_args
        if goal_vel is not None:
            self._goal_vel = goal_vel
        elif self._goal_vel is None:
            #self._goal_vel = np.random.uniform(0.1, 0.8)
            self._goal_vel = np.random.uniform(0.0, 3.0)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = -np.abs(comvel[0] - self._goal_vel) + 1.0 # make it happy, not suicidal
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

