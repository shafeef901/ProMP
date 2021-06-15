import numpy as np

from meta_mb.rllab.core.serializable import Serializable
from meta_mb.rllab.envs.base import Step
from meta_mb.rllab.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.rllab.misc import logger
from meta_mb.rllab.misc.overrides import overrides
import pickle

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

num_tasks = 1000
# goal_vels = np.random.uniform(-2.0, 2.0, (num_tasks, ))
# import pickle
# pickle.dump(goal_vels, open('all_goal_vels.pkl','wb'))
# import IPython
# IPython.embed()

class HalfCheetahEnvRand(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, choice=None, *args, **kwargs):
        self.choice = choice
        self.goal_vels = pickle.load(open('all_goal_vels.pkl','rb'))
        super(HalfCheetahEnvRand, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goals(self, num_goals):
        return np.array(range(num_goals))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        choice = reset_args
        if choice is not None:
            self.choice = choice
        elif self.choice is None:
            #self._goal_vel = np.random.uniform(0.1, 0.8)
            self.choice = np.random.randint(num_tasks)
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
        return obs

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
        run_cost = 1.*np.abs(self.get_body_comvel("torso")[0] - self.goal_vels[0])
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

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
