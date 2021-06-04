import numpy as np
from meta_policy_search.envs.base import MetaEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_policy_search.utils import logger
import gym


class MetalheadEnvV1RandDirec(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self, goal_direction=None):
        self.goal_direction = goal_direction if goal_direction else 1.0
        MujocoEnv.__init__(self, 'metalhead_v1.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return np.random.choice((-1.0, 1.0), (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_direction = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_direction

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        torsoanglebefore = self.sim.data.qpos[2]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        torso_pitch_angle_after = self.sim.data.qpos[2]
        torso_roll_angle_after = self.sim.data.qpos[3]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = self.goal_direction * (xposafter - xposbefore)/self.dt
        reward_torso_pitch = -0.02 * np.square(torso_pitch_angle_after)
        reward_torso_roll = -0.02 * np.square(torso_roll_angle_after)
        reward = reward_ctrl + reward_run + reward_torso_pitch + reward_torso_roll
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        progs = [np.mean(path["env_infos"]["reward_run"]) for path in paths]
        ctrl_cost = [-np.mean(path["env_infos"]["reward_ctrl"]) for path in paths]

        logger.logkv(prefix + 'AverageForwardReturn', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardReturn', np.max(progs))
        logger.logkv(prefix + 'MinForwardReturn', np.min(progs))
        logger.logkv(prefix + 'StdForwardReturn', np.std(progs))

        logger.logkv(prefix + 'AverageCtrlCost', np.mean(ctrl_cost))
