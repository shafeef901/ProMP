from collections import OrderedDict
import numpy as np
from gym.spaces import Dict
from meta_mb.rllab.spaces import Box
from meta_mb.rllab.misc import logger

from meta_mb.sandbox.ours.envs.sawyer.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from meta_mb.sandbox.ours.envs.sawyer.core.multitask_env import MultitaskEnv
from meta_mb.sandbox.ours.envs.sawyer.base import SawyerXYZEnv


class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance_obj_success',
            indicator_threshold=0.06,

            obj_init_pos=(0, 0.6, 0.02),

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6, 0.02),
            goal_low=None,
            goal_high=None,

            hide_goal_markers=False,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )
        self._observation_space_dict = Dict([
            ('observation', self.hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),
            ('state_observation', self.hand_and_obj_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
        ])
        self.observation_space = Box(np.concatenate([self.hand_and_obj_space.low, self.hand_and_obj_space.low]),
                                     np.concatenate([self.hand_and_obj_space.high, self.hand_and_obj_space.high]))

    @property
    def model_name(self):
        return 'sawyer_pick_and_place.xml'

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        obs_dict = self._get_obs_dict()
        reward = self.compute_reward(action, obs_dict)
        obs = self._convert_obs_dict_to_obs(obs_dict)
        info = self._get_info()
        done = False
        return obs, reward, done, info

    def _get_obs_dict(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

    def _convert_obs_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])

    def _get_obs(self):
        return self._convert_obs_dict_to_obs(self._get_obs_dict())

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        obj_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obj_distance = np.linalg.norm(obj_goal - self.get_obj_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_obj_pos()
        )
        return dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            hand_and_obj_distance=hand_distance+obj_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (
                -1000
            )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = pos.copy()
        qvel[8:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def put_obj_in_hand(self):
        new_obj_pos = self.data.get_site_xpos('endeffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation(-1)
        self.do_simulation(1)
        self._set_obj_xyz(new_obj_pos)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))
        self._set_obj_xyz(state_goal[3:])
        self.sim.forward()

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size, p_obj_in_hand=0.5):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(batch_size, self.hand_and_obj_space.low.size),
            )
        num_objs_in_hand = int(batch_size * p_obj_in_hand)

        # Put object in hand
        goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
        goals[:num_objs_in_hand, 4] -= 0.01

        # Put object one the table (not floating)
        goals[num_objs_in_hand:, 5] = self.obj_init_pos[2]
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'obj_distance':
            r = -obj_distances
        elif self.reward_type == 'obj_success':
            r = -(obj_distances < self.indicator_threshold).astype(float)

        elif self.reward_type == 'hand_and_obj_distance_weighted':
            r = -hand_distances - 10 * obj_distances
        elif self.reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif self.reward_type == 'hand_and_obj_distance_obj_success':
            r = -hand_and_obj_distances - (obj_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_obj_success':
            r = -(
                hand_and_obj_distances < self.indicator_threshold
            ).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances < self.indicator_threshold).astype(float)

        elif self.reward_type == 'sophisticated1':
            r = touch_distances - 10 * obj_distances

        elif self.reward_type == 'sophisticated2':
            a = touch_distances + 10 * obj_distances
            obj_success = (obj_distances < self.indicator_threshold).astype(float)
            r = - (1-obj_success) * a - obj_success * hand_distances

        elif self.reward_type == 'sophisticated3':
            obj_success = (obj_distances < self.indicator_threshold).astype(float)
            touch_success = (touch_distances < self.indicator_threshold).astype(float)

            r = - (1-touch_success)*(1-obj_success)*touch_distances \
                - (touch_success)*(1-obj_success)*obj_distances \
                - obj_success * hand_distances

        elif self.reward_type == 'sophisticated2':
            a = touch_distances + 10 * obj_distances
            obj_success = (obj_distances < self.indicator_threshold).astype(float)
            r = - (1 - obj_success) * a - obj_success * hand_distances

        elif self.reward_type == 'lift_up':
            obj_height = obj_pos[-1]
            r = touch_distances + 100 * obj_height

        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'obj_distance',
            'hand_and_obj_distance',
            'touch_distance',
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'touch_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

    def log_diagnostics(self, paths):
        diagnostics = self.get_diagnostics(paths)

        logger.record_tabular('HandDistanceMean', diagnostics['hand_distance Mean'])
        logger.record_tabular('ObjectDistanceMean', diagnostics['obj_distance Mean'])
        logger.record_tabular('TouchDistanceMean', diagnostics['touch_distance Mean'])

        logger.record_tabular('FinalHandDistanceMean', diagnostics['Final hand_distance Mean'])
        logger.record_tabular('FinalObjectDistanceMean', diagnostics['Final obj_distance Mean'])

        logger.record_tabular('FinalHandSuccessMean', diagnostics['Final hand_success Mean'])
        logger.record_tabular('FinalObjectSuccessMean', diagnostics['Final obj_success Mean'])
        logger.record_tabular('FinalHandAndObjSuccessMean', diagnostics['Final hand_and_obj_success Mean'])


if __name__ == "__main__":
    import time
    env = SawyerPickAndPlaceEnv()
    env.reset()
    diff = np.array([0,0,0])
    qpos_arr = []
    for i in range(200):
        #env.render()
        #action = env.action_space.sample()
        a=np.sin(i/10)
        action = np.array([diff[0], diff[1], diff[2], a])
        obs, reward, done, info = env.step(action)  # take a random action

        ee_pos = obs[:3]
        obj_pos =obs[3:6]
        hand_goals = obs[6:9]
        obj_goals = obs[9:12]

        qpos_arr.append(env.sim.get_state().qpos)
        print(env.get_gripper_state())

        #diff = obj_pos - ee_pos
        diff = [0,0,0]

        time.sleep(env.dt)

    qpos = np.vstack(qpos_arr)
    from pprint import pprint

    std = qpos.std(axis=0)
    min = qpos.min(axis=0)
    max = qpos.max(axis=0)
    for i in range(std.shape[0]):
        print(i, std[i], min[i], max[i])