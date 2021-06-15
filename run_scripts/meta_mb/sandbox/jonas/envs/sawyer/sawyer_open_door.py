from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from meta_mb.sandbox.ours.envs.sawyer.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from meta_mb.sandbox.ours.envs.sawyer.core.multitask_env import MultitaskEnv
from meta_mb.sandbox.ours.envs.sawyer.base import SawyerXYZEnv


class SawyerDoorOpenEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,

            obj_init_pos=(0, 0.6, 0.02),

            fix_goal=True,
            fixed_goal=(1.571),
            # target angle for the door
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

        self.max_path_length = 150

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )
        self.observation_space = Dict([
            ('observation', self.hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),

            ('state_observation', self.hand_and_obj_space),
            ('state_desired_goal', self.hand_and_obj_space),
            ('state_achieved_goal', self.hand_and_obj_space),
        ])

        self.reset()

    @property
    def model_name(self):
        return 'sawyer_long_gripper/sawyer_door_open.xml'

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.3
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1

    def step(self, action):

        self.set_xyz_action(action[:3])

        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation

        ob = self._get_obs()

        reward, doorOpenRew = self.compute_rewards(action, ob)
        self.curr_path_length += 1

        # info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, {'doorOpenRew': doorOpenRew}

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_site_pos('doorGraspPoint')
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('goal'), 2] = (
                -1000
            )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):

        self._reset_hand()

        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']

        self.curr_path_length = 0

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def put_obj_in_hand(self):
        new_obj_pos = self.data.get_site_xpos('endeffector')
        new_obj_pos[1] -= 0.01
        self.do_simulation(-1)
        self.do_simulation(1)
        self._set_obj_xyz(new_obj_pos)

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
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

        # num_objs_in_hand = int(batch_size * p_obj_in_hand)

        # # Put object in hand
        # goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
        # goals[:num_objs_in_hand, 4] -= 0.01

        # # Put object one the table (not floating)
        # goals[num_objs_in_hand:, 5] = self.obj_init_pos[2]
        return {
            'state_desired_goal': goals,

        }

    def get_site_pos(self, siteName):

        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obs):

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')

        doorGraspPoint = self.get_site_pos('doorGraspPoint')

        doorAngleTarget = self._state_goal

        fingerCOM = (rightFinger + leftFinger) / 2

        graspDist = np.linalg.norm(doorGraspPoint - fingerCOM)

        graspRew = -graspDist

        def doorOpenReward():
            doorAngle = self.data.get_joint_qpos('doorjoint')

            if graspDist < 0.1:
                return max(10 * doorAngle, 0)

            return 0

        doorOpenRew = doorOpenReward()

        reward = graspRew + doorOpenRew

        return [reward, doorOpenRew]
        # returned in a list because that's how compute_reward in multiTask.env expects it

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        # for stat_name in [
        #     'hand_distance',
        #     'obj_distance',
        #     'hand_and_obj_distance',
        #     'touch_distance',
        #     'hand_success',
        #     'obj_success',
        #     'hand_and_obj_success',
        #     'touch_success',
        # ]:
        #     stat_name = stat_name
        #     stat = get_stat_in_paths(paths, 'env_infos', stat_name)
        #     statistics.update(create_stats_ordered_dict(
        #         '%s%s' % (prefix, stat_name),
        #         stat,
        #         always_show_all_stats=True,
        #     ))
        #     statistics.update(create_stats_ordered_dict(
        #         'Final %s%s' % (prefix, stat_name),
        #         [s[-1] for s in stat],
        #         always_show_all_stats=True,
        #     ))
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

if __name__ == "__main__":
    env = SawyerDoorOpenEnv()
    import time
    env.reset()
    for _ in range(1000):
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())  # take a random action
        time.sleep(env.dt)