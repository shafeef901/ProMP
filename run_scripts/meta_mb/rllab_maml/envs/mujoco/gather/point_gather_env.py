from meta_mb.rllab_maml.envs.mujoco.gather.gather_env import GatherEnv
from meta_mb.rllab_maml.envs.mujoco.point_env import PointEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
