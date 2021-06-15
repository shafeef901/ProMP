from meta_mb.rllab.envs.mujoco.gather.gather_env import GatherEnv
from meta_mb.rllab.envs.mujoco.ant_env import AntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
