import joblib
import tensorflow as tf
import argparse
import time
import mujoco_py
from meta_policy_search.samplers.utils import rollout

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("param", type=str)
    # parser.add_argument('--max_path_length', type=int, default=1000,
    #                     help='Max length of rollout')
    # parser.add_argument('--speedup', type=float, default=1,
    #                     help='Speedup')
    # parser.add_argument('--video_filename', type=str,
    #                     help='path to the out video file')
    # parser.add_argument('--prompt', type=bool, default=False,
    #                     help='Whether or not to prompt for more sim')
    # parser.add_argument('--ignore_done', type=bool, default=False,
    #                     help='Whether stop animation when environment done or continue anyway')
    # args = parser.parse_args()

    with tf.compat.v1.Session().as_default() as sess:
        pkl_path = "../data/pro-mp/MetalheadEnvV1RandDirec/run_1622816416/params.pkl"
        max_path_length = 1000

        print("Testing policy %s" % pkl_path)
        data = joblib.load(pkl_path)
        policy = data['policy']
        policy._pre_update_mode = True
        # policy.meta_batch_size = 1
        env = data['env']
        # env.set_task(-1.0)
        path = rollout(env, policy, max_path_length=max_path_length, animated=True, speedup=10,
                       video_filename='sim_out.mp4', save_video=False, ignore_done=False)