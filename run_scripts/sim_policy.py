import joblib
import tensorflow as tf
import argparse
import time
import mujoco_py
import numpy as np

from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
# from meta_policy_search.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
# from meta_policy_search.envs.mujoco_envs.metalhead_v1_rand_direc import MetalheadEnvV1RandDirec
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.samplers.utils import rollout
from meta_policy_search.utils import logger


if __name__ == "__main__":

    config = {
            'seed': 1,

            'baseline': 'LinearFeatureBaseline',

            'env': 'HalfCheetahRandDirecEnv',

            # sampler config
            'rollouts_per_meta_task': 20,
            'max_path_length': 100,
            'parallel': False,

            # sample processor config
            'discount': 0.99,
            'gae_lambda': 1,
            'normalize_adv': True,

            # policy config
            'hidden_sizes': (64, 64),
            'learn_std': True, # whether to learn the standard deviation of the gaussian policy

            # ProMP config
            'inner_lr': 0.1, # adaptation step size
            'learning_rate': 1e-3, # meta-policy gradient step size
            'num_promp_steps': 5, # number of ProMp steps without re-sampling
            'clip_eps': 0.3, # clipping range
            'target_inner_step': 0.01,
            'init_inner_kl_penalty': 5e-4,
            'adaptive_inner_kl_penalty': False, # whether to use an adaptive or fixed KL-penalty coefficient
            'n_itr': 1001, # number of overall training iterations
            'meta_batch_size': 40, # number of sampled meta-tasks per iterations
            'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps

    }

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

    sess = tf.InteractiveSession()

    pkl_path = "../data/pro-mp/HalfCheetahRandDirecEnv/run_1622874740/params.pkl"
    max_path_length = 1000

    print("Testing policy %s" % pkl_path)
    data = joblib.load(pkl_path)
    policy = data['policy']
    policy.switch_to_pre_update()

    baseline = LinearFeatureBaseline()

    env = data['env']


    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )


    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))

    # Sampling tasks for pre-updating:
    tasks = env.sample_tasks(config['meta_batch_size'])
    sampler.vec_env.set_tasks(tasks)
    
    # Preupdate:
    for i in range(config['num_inner_grad_steps']):
        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % i)
        env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % i)
        algo._adapt(samples_data)

    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % config['num_inner_grad_steps'])
    env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % config['num_inner_grad_steps'])
    logger.dumpkvs()

    # Postupdate:
    while True:
        task_i = np.random.choice(range(config['meta_batch_size']))
        print("Current task {}".format(task_i))
        env.set_task(tasks[task_i])
        print(tasks[task_i])
        obs = env.reset()
        for _ in range(config['max_path_length']):
            env.render()
            action, _ = policy.get_action(obs, task_i)
            obs, reward, done, _ = env.step(action)
            time.sleep(0.001)
            if done:
                break