import os
import json
import tensorflow as tf
import numpy as np
import time
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs_dyn import *
from meta_mb.meta_algos.trpo_maml import TRPOMAML
from meta_mb.trainers.mbmpo_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.mbmpo_samplers.mbmpo_sampler import MBMPOSampler
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.envs.normalized_env import normalize


'''
    1. Change env in config
    2. Change path to load .pkl file
    
'''


if __name__ == "__main__":

    save_video = True
    mode='human'
    images = []

    kwargs = {
        'seed': 1,

        'algo': 'mbmpo',
        'baseline': LinearFeatureBaseline,
        'env': HalfCheetahHFieldEnv,
        'env_name': 'HalfCheetahHFieldEnv',

        # Problem Conf
        'n_itr': 201,
        'max_path_length': 150,
        'discount': 0.99,
        'gae_lambda': 1.,
        'normalize_adv': True,
        'positive_adv': False,
        'log_real_performance': True,
        'meta_steps_per_iter': 30,

        # Real Env Sampling
        'real_env_rollouts_per_meta_task': 1,
        'parallel': True,
        'fraction_meta_batch_size': .5,

        # Dynamics Model
        'num_models': 5,
        'dynamics_hidden_sizes': (500, 500),
        'dyanmics_hidden_nonlinearity': 'relu',
        'dyanmics_output_nonlinearity': None,
        'dynamics_max_epochs': 50,
        'dynamics_learning_rate': 1e-3,
        'dynamics_batch_size': 128,
        'dynamics_buffer_size': 10000,
        'deterministic': True,


        # Policy
        'policy_hidden_sizes': (64, 64),
        'policy_learn_std': True,
        'policy_output_nonlinearity': None,

        # Meta-Algo
        'meta_batch_size': 20,  # Note: It has to be multiple of num_models
        'rollouts_per_meta_task': 20,
        'num_inner_grad_steps': 1,
        'inner_lr': 0.001,
        'inner_type': 'log_likelihood',
        'step_size': 0.01,
        'exploration': False,
        'sample_from_buffer': True,

        'scope': None,
        'exp_tag': '', # For changes besides hyperparams
        'exp_name': '',  # Add time-stamp here to not overwrite the logging
    }


    sess = tf.InteractiveSession()

    pkl_path = "../data/mbmpo/HalfCheetahHFieldEnv/run_1623261030/params.pkl"
    max_path_length = 600

    print("Testing policy %s" % pkl_path)
    data = joblib.load(pkl_path)
    policy = data['policy']
    policy.switch_to_pre_update()

    baseline = LinearFeatureBaseline()

    env = data['env']


    env_sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=kwargs['real_env_rollouts_per_meta_task'],
            meta_batch_size=kwargs['meta_batch_size'],
            max_path_length=kwargs['max_path_length'],
            parallel=kwargs['parallel'],
        )

    model_sample_processor = MAMLSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

    algo = TRPOMAML(
            policy=policy,
            step_size=kwargs['step_size'],
            inner_type=kwargs['inner_type'],
            inner_lr=kwargs['inner_lr'],
            meta_batch_size=kwargs['meta_batch_size'],
            num_inner_grad_steps=kwargs['num_inner_grad_steps'],
            exploration=kwargs['exploration'],
    )


    uninit_vars = [var for var in tf.compat.v1.global_variables() if not sess.run(tf.is_variable_initialized(var))]
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
            env.render(mode)
            action, _ = policy.get_action(obs, task_i)
            obs, reward, done, _ = env.step(action)
            print(reward)
            time.sleep(0.0016)

            # if save_video:
            #     image = env.render(mode)
            #     images.append(image)

            if done:
                break

        # if save_video:
        #     fps = int(1/0.0016)
        #     clip = mpy.ImageSequenceClip(images, fps=fps)
        #     if video_filename[-3:] == 'gif':
        #         if tasks[task_i] == 1:
        #             video_filename = 'sim_out_fwd.mp4'
        #         else:
        #             video_filename = 'sim_out_back.mp4'
        #         clip.write_gif(video_filename, fps=fps)
        #     else:

        #         clip.write_videofile(video_filename, fps=fps)
        #     print("Video saved at %s" % video_filename)