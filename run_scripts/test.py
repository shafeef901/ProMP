# from meta_policy_search.utils.logger import Logger
# from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv

# x = Logger("test_dump",1)

# params = dict(hf = HalfCheetahRandDirecEnv())

# x.save_itr_params("1234", params)

import numpy as np
print("1")
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
print("1")
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
print("1")
from meta_policy_search.envs.normalized_env import normalize
print("1")
from meta_policy_search.meta_algos.pro_mp import ProMP
print("1")
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
print("1")
from meta_policy_search.utils.utils import set_seed, ClassEncoder
print("1")

import os


meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


config = {
            'seed': 1,

            'baseline': 'LinearFeatureBaseline',

            'env': 'HalfCheetahRandDirecEnv',

            # sampler config
            'rollouts_per_meta_task': 20,
            'max_path_length': 100,
            'parallel': True,

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

set_seed(config['seed'])


env = globals()[config['env']]() # instantiate env
print("Nomralise")
env = normalize(env)


print("policy")
policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )


print("algo")
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

