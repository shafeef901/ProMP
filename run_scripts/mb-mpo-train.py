from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
print("#####  1  ######")
from meta_mb.sandbox.ours.algos.ModelMAML.model_maml_trpo import ModelMAMLTRPO
print("#####  1  ######")
from meta_mb.sandbox.ours.policies.maml_improved_gauss_mlp_policy import MAMLImprovedGaussianMLPPolicy
print("#####  1  ######")

from meta_mb.sandbox.ours.envs.normalized_env import normalize
print("#####  1  ######")
from meta_mb.sandbox.ours.envs.base import TfEnv
print("#####  1  ######")

import tensorflow as tf
import sys
import argparse
import random
import os

EXP_PREFIX = 'mb-mpo'

def run_experiment(config):

	env = TfEnv(normalize(config['env'](log_scale_limit=config['log_scale_limit'])))

	dynamics_model = MLPDynamicsEnsemble(
		name="dyn_model",
		env_spec=env.spec,
		hidden_sizes=config['hidden_sizes_model'],
		weight_normalization=config['weight_normalization_model'],
		num_models=config['num_models'],
		optimizer=config['optimizer_model'],
		valid_split_ratio=config['valid_split_ratio'],
		rolling_average_persitency=config['rolling_average_persitency']
	)

	policy = MAMLImprovedGaussianMLPPolicy(
		name="policy",
		env_spec=env.spec,
		hidden_sizes=config['hidden_sizes_policy'],
		hidden_nonlinearity=config['hidden_nonlinearity_policy'],
		grad_step_size=config['fast_lr'],
		trainable_step_size=config['trainable_step_size'],
		bias_transform=config['bias_transform'],
		param_noise_std=config['param_noise_std']
	)

	baseline = LinearFeatureBaseline(env_spec=env.spec)

	algo = ModelMAMLTRPO(
		env=env,
		policy=policy,
		dynamics_model=dynamics_model,
		baseline=baseline,
		n_itr=config['n_itr'],
		n_iter=config['n_itr'],
		batch_size_env_samples=config['batch_size_env_samples'],
		batch_size_dynamics_samples=config['batch_size_dynamics_samples'],
		meta_batch_size=config['meta_batch_size'],
		initial_random_samples=config['initial_random_samples'],
		num_maml_steps_per_iter=config['num_maml_steps_per_iter'],
		reset_from_env_traj=config.get('reset_from_env_traj', False),
		max_path_length_env=config['path_length_env'],
		max_path_length_dyn=config.get('path_length_dyn', None),
		dynamic_model_max_epochs=config.get('dynamic_model_max_epochs', (500, 500)),
		discount=config['discount'],
		step_size=config["meta_step_size"],
		num_grad_updates=1,
		retrain_model_when_reward_decreases=config['retrain_model_when_reward_decreases'],
		reset_policy_std=config['reset_policy_std'],
		reinit_model_cycle=config['reinit_model_cycle'],
		frac_gpu=config.get('frac_gpu', 0.85),
		log_real_performance=True,
		clip_obs=config.get('clip_obs', True)
	)
	algo.train()

if __name__ == '__main__':

	config = {
		'seed': 22,
		'env': 'HalfCheetahEnvRandParams',
		'log_scale_limit': 0.0,
		'path_length_env': 200,

		# Model-based MAML algo spec
		'n_itr': 100,
		'fast_lr': 0.001,
		'meta_step_size': 0.01,
		'meta_batch_size': 20, # must be a multiple of num_model
		'discount': 0.99,

		'batch_size_env_samples': 1,
		'batch_size_dynamics_samples': 50,
		'initial_random_samples': 5000,
		'num_maml_steps_per_iter': 30,
		'retrain_model_when_reward_decreases': False,
		'reset_from_env_traj': False,
		'trainable_step_size': False,
		'num_models': 5,

		# neural network configuration
		'hidden_nonlinearity_policy': 'tanh',
		'hidden_nonlinearity_model': 'relu',
		'hidden_sizes_policy': (32, 32),
		'hidden_sizes_model': (512, 512),
		'weight_normalization_model': True,
		'reset_policy_std': False,
		'reinit_model_cycle': 0,
		'optimizer_model': 'adam',
		'policy': 'MAMLImprovedGaussianMLPPolicy',
		'bias_transform': False,
		'param_noise_std': 0.0,
		'dynamic_model_max_epochs': (500, 500),

		'valid_split_ratio': 0.2,
		'rolling_average_persitency': 0.99,

		# other stuff
		'exp_prefix': EXP_PREFIX,
	}

	run_experiment(config)