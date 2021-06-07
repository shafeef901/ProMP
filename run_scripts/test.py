from run_grbal import run_experiment
from experiment_utils.run_sweep import run_sweep
from meta_policy_search_model_based.envs import *


config = {
                # Environment
                'env': [HalfCheetahHFieldEnv],
                'env_name': ['HalfCheetahHFieldEnv'],
                'max_path_length': [1000],
                'task': [None],
                'normalize': [True],
                 'n_itr': [50],
                'discount': [1.],

                # Policy
                'n_candidates': [500],
                'horizon': [10],
                'use_cem': [False],
                'num_cem_iters': [5],

                # Training
                'num_rollouts': [5],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.99],
                'initial_random_samples': [True],

                # Dynamics Model
                'meta_batch_size': [10],
                'hidden_nonlinearity_model': ['relu'],
                'learning_rate': [1e-3],
                'inner_learning_rate': [0.001],
                'hidden_sizes_model': [(512, 512, 512)],
                'dynamic_model_epochs': [100],
                'adapt_batch_size': [16],

                #  Other
                'n_parallel': [1],

    }

run_sweep(run_experiment, config, 'grbal', 't2.micro')