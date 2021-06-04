from meta_policy_search_model_based.dynamics.mlp_dynamics import MLPDynamicsModel
from meta_policy_search_model_based.trainers.mb_trainer import Trainer
from meta_policy_search_model_based.policies.mpc_controller import MPCController
from meta_policy_search_model_based.samplers.sampler import Sampler
from meta_policy_search_model_based.logger import logger
from meta_policy_search_model_based.envs.normalized_env import normalize
from meta_policy_search_model_based.utils.utils import ClassEncoder
from meta_policy_search_model_based.samplers.model_sample_processor import ModelSampleProcessor
from meta_policy_search_model_based.envs import *
import json
import time
import os

EXP_NAME = 'mb_mpc'

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def run_experiment(config):
    start_itr = 0
    idx = int(time.time())
    exp_dir = '{}/data/{}/run_{}'.format(meta_policy_search_path, EXP_NAME, idx)

    # change flag to load checkpoint, dont forget to put the checkpoint number
    load_checkpoint = False

    if load_checkpoint:
        start_itr = 180
        checkpoint_name = exp_dir + '/checkpoints/cp_{}.meta'.format(start_itr)
        assert os.path.exists(checkpoint_name), "Provide valid checkpoint name."

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env'](reset_every_episode=True, task=config['task']))

    dynamics_model = MLPDynamicsModel(
        name="dyn_model",
        env=env,
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['batch_size'],
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
    )

    sample_processor = ModelSampleProcessor(recurrent=False)

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        checkpoint_path=exp_dir,
        start_itr=start_itr,
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
            # Environment
            'env': HalfCheetahEnv,
            'task': None,

            # Policy
            'n_candidates': 2000,
            'horizon': 20,
            'use_cem': False,
            'num_cem_iters': 5,
            'discount': 1.,

            # Sampling
            'max_path_length': 100,
            'num_rollouts': 10,
            'initial_random_samples': True,

            # Training
            'n_itr': 50,
            'learning_rate': 1e-3,
            'batch_size': 128,
            'dynamic_model_epochs': 100,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'hidden_sizes': (512, 512),
            'hidden_nonlinearity': 'relu',


            #  Other
            'n_parallel': 1,
            }

    run_experiment(config)