from meta_policy_search_model_based.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel
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

def run_experiment(config):

    EXP_NAME = 'grbal'
    meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

    start_itr = 0
    idx = int(time.time())
    exp_dir = '{}/data/{}/{}/run_{}'.format(meta_policy_search_path, EXP_NAME, config['env_name'], idx)

    # change flag to load checkpoint, dont forget to put the checkpoint number
    load_checkpoint = False

    if load_checkpoint:
        start_itr = 180
        checkpoint_name = exp_dir + '/checkpoints/cp_{}.meta'.format(start_itr)
        assert os.path.exists(checkpoint_name), "Provide valid checkpoint name."

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env'](reset_every_episode=True, task=config['task']))

    dynamics_model = MetaMLPDynamicsModel(
        name="dyn_model",
        env=env,
        meta_batch_size=config['meta_batch_size'],
        inner_learning_rate=config['inner_learning_rate'],
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes_model'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity_model'],
        batch_size=config['adapt_batch_size'],
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
        n_parallel=config['n_parallel'],
        max_path_length=config['max_path_length'],
        num_rollouts=config['num_rollouts'],
        adapt_batch_size=config['adapt_batch_size'],  # Comment this out and it won't adapt during rollout
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

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
                'env': HalfCheetahHFieldEnv,
                'env_name': 'HalfCheetahHFieldEnv',
                'max_path_length': 1000,
                'task': None,
                'normalize': True,
                 'n_itr': 50,
                'discount': 1.,

                # Policy
                'n_candidates': 500,
                'horizon': 10,
                'use_cem': False,
                'num_cem_iters': 5,

                # Training
                'num_rollouts': 5,
                'valid_split_ratio': 0.1,
                'rolling_average_persitency': 0.99,
                'initial_random_samples': True,

                # Dynamics Model
                'meta_batch_size': 10,
                'hidden_nonlinearity_model': 'relu',
                'learning_rate': 1e-3,
                'inner_learning_rate': 0.001,
                'hidden_sizes_model': (512, 512, 512),
                'dynamic_model_epochs': 100,
                'adapt_batch_size': 16,

                #  Other
                'n_parallel': 1,

    }

    print(config['env'])

    run_experiment(config)
