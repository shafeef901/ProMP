import time
from meta_mb.rllab.algos.base import RLAlgorithm
import meta_mb.rllab.misc.logger as logger
import tensorflow as tf
from meta_mb.rllab.sampler.utils import rollout
import numpy as np
from meta_mb.sandbox.ours.sampler import ModelBaseSampler, RandomVectorizedSampler, EnvVectorizedSampler

MAX_BUFFER = int(3e5)


class ModelMPCBatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size_env_samples=5000,
            initial_random_samples=None,
            max_path_length=500,
            discount=0.99,
            dynamic_model_max_epochs=(1000, 1000),
            reinit_model_cycle=0,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :param dynamics_model: Dynamics Model
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size_env_samples: Number of samples from the environment per iteration.
        :param batch_size_dynamics_samples: Number of (imaginary) samples from the dynamics model
        :param initial_random_samples: either None -> use initial policy to sample from env
                                       or int: number of random samples at first iteration to train dynamics model
                                               if provided, in the first iteration no samples from the env are generated
                                               with the policy
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param dynamic_model_epochs: (2-tuple) number of epochs to train the dynamics model
                                        (n_epochs_at_first_iter, n_epochs_after_first_iter)
        :param num_gradient_steps_per_iter: number of policy gradients steps before retraining dynamics model
        :param retrain_model_when_reward_decreases: (boolean) if true - stop inner gradient steps when performance decreases
        :param reset_policy_std: whether to reset the policy std after each iteration
        :param reinit_model_cycle: number of iterations before re-initializing the dynamics model (if 0 the dynamic model is not re-initialized at all)
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.scope = scope
        self.n_itr = n_itr
        self.plot = plot
        self.start_itr = start_itr
        self.batch_size = batch_size_env_samples
        self.initial_random_samples = initial_random_samples
        self.max_path_length = max_path_length
        self.discount = discount
        self.dynamic_model_max_epochs = dynamic_model_max_epochs
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.reinit_model = reinit_model_cycle

        # sampler for the environment
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = EnvVectorizedSampler
            else:
                sampler_cls = BatchSampler  # TODO: use batch sampler rather than Vectorized Sampler
        if sampler_args is None:
            sampler_args = dict()
        self.env_sampler = sampler_cls(self, **sampler_args)

        # sampler for (imaginary) rollouts with the estimated dynamics model
        self.model_sampler_processor = ModelBaseSampler(self).process_samples

        if self.initial_random_samples:
            self.random_sampler = RandomVectorizedSampler(self)
        else:
            self.random_sampler = None

        self.init_opt()

    def start_worker(self):
        self.env_sampler.start_worker()
        if self.initial_random_samples:
            self.random_sampler.start_worker()

    def shutdown_worker(self):
        self.env_sampler.shutdown_worker()

    def obtain_env_samples(self, itr, log=True):
        return self.env_sampler.obtain_samples(itr, log=log, log_prefix='EnvSampler-')

    def obtain_random_samples(self, itr, log=False):
        assert self.random_sampler is not None
        assert self.initial_random_samples is not None
        return self.random_sampler.obtain_samples(itr, num_samples=self.initial_random_samples, log=log, log_prefix='EnvSampler-')

    def process_samples_for_dynamics(self, itr, paths , log=False, log_prefix=''):
        return self.model_sampler_processor(itr, paths, log=log, log_prefix=log_prefix)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        self.initialize_unitialized_variables(sess)

        self.start_worker()
        start_time = time.time()
        n_env_timesteps = 0

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            with logger.prefix('itr #%d | ' % itr):
                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    new_env_paths = self.obtain_random_samples(itr, log=True)

                    n_env_timesteps += self.initial_random_samples
                    logger.record_tabular("n_timesteps", n_env_timesteps)

                    samples_data_dynamics = self.random_sampler.process_samples(itr, new_env_paths, log=True, log_prefix='EnvTrajs-') # must log in the same way as the model sampler below
                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    new_env_paths = self.obtain_env_samples(itr)

                    n_env_timesteps += self.batch_size
                    logger.record_tabular("n_timesteps", n_env_timesteps)

                    logger.log("Processing environment samples...")
                    # first processing just for logging purposes
                    self.process_samples_for_dynamics(itr, new_env_paths, log=True, log_prefix='EnvTrajs-')

                    new_samples_data_dynamics = self.process_samples_for_dynamics(itr, new_env_paths)
                    for k, v in samples_data_dynamics.items():
                        samples_data_dynamics[k] = np.concatenate([v, new_samples_data_dynamics[k]], axis=0)[
                                                   -MAX_BUFFER:]

                epochs = self.dynamic_model_max_epochs[min(itr, len(self.dynamic_model_max_epochs) - 1)]
                # fit dynamics model
                if self.reinit_model and itr % self.reinit_model == 0:
                    self.dynamics_model.reinit_model()
                    epochs = self.dynamic_model_max_epochs[0]
                logger.log("Training dynamics model for %i epochs ..." % (epochs))
                self.dynamics_model.fit(samples_data_dynamics['observations_dynamics'],
                                        samples_data_dynamics['actions_dynamics'],
                                        samples_data_dynamics['next_observations_dynamics'],
                                        epochs=epochs, verbose=True)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, None)  # , **kwargs)
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        pass

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        return dict(
            itr=itr,
            dynamics_model=self.dynamics_model,
            env=self.env,
        )

    def optimize_policy(self, itr, samples_data, log=True, log_prefix=''):
        raise NotImplementedError

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))