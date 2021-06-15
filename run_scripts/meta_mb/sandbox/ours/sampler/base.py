import numpy as np
from meta_mb.rllab.misc import special
from meta_mb.rllab.misc import tensor_utils
from meta_mb.rllab.algos import util
from meta_mb.rllab.sampler.base import Sampler
from meta_mb.rllab.sampler.base import BaseSampler
import meta_mb.rllab.misc.logger as logger

class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths, log=True, log_prefix='', return_reward=False):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

        if log == 'reward':
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
        elif log == 'all' or log is True:
            logger.record_tabular('Iteration', itr)
            logger.record_tabular(log_prefix + 'AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(log_prefix + 'ExplainedVariance', ev)
            logger.record_tabular(log_prefix + 'NumTrajs', len(paths))
            logger.record_tabular(log_prefix + 'Entropy', ent)
            logger.record_tabular(log_prefix + 'Perplexity', np.exp(ent))
            logger.record_tabular(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        if return_reward:
            return samples_data, np.mean(undiscounted_returns)
        else:
            return samples_data

class RandomBaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo


    def process_samples(self, itr, paths, log=True, log_prefix=''):

        # compute discounted rewards - > returns
        returns = []
        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            returns.append(path["returns"])


        observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][:-1] for path in paths])
        next_observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][1:] for path in paths])
        actions_dynamics = tensor_utils.concat_tensor_list([path["actions"][:-1] for path in paths])
        timesteps_dynamics = tensor_utils.concat_tensor_list([np.arange((len(path["observations"]) - 1)) for path in paths])

        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])

        samples_data = dict(
            observations_dynamics=observations_dynamics,
            next_observations_dynamics=next_observations_dynamics,
            actions_dynamics=actions_dynamics,
            timesteps_dynamics=timesteps_dynamics,

            rewards=rewards,
            returns=returns,
        )

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        if log:
            logger.record_tabular('Iteration', itr)
            logger.record_tabular(log_prefix + 'AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(log_prefix + 'NumTrajs', len(paths))
            logger.record_tabular(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        return samples_data

class ModelBaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths, log=True, log_prefix=''):
        """ Compared with the standard Sampler, ModelBaseSampler.process_samples provides 3 additional data fields
                - observations_dynamics
                - next_observations_dynamics
                - actions_dynamics
            since the dynamics model needs (obs, act, next_obs) for training, observations_dynamics and actions_dynamics
            skip the last step of a path while next_observations_dynamics skips the first step of a path
        """

        assert len(paths) > 0
        # compute discounted rewards - > returns
        returns = []
        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            returns.append(path["returns"])


        observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][:-1] for path in paths])
        next_observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][1:] for path in paths])
        actions_dynamics = tensor_utils.concat_tensor_list([path["actions"][:-1] for path in paths])
        timesteps_dynamics = tensor_utils.concat_tensor_list([np.arange((len(path["observations"])-1)) for path in paths])

        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])

        samples_data = dict(
            observations_dynamics=observations_dynamics,
            next_observations_dynamics=next_observations_dynamics,
            actions_dynamics=actions_dynamics,
            timesteps_dynamics=timesteps_dynamics,
            rewards=rewards,
            returns=returns,
            paths=paths,
        )

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        if log:
            logger.record_tabular('Iteration', itr)
            logger.record_tabular(log_prefix + 'AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(log_prefix + 'NumTrajs', len(paths))
            logger.record_tabular(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        return samples_data

class MAMLBaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths, log=True, log_prefix='', return_reward=False):
        baselines = []
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
        if log:
            logger.log("fitting baseline...")

        self.algo.baseline.fit(paths, log=log)
        if log:
            logger.log("fitted")


        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        if log == 'reward':
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
        elif log == 'all' or log is True:
            logger.record_tabular('Iteration', itr)
            logger.record_tabular(log_prefix + 'AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(log_prefix + 'ExplainedVariance', ev)
            logger.record_tabular(log_prefix + 'NumTrajs', len(paths))
            logger.record_tabular(log_prefix + 'Entropy', ent)
            logger.record_tabular(log_prefix + 'Perplexity', np.exp(ent))
            logger.record_tabular(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        if return_reward:
            return samples_data, np.mean(undiscounted_returns)
        else:
            return samples_data
