import numpy as np
from collections import OrderedDict

from meta_mb.sandbox.ignasi.policies.new_policy.utils import *
from meta_mb.sandbox_maml.rocky.tf.spaces.box import Box as BoxMAML
from meta_mb.sandbox.rocky.tf.spaces.box import Box

from meta_mb.rllab_maml.core.serializable import Serializable
from meta_mb.sandbox_maml.rocky.tf.policies.base import StochasticPolicy
from meta_mb.sandbox_maml.rocky.tf.distributions.diagonal_gaussian import \
    DiagonalGaussian  # This is just a util class. No params.
from meta_mb.rllab_maml.misc.overrides import overrides
from meta_mb.rllab.misc import logger
from meta_mb.rllab_maml.misc.tensor_utils import flatten_tensors, unflatten_tensors
from meta_mb.sandbox_maml.rocky.tf.misc import tensor_utils
from meta_mb.sandbox.ours.core.utils import make_dense_layer_with_bias_transform, forward_dense_bias_transform, \
    make_dense_layer
import time
import tensorflow as tf
from meta_mb.sandbox_maml.rocky.tf.misc.xavier_init import xavier_initializer
from meta_mb.sandbox_maml.rocky.tf.core.utils import make_input, make_param_layer, forward_param_layer, forward_dense_layer
load_params = True


class BaseMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            bias_transform=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.identity,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std: boolean indicating whether std shall be a trainable variable
        :param bias_transform: boolean indicating whether bias transformation shall be added to the MLP
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :param grad_step_size: (float) the step size taken in the learner's gradient update
        :param trainable_step_size: boolean indicating whether the inner grad_step_size shall be trainable
        :param stop_grad: whether or not to stop the gradient through the gradient.
        :param: parameter_space_noise: (boolean) whether parameter space noise shall be used when sampling from the policy
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box) or isinstance(env_spec.action_space, BoxMAML)

        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim,)
        self.name = name

        with tf.variable_scope(self.name):

            # create network
            if mean_network is None:
                self.all_params = create_MLP(  # TODO: this should not be a method of the policy! --> helper
                    name="mean_network",
                    input_shape=self.input_shape,
                    output_dim=self.action_dim,
                    hidden_sizes=hidden_sizes,
                    bias_transform=bias_transform,
                )
                self.input_tensor, _ = forward_MLP('mean_network', self.input_shape, self.n_hidden,
                                                   self.hidden_nonlinearity, self.output_nonlinearity,
                                                   self.all_params, reuse=None,  # Need to run this for batch norm
                                                   bias_transform=bias_transform,)

                forward_mean = lambda x, params, is_train: forward_MLP('mean_network', self.input_shape, self.n_hidden,
                                                                       self.hidden_nonlinearity,
                                                                       self.output_nonlinearity,
                                                                       params,
                                                                       input_tensor=x, is_training=is_train,
                                                                       bias_transform=bias_transform)[1]
            else:
                raise NotImplementedError('Not supported.')

            if std_network is not None:
                raise NotImplementedError('Not supported.')
            else:
                if adaptive_std:
                    raise NotImplementedError('Not supported.')
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    self.all_params['std_param'] = make_param_layer(
                        num_units=self.action_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )
                    forward_std = lambda x, params: forward_param_layer(x, params['std_param'])

                # unify forward mean and forward std into a single function
                self._forward = lambda obs, params, is_train: (
                    forward_mean(obs, params, is_train), forward_std(obs, params))

                self.std_parametrization = std_parametrization

                if std_parametrization == 'exp':
                    min_std_param = np.log(min_std)
                elif std_parametrization == 'softplus':
                    min_std_param = np.log(np.exp(min_std) - 1)
                else:
                    raise NotImplementedError

                self.min_std_param = min_std_param

                self._dist = DiagonalGaussian(self.action_dim)

                self._cached_params = {}

                super(BaseMLPPolicy, self).__init__(env_spec)

                dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
                mean_var = dist_info_sym["mean"]
                log_std_var = dist_info_sym["log_std"]

                # pre-update policy
                self._cur_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor],
                    outputs=[mean_var, log_std_var],
                )

    @property
    def vectorized(self):
        return True

    @property
    def distribution(self):
        return self._dist

    def get_variable_values(self, tensor_dict):
        sess = tf.get_default_session()
        result = sess.run(tensor_dict)
        return result

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        feed_dict = {self.assign_placeholders[key]: param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    def dist_info_sym(self, obs_var, state_info_vars=None, all_params=None, is_training=True):
        # This function constructs the tf graph, only called during beginning of meta-training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params

        mean_var, std_param_var = self._forward(obs_var, all_params, is_training)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        if return_params:
            return dict(mean=mean_var, log_std=log_std_var), all_params
        else:
            return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation, idx=None):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        flat_obs = self.observation_space.flatten(observation)
        f_dist = self._cur_f_dist
        mean, log_std = [x[0] for x in f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        # this function takes a numpy array observations and outputs sampled actions.
        # Assumes that there is one observation per post-update policy distr

        flat_obs = self.observation_space.flatten_n(observations)
        result = self._cur_f_dist(flat_obs)

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.array([res[0] for res in result])[:, 0, :]
            log_stds = np.array([res[1] for res in result])[:, 0, :]

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables(scope=self.name)
        else:
            params = tf.global_variables(scope=self.name)

        # params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith('output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]

        return params

    def get_params(self, all_params=False, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params, **tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, all_params=False, **tags):
        params = self.get_params(all_params, **tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def log_diagnostics(self, paths, prefix=''):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular(prefix + 'AveragePolicyStd', np.mean(np.exp(log_stds)))

    #### code largely not used after here except when resuming/loading a policy. ####
    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        # Not used
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def get_param_dtypes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, all_params=False, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(all_params, **tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(all_params, **tags),
                self.get_param_dtypes(all_params, **tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def set_std(self, new_std=None):
        std_params = {"std_param": self.all_params["std_param"]}
        if new_std is None:
            new_std = np.zeros(std_params["std_param"].shape)
        new_std = {"std_param": new_std}
        self.assign_params(std_params, new_std)

    def flat_to_params(self, flattened_params, all_params=False, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(all_params, **tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values(all_params=True)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params(all_params=True)))
            self.set_param_values(d["params"], all_params=True)

