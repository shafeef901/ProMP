import numpy as np
from collections import OrderedDict

from meta_mb.rllab_maml.misc import ext
import meta_mb.sandbox_maml.rocky.tf.core.layers as L
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

import itertools
import time

import tensorflow as tf
from meta_mb.sandbox_maml.rocky.tf.misc.xavier_init import xavier_initializer
from meta_mb.sandbox_maml.rocky.tf.core.utils import make_input, make_param_layer, forward_param_layer, forward_dense_layer

load_params = True


class MAMLImprovedGaussianMLPPolicy(StochasticPolicy, Serializable):
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
            grad_step_size=0.1,
            trainable_step_size=True,
            stop_grad=False,
            param_noise_std=0.00
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
        self.stop_grad = stop_grad
        self.name = name
        self.param_noise_std = param_noise_std
        self.all_param_ph = None
        self.params_ph = None
        self.compiled = False

        with tf.variable_scope(self.name):
            self.param_noise_std_ph = tf.placeholder_with_default(0.0, ())  # default parameter noise std is 0 -> no noise

            # create network
            if mean_network is None:
                self.all_params = self.create_MLP(  # TODO: this should not be a method of the policy! --> helper
                    name="mean_network",
                    output_dim=self.action_dim,
                    hidden_sizes=hidden_sizes,
                    bias_transform=bias_transform,
                    param_noise_std_ph=self.param_noise_std_ph
                )
                self.input_tensor, _ = self.forward_MLP('mean_network', self.all_params,
                                                        reuse=None,  # Need to run this for batch norm
                                                        bias_transform=bias_transform,
                                                        )
                forward_mean = lambda x, params, is_train: self.forward_MLP('mean_network', params,
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
                self.all_param_vals = None

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

                super(MAMLImprovedGaussianMLPPolicy, self).__init__(env_spec)

                dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
                mean_var = dist_info_sym["mean"]
                log_std_var = dist_info_sym["log_std"]

                # pre-update policy
                self._init_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor, self.param_noise_std_ph],
                    outputs=[mean_var, log_std_var],
                )
                self._cur_f_dist = self._init_f_dist

                # stepsize for each parameter
                self.param_step_sizes = {}
                with tf.variable_scope("mean_network", reuse=True):
                    for key, param in self.all_params.items():
                        shape = param.get_shape().as_list()
                        init_stepsize = np.ones(shape, dtype=np.float32) * grad_step_size
                        self.param_step_sizes[key + "_step_size"] = tf.Variable(initial_value=init_stepsize,
                                                                                name='step_size_%s' % key,
                                                                                dtype=tf.float32,
                                                                                trainable=trainable_step_size)

    @property
    def vectorized(self):
        return True

    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor

    def compute_updated_dists(self, samples):
        """
        Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        """
        start = time.time()
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        sess = tf.get_default_session()

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i],
                                 'observations', 'actions', 'advantages')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = obs_list + action_list + adv_list

        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
        init_param_values = None
        if self.all_param_vals is not None:  # skip this in first iteration
            init_param_values = self.get_variable_values(self.all_params)

        for i in range(num_tasks):
            if self.all_param_vals is not None:  # skip this in first iteration
                self.assign_params(self.all_params, self.all_param_vals[i])

        if 'all_fast_params_tensor' not in dir(self):  # only enter if first iteration
            # make computation graph once
            self.all_fast_params_tensor = []
            # compute gradients for a current task (symbolic)
            for i in range(num_tasks):
                # compute gradients for a current task (symbolic)
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i],
                                                                     [self.all_params[key] for key in
                                                                      update_param_keys])))

                # gradient update for params of current task (symbolic)
                fast_params_tensor = OrderedDict(zip(update_param_keys,
                                                     [self.all_params[key] - tf.multiply(
                                                         self.param_step_sizes[key + "_step_size"], gradients[key]) for
                                                      key in update_param_keys]))

                # add step sizes to fast_params_tensor
                fast_params_tensor.update(self.param_step_sizes)

                # undo gradient update for no_update_params (symbolic)
                for k in no_update_param_keys:
                    fast_params_tensor[k] = self.all_params[k]

                # tensors that represent the updated params for all of the tasks (symbolic)
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step

        self.all_param_vals = sess.run(self.all_fast_params_tensor,
                                       feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
        # if self.all_params_ph is None:
        #     self.
        if self.all_param_ph is None:
            self.all_param_ph = [OrderedDict([(key, tf.placeholder(tf.float32, shape=value.shape))
                                              for key, value in self.all_param_vals[0].items()])
                                 for _ in range(num_tasks)]

        # reset parameters to original ones
        if init_param_values is not None:  # skip this in first iteration
            self.assign_params(self.all_params, init_param_values)

        # compile the _cur_f_dist with updated params
        if not self.compiled:
            outputs = []
            with tf.variable_scope("post_updated_policy"):
                inputs = tf.split(self.input_tensor, num_tasks, 0)
                for i in range(num_tasks):
                    # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
                    task_inp = inputs[i]
                    info, _ = self.dist_info_sym(task_inp, dict(), all_params=self.all_param_ph[i],
                                                 is_training=False)

                    outputs.append([info['mean'], info['log_std']])

                self.__cur_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor, self.param_noise_std_ph] + sum([list(param_ph.values())
                                                                               for param_ph in self.all_param_ph], []),
                    outputs=outputs,
                )
            self.compiled = True
        self._cur_f_dist = self.__cur_f_dist

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

    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        self._cur_f_dist = self._init_f_dist
        self._cur_f_dist_i = None
        self.all_param_vals = None
        self.first_inner_step = True

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

    def updated_dist_info_sym(self, task_id, surr_obj, new_obs_var, params_dict=None, is_training=True):
        """
        symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one grad step.
        """
        old_params_dict = params_dict

        if old_params_dict == None:
            old_params_dict = self.all_params
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [
            old_params_dict[key] - tf.multiply(self.param_step_sizes[key + "_step_size"], gradients[key]) for key in
            update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]

        return self.dist_info_sym(new_obs_var, all_params=params_dict, is_training=is_training)

    @overrides
    def get_action(self, observation, idx=None, param_noise_std=None):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        if param_noise_std is None:
            param_noise_std = self.param_noise_std

        flat_obs = self.observation_space.flatten(observation)
        f_dist = self._cur_f_dist
        mean, log_std = [x[0] for x in f_dist([flat_obs], param_noise_std)]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations, param_noise_std=None):
        # this function takes a numpy array observations and outputs sampled actions.
        # Assumes that there is one observation per post-update policy distr
        if param_noise_std is None:
            param_noise_std = self.param_noise_std

        flat_obs = self.observation_space.flatten_n(observations)
        result = self._cur_f_dist(flat_obs, param_noise_std)

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.array([res[0] for res in result])[:, 0, :]
            log_stds = np.array([res[1] for res in result])[:, 0, :]

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_actions_batch(self, observations, param_noise_std=None):
        """

        :param observations: list of numpy arrays containing a batch of observations corresponding to a task -
                             shape of each numpy array must be (batch_size, ndim_obs)
        :return: actions - shape (batch_size * tasks, ndim_obs)
        """
        batch_size = None

        if param_noise_std is None:
            param_noise_std = self.param_noise_std

        # assert that obs of all tasks have the same batch size
        for obs_batch in observations:
            if batch_size is None:
                batch_size = obs_batch.shape[0]
            else:
                assert obs_batch.shape[0] == batch_size

                obs_batch.flatten()

        obs_stack = np.concatenate(observations, axis=0)
        if self._cur_f_dist == self._init_f_dist:
            result = self._cur_f_dist(obs_stack, param_noise_std)
        else:
            params = self.all_param_vals
            result = self._cur_f_dist(obs_stack, param_noise_std,
                                  *sum([list(param.values()) for param in params], []),
                                      )

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.concatenate([res[0] for res in result], axis=0)
            log_stds = np.concatenate([res[1] for res in result], axis=0)

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def cur_f_dist(self, *args):
        if self._cur_f_dist == self._init_f_dist:
            return self._cur_f_dist(*args)
        else:
            params = self.all_param_vals
            return self._cur_f_dist(*args, sum([list(param.values()) for param in params], []))

    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables(scope=self.name)
        else:
            params = tf.global_variables(scope=self.name)

        # params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith('output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]

        return params

    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
                   output_W_init=xavier_initializer(), output_b_init=tf.zeros_initializer(),
                   weight_normalization=False, bias_transform=False, param_noise_std_ph=None):

        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            if bias_transform:
                for idx, hidden_size in enumerate(hidden_sizes):
                    # hidden layers
                    W, b, bias_transform, cur_shape = make_dense_layer_with_bias_transform(
                        cur_shape,
                        num_units=hidden_size,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        bias_transform=hidden_b_init,
                        weight_norm=weight_normalization,
                    )
                    all_params['W' + str(idx)] = W
                    all_params['b' + str(idx)] = b
                    all_params['bias_transform' + str(idx)] = bias_transform

                # output layer
                W, b, bias_transform, _ = make_dense_layer_with_bias_transform(
                    cur_shape,
                    num_units=output_dim,
                    name='output',
                    W=hidden_W_init,
                    b=hidden_b_init,
                    bias_transform=hidden_b_init,
                    weight_norm=weight_normalization,
                    param_noise_std_ph=param_noise_std_ph
                )
                all_params['W' + str(len(hidden_sizes))] = W
                all_params['b' + str(len(hidden_sizes))] = b
                all_params['bias_transform' + str(len(hidden_sizes))] = bias_transform

            else:
                for idx, hidden_size in enumerate(hidden_sizes):
                    W, b, cur_shape = make_dense_layer(
                        cur_shape,
                        num_units=hidden_size,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        weight_norm=weight_normalization,
                    )
                    all_params['W' + str(idx)] = W
                    all_params['b' + str(idx)] = b
                W, b, _ = make_dense_layer(
                    cur_shape,
                    num_units=output_dim,
                    name='output',
                    W=output_W_init,
                    b=output_b_init,
                    weight_norm=weight_normalization,
                    param_noise_std_ph=param_noise_std_ph
                )
                all_params['W' + str(len(hidden_sizes))] = W
                all_params['b' + str(len(hidden_sizes))] = b

        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False, bias_transform=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor

            l_hid = l_in

            for idx in range(self.n_hidden):
                bias_transform_ = all_params['bias_transform' + str(idx)] if bias_transform else None
                l_hid = forward_dense_bias_transform(l_hid, all_params['W' + str(idx)], all_params['b' + str(idx)],
                                                     bias_transform=bias_transform_,
                                                     batch_norm=batch_normalization,
                                                     nonlinearity=self.hidden_nonlinearity,
                                                     scope=str(idx), reuse=reuse,
                                                     is_training=is_training
                                                     )

            bias_transform = all_params['bias_transform' + str(self.n_hidden)] if bias_transform else None
            output = forward_dense_bias_transform(l_hid, all_params['W' + str(self.n_hidden)],
                                                  all_params['b' + str(self.n_hidden)],
                                                  bias_transform=bias_transform, batch_norm=False,
                                                  nonlinearity=self.output_nonlinearity,
                                                  )
            return l_in, output

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

    def get_mean_step_size(self):
        """ returns the mean gradient stepsize """
        sess = tf.get_default_session()
        return np.concatenate(
            [sess.run(step_size_values).flatten() for step_size_values in self.param_step_sizes.values()]).mean()

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


class PPOMAMLImprovedGaussianMLPPolicy(MAMLImprovedGaussianMLPPolicy):

    def compute_updated_dists(self, samples):
        """
        Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        """
        start = time.time()
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        sess = tf.get_default_session()

        obs_list, action_list, adv_list, distr_list = [], [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i],
                                 'observations', 'actions', 'advantages', 'agent_infos')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])
            distr_list.extend(inputs[3][k] for k in self.distribution.dist_info_keys)
             
        inputs = obs_list + action_list + adv_list + distr_list

        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
        if self.first_inner_step:  # skip this in first iteration
            self.init_param_values = self.get_variable_values(self.all_params)
            self.all_param_vals = [self.get_variable_values(self.all_params) for _ in range(num_tasks)]

        if self.params_ph is None:
            self.params_ph = [OrderedDict([(key, tf.placeholder(tf.float32, shape=value.shape))
                                              for key, value in self.all_params.items()])
                                 for _ in range(num_tasks)]

        if 'all_fast_params_tensor' not in dir(self):  # only enter if first iteration
            # make computation graph once
            self.all_fast_params_tensor = []
            # compute gradients for a current task (symbolic)
            for i in range(num_tasks):
                # compute gradients for a current task (symbolic)
                for key in self.all_params.keys():
                    tf.assign(self.all_params[key], self.params_ph[i][key])
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i],
                                                                     [self.all_params[key] for key in
                                                                      update_param_keys])))

                # gradient update for params of current task (symbolic)
                fast_params_tensor = OrderedDict(zip(update_param_keys,
                                                     [self.all_params[key] - tf.multiply(
                                                         self.param_step_sizes[key + "_step_size"], gradients[key]) for
                                                      key in update_param_keys]))

                # add step sizes to fast_params_tensor
                fast_params_tensor.update(self.param_step_sizes)

                # undo gradient update for no_update_params (symbolic)
                for k in no_update_param_keys:
                    fast_params_tensor[k] = self.all_params[k]

                # tensors that represent the updated params for all of the tasks (symbolic)
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        feed_dict = list(zip(self.input_list_for_grad, inputs))
        feed_dict_params = list((self.params_ph[task][key], self.all_param_vals[task][key])
                                     for task in range(num_tasks) for key in self.params_ph[0].keys())
        feed_dict = dict(feed_dict + feed_dict_params)

        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=feed_dict)
        if self.all_param_ph is None:
            self.all_param_ph = [OrderedDict([(key, tf.placeholder(tf.float32, shape=value.shape))
                                              for key, value in self.all_param_vals[0].items()])
                                 for _ in range(num_tasks)]

        # reset parameters to original ones
        self.assign_params(self.all_params, self.init_param_values)

        # compile the _cur_f_dist with updated params
        if not self.compiled:
            outputs = []
            with tf.variable_scope("post_updated_policy"):
                inputs = tf.split(self.input_tensor, num_tasks, 0)
                for i in range(num_tasks):
                    # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
                    task_inp = inputs[i]
                    info, _ = self.dist_info_sym(task_inp, dict(), all_params=self.all_param_ph[i],
                                                 is_training=False)

                    outputs.append([info['mean'], info['log_std']])

                self.__cur_f_dist = tensor_utils.compile_function(
                    inputs=[self.input_tensor, self.param_noise_std_ph] + sum([list(param_ph.values())
                                                                               for param_ph in self.all_param_ph], []),
                    outputs=outputs,
                )
            self.compiled = True
        self._cur_f_dist = self.__cur_f_dist
        self.first_inner_step = False