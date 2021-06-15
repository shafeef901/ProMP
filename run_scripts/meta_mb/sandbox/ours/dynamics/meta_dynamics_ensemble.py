from meta_mb.sandbox.rocky.tf.core.network import MLP

import tensorflow as tf
import numpy as np
from meta_mb.rllab.core.serializable import Serializable
from meta_mb.sandbox.rocky.tf.core.layers_powered import LayersPowered
from meta_mb.sandbox.rocky.tf.misc import tensor_utils
from meta_mb.rllab.misc import logger
from collections import OrderedDict
import meta_mb.sandbox.rocky.tf.core.layers as L
import joblib


class MetaDynamicsEnsemble(LayersPowered, Serializable):
    """
    Class for a meta dynamics model that holds several model ensembles
    """

    def __init__(self,
                 name,
                 env_spec,
                 num_ensembles=5,
                 num_models_per_ensemble=3,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=500,
                 step_size=0.001,
                 weight_normalization=False,
                 normalize_input=True,
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input

        self.batch_size = batch_size
        self.step_size = step_size
        self.num_ensembles = num_ensembles
        self.num_models_per_ensemble = num_models_per_ensemble
        self.num_models = num_ensembles * num_models_per_ensemble

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env_spec.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env_spec.action_space.shape[0]

        # set model - ensemble assignment
        self.model_ensemble_assignment = [
            list(range(i * self.num_models_per_ensemble, (i + 1) * self.num_models_per_ensemble))
            for i in range(self.num_ensembles)]

        with tf.variable_scope(name):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            mlps = []
            delta_preds = []
            self.obs_next_pred = []
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i)):
                    mlp = MLP(name,
                              obs_space_dims,
                              hidden_sizes,
                              hidden_nonlinearity,
                              output_nonlinearity,
                              input_var=self.nn_input,
                              input_shape = (obs_space_dims+action_space_dims,),
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                delta_preds.append(mlp.output)
            self.delta_pred = tf.stack(delta_preds, axis=2) # shape: (batch_size, ndim_obs, n_models)

            # define loss and train_op
            self.loss = tf.reduce_mean((self.delta_ph[:, :, None] - self.delta_pred)**2)
            self.optimizer = tf.train.AdamOptimizer(self.step_size)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = tensor_utils.compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def fit(self, obs, act, obs_next, epochs=50, compute_normalization=True, verbose=False):
        """
        Fits the all the NN dynamics models
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param verbose: logging verbosity
        """
        assert obs.ndim == 2 and obs.shape[1]==self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims


        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        # create data queue
        next_batch, iterator = self._data_input_fn(obs, act, delta, batch_size=self.batch_size)

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            sess.run(iterator.initializer,
                     feed_dict={self.obs_dataset_ph: obs,
                                self.act_dataset_ph: act,
                                self.delta_dataset_ph: delta})

            batch_losses = []

            while True:
                try:
                    obs_batch, act_batch, delta_batch = sess.run(next_batch)
                    batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.obs_ph: obs_batch,
                                                                                    self.act_ph: act_batch,
                                                                                    self.delta_ph: delta_batch})

                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    if verbose:
                        logger.log("Training NNDynamicsModel - finished epoch {} -- mean loss: {}".format(epoch, np.mean(batch_losses)))
                    break

    def predict(self, obs, act, pred_type='rand'):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param pred_type:  prediction type
                   - rand: choose one of the models per ensemble and row randomly
                   - mean: mean prediction all models within an ensemble
                   - all: returns the prediction of all the models
        :return: pred_obs_next: predicted batch of next observations -
                                shape:  (n_samples, ndim_obs, n_ensembles) - in case of 'rand' and 'mean' mode
                                        (n_samples, ndim_obs, n_models) - in case of 'all' mode
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self.f_delta_pred(obs, act))

        assert delta.ndim == 3

        pred_obs = obs_original[:, :, None] + delta

        batch_size = delta.shape[0]
        if pred_type == 'rand':
            # randomly selecting the prediction of one model per ensemble in each row
            id_per_ensemble = np.stack([np.random.choice(self.model_ensemble_assignment[i], size=batch_size) for i in range(self.num_ensembles)], axis=1)
            pred_obs = np.stack([pred_obs[row, :, model_ids].T for row, model_ids in enumerate(id_per_ensemble)], axis=0)
        elif pred_type == 'mean':
            pred_obs = np.stack([np.mean(pred_obs[:, :, ensemble_ids], axis=2) for ensemble_ids in self.model_ensemble_assignment], axis=2)
        elif pred_type == 'all':
            pass
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')
        return pred_obs

    def predict_std(self, obs, act):
        """
        calculates the std of predicted next observations among the models within an ensemble
        given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: std_pred_obs: std of predicted next observatations - (n_samples, ndim_obs, n_ensembles)
        """
        assert self.num_models > 1, "calculating the std requires at "
        pred_obs = self.predict(obs, act, pred_type='all')
        assert pred_obs.ndim == 3
        return np.stack([np.std(pred_obs[:, :, ensemble_ids], axis=2) for ensemble_ids in self.model_ensemble_assignment], axis=2)

    def shuffle_model_ensemble_assignment(self):
        """ Randomly changes the assignment of the models to the ensembles """
        model_ids = np.arange(self.num_models)
        np.random.shuffle(model_ids)
        self.model_ensemble_assignment = [
            list(model_ids[list(range(i * self.num_models_per_ensemble, (i + 1) * self.num_models_per_ensemble))])
            for i in range(self.num_ensembles)]

    def compute_normalization(self, obs, act, obs_next):
        """
        Computes the mean and std of the data and saves it in a instance variable
        -> the computed values are used to normalize the data at train and test time
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        """

        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        delta = obs_next - obs
        assert delta.ndim == 2 and delta.shape[0] == obs_next.shape[0]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=100000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, obs.shape)
        self.act_dataset_ph = tf.placeholder(tf.float32, act.shape)
        self.delta_dataset_ph = tf.placeholder(tf.float32, delta.shape)

        dataset = tf.data.Dataset.from_tensor_slices((self.obs_dataset_ph, self.act_dataset_ph, self.delta_dataset_ph))
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized


    def __getstate__(self):
        state = LayersPowered.__getstate__(self)
        state['normalization'] = self.normalization
        state['model_ensemble_assignment'] = self.model_ensemble_assignment
        return state

    def __setstate__(self, state):
        LayersPowered.__setstate__(self, state)
        self.normalization = state['normalization']
        self.model_ensemble_assignment = state['model_ensemble_assignment']


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    if data_array.ndim == 3: # assumed shape (batch_size, ndim_obs, n_models)
        return data_array * (std[None, :, None] + 1e-10) + mean[None, :, None]
    elif data_array.ndim == 2:
        return data_array * (std[None, :] + 1e-10) + mean[None, :, None]