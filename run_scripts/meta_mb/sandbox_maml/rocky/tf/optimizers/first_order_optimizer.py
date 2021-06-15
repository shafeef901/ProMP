from meta_mb.rllab_maml.misc import ext
from meta_mb.rllab_maml.misc import logger
from meta_mb.rllab_maml.core.serializable import Serializable
from meta_mb.sandbox_maml.rocky.tf.misc import tensor_utils
# from meta_mb.rllab_maml.algo.first_order_method import parse_update_method
from meta_mb.rllab_maml.optimizers.minibatch_dataset import BatchDataset, MAMLBatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import pyprind


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            step_size=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            init_learning_rate=None,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=step_size)
        self.learning_rate = tf_optimizer_args['learning_rate']
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._init_tf_optimizer = None
        if init_learning_rate is not None:
            init_tf_optimizer_args = dict(learning_rate=init_learning_rate)
            self._init_tf_optimizer = tf_optimizer_cls(**init_tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None
        self._init_train_op = None

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        # Initializes the update opt used in the optimization
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab_maml.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            # for batch norm
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):

                self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))
                if self._init_tf_optimizer is not None:
                    self._init_train_op = self._init_tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))
        else:
            self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))
            if self._init_tf_optimizer is not None:
                self._init_train_op = self._init_tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )

        self.debug_loss = loss
        self.debug_vars = target.get_params(trainable=True)
        self.debug_target = target

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        sess = tf.get_default_session()

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            for batch in dataset.iterate(update=True):
                if self._init_train_op is not None:
                    sess.run(self._init_train_op, dict(list(zip(self._input_vars, batch))))
                    self._init_train_op = None  # only use it once
                else:
                    sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))

                if self._verbose:
                    progbar.update(len(batch[0]))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss


class MAMLPPOOptimizer(FirstOrderOptimizer):
    ## Right now it's just implemented one gradient step with all the data
    
    def update_opt(self, loss, target, inputs, inner_kl, extra_inputs=None, meta_batch_size=1, num_grad_updates=1, **kwargs):
        """
        :param inner_kl: Symbolic expression for inner kl
        :param meta_batch_size: number of MAML tasks, for batcher
        """
        super().update_opt(loss, target, inputs, extra_inputs, **kwargs)
        if extra_inputs is None:
            extra_inputs = list()
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
            f_inner_kl=lambda: tensor_utils.compile_function(inputs + extra_inputs, inner_kl)
        )
        self.meta_batch_size = meta_batch_size
        self.num_grad_updates = num_grad_updates

    def inner_kl(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_inner_kl"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()
        # Overload self._batch size
        dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)

        sess = tf.get_default_session()
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            for batch in dataset.iterate(update=True):
                if self._init_train_op is not None:
                    sess.run(self._init_train_op, dict(list(zip(self._input_vars, batch))))
                    self._init_train_op = None  # only use it once
                else:
                    sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))

                if self._verbose:
                    progbar.update(len(batch[0]))

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss
