from meta_mb.rllab_maml.algos.npo import NPO
from meta_mb.rllab_maml.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from meta_mb.rllab_maml.core.serializable import Serializable


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)
