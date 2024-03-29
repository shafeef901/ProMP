from meta_mb.rllab_maml.algos.npo import NPO
from meta_mb.rllab_maml.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from meta_mb.rllab_maml.misc import ext


class TNPG(NPO):
    """
    Truncated Natural Policy Gradient.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TNPG, self).__init__(optimizer=optimizer, **kwargs)
