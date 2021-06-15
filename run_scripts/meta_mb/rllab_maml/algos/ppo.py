from meta_mb.rllab_maml.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from meta_mb.rllab_maml.algos.npo import NPO
from meta_mb.rllab_maml.core.serializable import Serializable


class PPO(NPO, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
