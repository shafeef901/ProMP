print("TRPO")
from meta_mb.sandbox.ours.algos.ModelTRPO.model_npo import ModelNPO
print("TRPO")
from meta_mb.sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
print("TRPO")


class ModelTRPO(ModelNPO):
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
        super(ModelTRPO, self).__init__(optimizer=optimizer, **kwargs)
