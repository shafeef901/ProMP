

from meta_mb.sandbox.ours.algos.ModelMAML.model_maml_npo import ModelMAMLNPO
from meta_mb.sandbox_maml.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class ModelMAMLTRPO(ModelMAMLNPO):
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
        super(ModelMAMLTRPO, self).__init__(optimizer=optimizer, **kwargs)
