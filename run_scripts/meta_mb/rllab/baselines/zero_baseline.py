import numpy as np
from meta_mb.rllab.baselines.base import Baseline
from meta_mb.rllab.misc.overrides import overrides


class ZeroBaseline(Baseline):

    def __init__(self, env_spec):
        pass

    @overrides
    def get_param_values(self, **kwargs):
        return None

    @overrides
    def set_param_values(self, val, **kwargs):
        pass

    @overrides
    def fit(self, paths):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path["rewards"])
