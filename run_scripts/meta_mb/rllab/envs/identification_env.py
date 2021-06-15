from meta_mb.rllab.core.serializable import Serializable
from meta_mb.rllab.envs.proxy_env import ProxyEnv
from meta_mb.rllab.misc.overrides import overrides


class IdentificationEnv(ProxyEnv, Serializable):

    def __init__(self, mdp_cls, mdp_args):
        Serializable.quick_init(self, locals())
        self.mdp_cls = mdp_cls
        self.mdp_args = dict(mdp_args)
        self.mdp_args["template_args"] = dict(noise=True)
        mdp = self.gen_mdp()
        super(IdentificationEnv, self).__init__(mdp)

    def gen_mdp(self):
        return self.mdp_cls(**self.mdp_args)

    @overrides
    def reset(self):
        if getattr(self, "_mdp", None):
            if hasattr(self._wrapped_env, "release"):
                self._wrapped_env.release()
        self._wrapped_env = self.gen_mdp()
        return super(IdentificationEnv, self).reset()
