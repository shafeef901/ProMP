from meta_mb.sandbox_maml.rocky.tf.algos.trpo import TRPO
from meta_mb.rllab_maml.baselines.linear_feature_baseline import LinearFeatureBaseline
from meta_mb.rllab_maml.envs.box2d.cartpole_env import CartpoleEnv
from meta_mb.rllab_maml.envs.normalized_env import normalize
from meta_mb.sandbox_maml.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from meta_mb.sandbox_maml.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from meta_mb.sandbox_maml.rocky.tf.envs.base import TfEnv
import meta_mb.sandbox_maml.rocky.tf.core.layers as L
from meta_mb.sandbox_maml.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from meta_mb.rllab_maml.misc.instrument import stub, run_experiment_lite

stub(globals())

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianLSTMPolicy(
    name="policy",
    env_spec=env.spec,
    lstm_layer_cls=L.TfBasicLSTMLayer,
    # gru_layer_cls=L.GRULayer,
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)
