from meta_mb.sandbox.rocky.tf.algos.vpg import VPG
from meta_mb.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from meta_mb.rllab.envs.box2d.cartpole_env import CartpoleEnv
from meta_mb.rllab.envs.normalized_env import normalize
from meta_mb.sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.sandbox.rocky.tf.envs.base import TfEnv
from meta_mb.rllab.misc.instrument import stub, run_experiment_lite

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    optimizer_args=dict(
        tf_optimizer_args=dict(
            learning_rate=0.01,
        )
    )
)
algo.train()
