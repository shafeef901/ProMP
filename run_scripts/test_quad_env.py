from rand_param_envs import gym

gym.envs.register(
    id='Metalhead-v1',
    entry_point='rand_param_envs.gym.envs.mujoco.metalhead_v1:MetalheadEnvV1',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym.envs.register(
    id='Metalhead-v3',
    entry_point='rand_param_envs.gym.envs.mujoco.metalhead_v3:MetalheadEnvV3',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)


gym.envs.register(
    id='FullCheetah-v1',
    entry_point='rand_param_envs.gym.envs.mujoco.full_cheetah_v1:FullCheetahEnvV1',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
env = gym.make('Metalhead-v1')
env.reset()

while True:

	env.render()
