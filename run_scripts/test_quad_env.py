from meta_policy_search.envs.mujoco_envs.metalhead_v1_rand_direc import MetalheadEnvV1RandDirec


env = MetalheadEnvV1RandDirec()
env.reset()

while True:
	env.step(env.action_space.sample())
	env.render()
