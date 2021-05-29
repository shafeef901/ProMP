import mujoco_py
import time
import joblib
import tensorflow as tf


sess = tf.compat.v1.Session()

with sess.as_default() as sess:

	# load dumped files
	filename = "../data/pro-mp/AntRandDirecEnv/run_1622210986/params.pkl"
	params = joblib.load(filename)
	env, policy, baseline = params['env'], params['policy'], params['baseline']

	# setting task to render and test
	# env.set_task(1.0)

	print(env.get_task())

	# mujoco viewer setup
	viewer = mujoco_py.MjViewer(env.sim)
	viewer._render_every_frame = True

	# environment simulation
	obs = env.reset()
	policy.switch_to_pre_update()  # Switch to pre-update policy
	while True:
		action, agent_infos = policy.get_action_render(obs)

		obs, reward, done, env_info = env.step(action)
		print(reward)

		time.sleep(0.01)
		viewer.render()
