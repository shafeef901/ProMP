import mujoco_py
import time
import joblib
import tensorflow as tf


sess = tf.compat.v1.Session()

with sess.as_default() as sess:

	# load dumped files
	filename = "../data/pro-mp/run_1621852426/params.pkl"
	params = joblib.load(filename)
	env, policy, baseline = params['env'], params['policy'], params['baseline']

	# viewer setup
	viewer = mujoco_py.MjViewer(env.sim)
	viewer._render_every_frame = True

	# environment simulation
	obs = env.reset()
	while True:
		action, agent_infos = policy.get_action_render(obs)

		obs, reward, done, env_info = env.step(action)

		time.sleep(0.01)
		viewer.render()
