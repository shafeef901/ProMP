from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
import mujoco_py
import time
import joblib
import tensorflow as tf


sess = tf.compat.v1.Session()

with sess.as_default() as sess:

	# load trained tf model
	filename = "../data/pro-mp/run_1621852426/params.pkl"
	print("-- loading start --")
	params = joblib.load(filename)
	print("-- loading done --")
	
	print(params['policy'])
	print(params['env'])
	print(params['itr'])


	#load environment
	hf = HalfCheetahRandDirecEnv()


	# viewer = mujoco_py.MjViewer(hf.sim)
	# viewer._render_every_frame = True
	# hf.sim.forward()
	# time.sleep(0.01)
	# viewer.render()
