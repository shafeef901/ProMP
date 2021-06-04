import os
import time

EXP_NAME = 'grbal'

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

idx = int(time.time())
exp_dir = '{}/data/{}/run_{}'.format(meta_policy_search_path, EXP_NAME, idx)

print(exp_dir)