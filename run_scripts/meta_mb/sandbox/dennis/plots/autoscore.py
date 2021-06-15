from meta_mb.rllab.viskit import frontend
from meta_mb.rllab.viskit import core
import matplotlib.pyplot as plt
from meta_mb.sandbox.dennis.plots.plot_utils_ppo import *
import numpy as np

plt.style.use('ggplot')
#plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.font_manager._rebuild()


SMALL_SIZE = 30
MEDIUM_SIZE = 32
BIGGER_SIZE = 36
LINEWIDTH = 4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
COLORS = dict(ours=colors.pop(0))

LEGEND_ORDER = {'ppo': 0, 'trpo': 1, 'vpg': 2}

########## Add data path here #############
data_path = "data/s3/ppo-maml-hyperparam-final-2-lmao"
###########################################
exps_data = core.load_exps_data([data_path], False)



def sorting_legend(label):
    return LEGEND_ORDER[label]


def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]

def total_under_curve(exp_data):
  mean_scores = []
  for exp in exp_data:
      mean_scores.append(np.mean(exp['progress']['1AverageReturn']))
  return np.mean(mean_scores)

def final_hundred(exp_data):
  mean_scores = []
  for exp in exp_data:
      mean_scores.append(np.mean(exp['progress']['1AverageReturn'][-100:]))
  return np.mean(mean_scores)

def total_minus_std(exp_data):
  mean_scores = []
  std_scores = []
  for exp in exp_data:
    mean_scores.append(np.mean(exp['progress']['1AverageReturn']))
    std_scores.append(np.mean(exp['progress']['1StdReturn']))
  return np.mean(mean_scores) - np.mean(std_scores)

def sample_time(exp_data):
  times = {}
  for exp in exp_data:
    for key, progress in exp['progress'].items():
      if 'Time' in key:
        if key not in times:
          times[key] = []
        times[key].append(np.mean(progress))
  for key in times:
    times[key] = np.mean(times[key])
  return times

def split_and_score(exp_data,
                    split_by=[],
                    score_by=total_under_curve
                    ):

    output = []
    def filter_dead(exp):
      if exp['progress']:
        return True
      return False

    def recursive_filter(exp_data,
                         split_by=[],
                         keys_so_far=[]
                        ):
        if split_by:
          remaining_exps = group_by(exp_data, split_by[0])
          for new_key, exps in remaining_exps.items():
            recursive_filter(exps, split_by[1:], keys_so_far + [new_key])
        else:
          output.append(('\t'.join(keys_so_far), score_by(exp_data)))
    exp_data = __builtins__.filter(lambda exp: exp['progress'], exp_data)
    recursive_filter(exp_data, split_by)
    output.sort(key=lambda x: x[-1])
    for group, score in output:
      print(group + '\t %f' % score )

split_and_score(exps_data,
                split_by=['env.$class', 'max_epochs', 'clip_eps', 'target_inner_step'],
                score_by=final_hundred,
                )