from meta_mb.rllab.viskit import frontend
from meta_mb.rllab.viskit import core
import matplotlib.pyplot as plt
from meta_mb.sandbox.dennis.plots.plot_utils_ppo import *

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
data_path = "data/all_model_comparison"
###########################################
exps_data = core.load_exps_data([data_path], False)



def sorting_legend(label):
    return LEGEND_ORDER[label]


def get_color(label):
    if label not in COLORS.keys():
        COLORS[label] = colors.pop(0)
    return COLORS[label]


def plot_from_exps(exp_data,
                   filters={},
                   split_figures_by=None,
                   split_plots_by=None,
                   x_key='Iteration',
                   y_key=None,
                   plot_name='./env-comparison',
                   subfigure_titles=None,
                   plot_labels=None,
                   x_label=None,
                   y_label=None,
                   ):

    exp_data = filter(exp_data, filters=filters)
    exps_per_plot = group_by(exp_data, group_by_key=split_figures_by)
    fig, axarr = plt.subplots(len(exps_per_plot.keys()), 1, figsize=(30, 40))
    fig.tight_layout(pad=4., w_pad=1.5, h_pad=3.0, rect=[0, 0, 1, 1])

    # x axis formatter
    xfmt = matplotlib.ticker.ScalarFormatter()
    # xfmt.set_powerlimits((3, 3))

    # iterate over subfigures
    for i, (default_plot_title, plot_exps) in enumerate(sorted(exps_per_plot.items())):      
      plots_in_figure_exps = group_by(plot_exps, split_plots_by)
      subfigure_title = subfigure_titles[i] if subfigure_titles else default_plot_title
      axarr[i].set_title(subfigure_title)
      axarr[i].xaxis.set_major_formatter(xfmt)
      axarr[i].xaxis.set_major_locator(plt.MaxNLocator(5))

      # iterate over plots in figure
      for j, default_label in enumerate(sorted(plots_in_figure_exps, key=sorting_legend)):
          exps = plots_in_figure_exps[default_label]
          
          x, y_mean, y_std = prepare_data_for_plot(exps, x_key=x_key, y_key=y_key)
          label = plot_labels[j] if plot_labels else default_label
          _label = label
          axarr[i].plot(x, y_mean, label=_label,  linewidth=LINEWIDTH, color=get_color(label))
          axarr[i].fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.2, color=get_color(label))

          # axis labels
          axarr[i].set_xlabel(x_label if x_label else x_key)
          axarr[i].set_ylabel(y_label if y_label else y_key)
    handles, labels = axarr[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_transform=plt.gcf().transFigure)
    fig.savefig(plot_name + '.png')


plot_from_exps(exps_data,
               split_figures_by='env.$class',
               split_plots_by='exp_name',
               y_key='1AverageReturn',
               subfigure_titles=['AntRandDir', 'AntRandGoal', 'HalfCheetahRandVel', 'HalfCheetahRandDir', 'SwimmerRandGoal', 'HopperEnvRandParams', 'WalkerEnvRandParams'],
               x_label='Time steps',
               y_label='Average return',
               )