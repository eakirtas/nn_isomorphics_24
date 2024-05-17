import itertools

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
import wandb
from tabulate import tabulate
from torch.functional import _lu_impl

wandb_path = "<entity>/<project>"


def init_plts(x, y, figsize):
    plt.style.use(['science', 'ieee'])

    SMALL_SIZE = 4
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 8

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes

    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    return plt.subplots(x, y, figsize=figsize)


COLORS_PALETTE = {
    'alexnet': '#00429d',
    'vgg11': '#93003a',
    'vgg13': '#00ff00',
    'vgg16': '#ff005e',
    'vgg19': '#4ba1c4'
}


def plot_results(runs_df, architectures, l_bound):

    runs_df = runs_df.loc[runs_df['Architecture'].isin(architectures)]
    runs_df = runs_df.loc[runs_df['ReLU Bound'] > l_bound]

    fig, ax_l = init_plts(1, 1, figsize=(2.5, 0.9))

    grph_l = sns.lineplot(data=runs_df,
                          x='ReLU Bound',
                          y='NN Acc@1',
                          hue='Architecture',
                          marker='*',
                          markersize=10,
                          ax=ax_l,
                          palette=COLORS_PALETTE)
    grph_l.set(title='ImageNet1K Acc@1', ylabel='Accuracy(\%)')

    sns.move_legend(ax_l, "upper left", bbox_to_anchor=(1, 1))

    ax_l.grid()

    fig.savefig('./exports/cvpr_imagenet.pdf')


def viz_imagenet_results(baseline_act: str):
    runs = wandb.Api().runs("eakirtas/imagenet_nn_adapt")

    bench_rows = []
    base_rows = []

    for run in runs:
        summary_dict = run.summary._json_dict

        if "nn_acc1" in summary_dict and run.config[
                'activation'] != baseline_act:
            bench_rows.append({
                'Architecture': run.config['architecture'],
                'Activation': run.config['activation'],
                'ReLU Bound': int(run.config['activation'][4:]),
                'NN Acc@1': float(summary_dict['nn_acc1']),
                'NN Acc@5': float(summary_dict['nn_acc5']),
                'R Acc@1': float(summary_dict['r_acc1']),
                'R Acc@5': float(summary_dict['r_acc5']),
            })
        elif 'r_acc1' in summary_dict:
            base_rows.append({
                'Architecture': run.config['architecture'],
                'Activation': run.config['activation'],
                'ReLU Bound': None,
                'NN Acc@1': None,
                'NN Acc@5': None,
                'R Acc@1': float(summary_dict['r_acc1']),
                'R Acc@5': float(summary_dict['r_acc5']),
            })

    bench_df = pd.DataFrame(bench_rows)
    base_df = pd.DataFrame(base_rows)

    return bench_df, base_df


plot_results(*viz_imagenet_results(baseline_act='relu'),
             architectures=['alexnet', 'vgg11'],
             l_bound=6)
