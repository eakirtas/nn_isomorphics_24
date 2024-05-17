from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as T
from matplotlib import ticker
from nn_isomorphics.networks import get_model
from nn_isomorphics.nn_optim.custom_sgd import CustomSGD
from nn_isomorphics.nn_optim.nnsgd import NNSGD
from nn_isomorphics.nn_optim.updates import M_ABS, N_Clip
from nn_isomorphics.nnn.tranformation import nn_transformation
from nn_isomorphics.utils.configurations import (get_fully_nn_config,
                                                 get_nn_post_config,
                                                 get_transformation_config)
from nn_isomorphics.utils.datasets import DATASET_LOADERS
from nn_isomorphics.utils.parser import get_args
from nn_isomorphics.utils.runner import MulticlassRunner
from torchvision.transforms import Compose, ToTensor

plt.style.use(['science', 'ieee'])

MNIST_SIZES = [200, 100]
MNIST_BATCH = 256
FMNIST_TRANSFORM = Compose([ToTensor()])
PH_SINUSOIDAL_ALPHA = [1.0, 1.0, 1.0]
MNIST_INPUT = 784
MNIST_OUTPUT = 10
MNIST_EPOCHS = 10
ROOT_EXPORT = './exports/'
MODELS_PATH = ROOT_EXPORT + '/models//model.pt'
PREVIOUS_PATH = './exports/01_fcnn_regular'
DATASET = 'fashion_mnist'
ACTIVATION = 'photonic_sinusoidal'

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class FixedOrderFormatter(ticker.ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude"""

    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ticker.ScalarFormatter.__init__(self,
                                        useOffset=useOffset,
                                        useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag


def get_best_lr():
    exp_dir = '{}/{}/{}/tune/tr=-1/regular_{}'.format(PREVIOUS_PATH, DATASET,
                                                      ACTIVATION, 'simple_mlp')
    analysis = tune.Analysis(exp_dir)
    previous_best_config = analysis.get_best_config(metric='train_acc',
                                                    mode='max')
    return previous_best_config['lr']


def init_biases(m):
    if isinstance(m, T.nn.Linear):
        T.nn.init.constant_(m.bias.data, 0)


def train(architecture, dataset, activation, _):

    config = get_transformation_config(architecture, dataset, activation)

    train_dl, test_dl = DATASET_LOADERS[architecture][dataset](config)

    runner = MulticlassRunner(criterion=T.nn.NLLLoss())

    model = get_model(config, architecture, dataset)

    model.apply(init_biases)
    model.train()

    optimizer = T.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)

    runner.fit(model, optimizer, train_dl, config['epochs'], verbose=0)

    T.save(model.state_dict(), MODELS_PATH)

    nn_model = nn_transformation(model)

    nn_model.eval()
    model.eval()
    _, eval_acc = runner.eval(model, test_dl, verbose=0)
    _, nn_eval_acc = runner.eval(nn_model, test_dl, verbose=0)

    print(
        'Regular Accuracy: {} | Non-negative Isomorphic Accuracy: {} '.format(
            eval_acc, nn_eval_acc))

    match = (nn_eval_acc == nn_eval_acc)

    return model, nn_model


def get_models(architecture, dataset, activation):
    config = get_transformation_config(architecture, dataset, activation)
    train_dl, test_dl = DATASET_LOADERS[architecture][dataset](config)
    model = get_model(config, architecture, dataset)

    model.load_state_dict(T.load(MODELS_PATH))
    model.eval()

    nn_model = nn_transformation(model)
    nn_model.eval()

    return model, nn_model, test_dl


def get_regular_visual_dict(model, x_1, x1_act, x_2, x2_act, x_3, x3_act, y):
    ordr_dict = OrderedDict()

    ordr_dict['x_1'] = x_1.detach().flatten().cpu().numpy()
    ordr_dict['x1_act'] = x1_act.detach().flatten().cpu().numpy()
    ordr_dict['w_1'] = model.layer_1.weight.detach().flatten().cpu().numpy()
    ordr_dict['b_1'] = model.layer_1.bias.detach().flatten().cpu().numpy()
    ordr_dict['x2'] = x_2.detach().flatten().cpu().numpy()
    ordr_dict['x2_act'] = x2_act.detach().flatten().cpu().numpy()
    ordr_dict['w_2'] = model.layer_2.weight.detach().flatten().cpu().numpy()
    ordr_dict['b_2'] = model.layer_2.bias.detach().flatten().cpu().numpy()
    ordr_dict['x3'] = x_3.detach().flatten().cpu().numpy()
    ordr_dict['x3_act'] = x3_act.detach().flatten().cpu().numpy()
    ordr_dict['w_3'] = model.layer_3.weight.detach().flatten().cpu().numpy()
    ordr_dict['b_3'] = model.layer_3.bias.detach().flatten().cpu().numpy()
    ordr_dict['y'] = y.detach().flatten().cpu().numpy()

    return ordr_dict


def get_nn_visual_dict(model, x_1, x1_act, x_2, x2_act, x_3, x3_act, y):
    ordr_dict = OrderedDict()

    ordr_dict['x_1'] = x_1.detach().flatten().cpu().numpy()
    ordr_dict['x1_act'] = x1_act.detach().flatten().cpu().numpy(
    ) + model.layers[0].activation_shift.item()
    ordr_dict['w_1'] = model.layers[0].get_weight().detach().flatten().cpu(
    ).numpy()
    # ordr_dict['w_1'] = model.layers[0].l_neg.weight.detach().flatten().cpu(
    # ).numpy()
    ordr_dict['b_1'] = model.layers[0].bias.flatten().detach().cpu().numpy()
    ordr_dict['x2'] = x_2.detach().flatten().cpu().numpy()
    ordr_dict['x2_act'] = x2_act.detach().flatten().cpu().numpy(
    ) + model.layers[1].activation_shift.item()
    ordr_dict['w_2'] = model.layers[1].get_weight().detach().flatten().cpu(
    ).numpy()
    # ordr_dict['w_2'] = model.layers[1].l_neg.weight.detach().flatten().cpu(
    # ).numpy()
    ordr_dict['b_2'] = model.layers[1].bias.detach().flatten().cpu().numpy()
    ordr_dict['x3'] = x_3.detach().flatten().cpu().numpy()
    ordr_dict['x3_act'] = x3_act.detach().flatten().cpu().numpy(
    ) + model.layers[2].activation_shift.item()
    ordr_dict['w_3'] = model.layers[2].get_weight().detach().flatten().cpu(
    ).numpy()
    # ordr_dict['w_3'] = model.layers[2].l_neg.weight.detach().flatten().cpu(
    # ).numpy()

    ordr_dict['b_3'] = model.layers[2].bias.detach().flatten().cpu().numpy()
    ordr_dict['y'] = y.detach().flatten().cpu().numpy()

    return ordr_dict


def forward_model(model, eval_dl):
    model.to(DEVICE)

    rdm_batch = iter(eval_dl)

    x_1, target = next(rdm_batch)
    x_1, target = x_1.to(DEVICE), target.to(DEVICE)

    x_1 = T.flatten(x_1, start_dim=1)
    x1_act = model.layers[0](x_1)
    x_2 = model.acts[0](x1_act)

    x2_act = model.layers[1](x_2)
    x_3 = model.acts[1](x2_act)

    x3_act = model.layers[2](x_3)
    y = model.acts[2](x3_act)

    return model, x_1, x1_act, x_2, x2_act, x_3, x3_act, y


def plot_figures(regular_dict,
                 nn_dict,
                 path,
                 regular_color='red',
                 nn_color='green'):
    # plt.style.use(['science', 'ieee'])
    # sns.set_style("ticks", {'axes.grid': True})

    # sns.set(font_scale=2)
    fig = plt.figure(figsize=(8.27 / 2.0, 5.5))
    spec = fig.add_gridspec(ncols=2, nrows=7)

    titles = [
        ['Input Layer 1', 'Linear Output 1'],
        ['Weights Layer 1', 'Biases Layer 1'],
        ['Input Layer 2', 'Linear Output 2'],
        ['Weights Layer 2', 'Biases Layer 2'],
        ['Input Layer 3', 'Linear Output 3'],
        ['Weights Layer 3', 'Biases Layer 3'],
        ['Softmax Output'],
    ]

    for k, key in enumerate(regular_dict.keys()):
        plt.subplots_adjust(wspace=0.15, hspace=0.6)
        (i, j) = np.unravel_index(k, (7, 2))

        if key != 'y':
            ax = fig.add_subplot(spec[i, j])
        else:
            ax = fig.add_subplot(spec[i, :])
        # ax.xticks(fontsize=0.6)
        ax.tick_params(axis='both', which='major', labelsize=6)
        h1 = sns.histplot(
            x=regular_dict[key],
            bins=100,
            color=regular_color,
            edgecolor=regular_color,
            linewidth=0,
            alpha=0.7,
            ax=ax,
        )

        h1.set(ylabel=None)

        h2 = sns.histplot(
            x=nn_dict[key],
            bins=100,
            color=nn_color,
            edgecolor=nn_color,
            linewidth=0,
            alpha=0.7,
            ax=ax,
        )

        h2.set(ylabel=None)

        h2.axes.set_title(titles[i][j], fontsize=7, y=0.9)

        if k != 7 and k != 11:
            h1.set_yscale('log')
            h2.set_yscale('log')

        if k == 12:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.67, box.height])
            ax.legend(['Regular', 'Non-Negative'],
                      loc='center left',
                      bbox_to_anchor=(1, 0.5),
                      fontsize=7)

    # plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def visualize_models(model, nn_model, eval_dl):

    regular_viz_dict = get_regular_visual_dict(*forward_model(model, eval_dl))
    nn_viz_dict = get_nn_visual_dict(*forward_model(nn_model, eval_dl))

    plot_figures(regular_viz_dict,
                 nn_viz_dict,
                 path=ROOT_EXPORT + '/transformation_plot.pdf',
                 regular_color='#cf3759',
                 nn_color='#4771b2')


train('mlp', DATASET, ACTIVATION, '_')
visualize_models(*get_models('mlp', DATASET, ACTIVATION))
