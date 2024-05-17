import itertools
import logging
import os

import torch as T
import wandb
from tabulate import tabulate

from nn_isomorphics.networks.alexnet import AlexNet
from nn_isomorphics.networks.nn_resnet import ALL_RESNETS
from nn_isomorphics.networks.vgg import VGG
from nn_isomorphics.nnn.tranformation import nn_transformation
from nn_isomorphics.utils.activations import GELUN, ReLUN
from nn_isomorphics.utils.imagenet_configuriations import (
    MAX_BOUNDS, get_finetuning_config)
from nn_isomorphics.utils.imagenet_parser import get_imagenet_args
from nn_isomorphics.utils.imagenet_runner import evaluate
from nn_isomorphics.utils.loaders import get_imagenet_val
from nn_isomorphics.utils.utils import ifnot_create

WANDB_DIR = os.path.join(os.path.abspath(os.curdir),
                         ifnot_create('exports/wandb/'))

PRETRAINED_MODELS = {
    'vgg11': 'vgg/vgg11-8a719046.pth',
    'vgg13': 'vgg//vgg13-19584684.pth',
    'vgg16': 'vgg/vgg16-397923af.pth',
    'vgg19': 'vgg/vgg19-dcbb9e9d.pth',
    'alexnet': 'alexnet/alexnet-owt-7be5be79.pth',
    'resnet18': 'resnet/resnet18-f37072fd.pth',
    'resnet34': 'resnet/resnet34-b627a593.pth',
    'resnet50': 'resnet/resnet50-11ad3fa6.pth',
    'resnet101': 'resnet/resnet101-cd907fc2.pth',
    'resnet152': 'resnet/resnet152-f82ba261.pth',
}


def get_model(model_name, activation_str):

    if activation_str == 'relu':
        bound = MAX_BOUNDS[model_name]
        activation_cls = T.nn.ReLU
    elif 'relu' in activation_str:
        bound = int(activation_str[4:])
        activation_cls = lambda: ReLUN(bound)
    elif 'gelu' == activation_str:
        bound = MAX_BOUNDS[model_name]
        activation_cls = T.nn.GELU
    elif 'gelu' in activation_str:
        bound = int(activation_str[4:])
        activation_cls = lambda: GELUN(n=bound)
    else:
        raise Exception('The introduced activation function is not supported')

    if model_name == 'alexnet':
        model = AlexNet(
            alpha=bound,
            act_func_cls=activation_cls,
        )
    elif 'vgg' in model_name:
        model = VGG(
            model_name,
            activation_cls,
            alpha=bound,
        )
    elif 'resnext' in model_name:
        pass
    elif 'wide_resnet' in model_name:
        pass
    elif 'resnet' in model_name:
        model = ALL_RESNETS[model_name](activation_cls, None, bound)
    else:
        raise Exception('This models is not supported yet')
    return model


def neat_transformation(architecture, dataset, activation_str, use_wandb):
    logging.info('==================== Transformation ====================')

    logging.info(
        '==================== {} - {} - {} ===================='.format(
            architecture, dataset, activation_str))

    config = get_finetuning_config(dataset)
    config['activation'] = activation_str
    config['architecture'] = architecture

    if use_wandb is not None:
        (entity, project) = use_wandb.split('/')
        wandb.init(project=project,
                   name=None,
                   dir=WANDB_DIR,
                   group='NIPS24',
                   entity=entity,
                   config=config)

    model = get_model(architecture, activation_str)

    model.load_state_dict(T.load('models/' + PRETRAINED_MODELS[architecture]))

    val_dl = get_imagenet_val(config['batch_size'])

    criterion = T.nn.CrossEntropyLoss(
        label_smoothing=config['label_smoothing'])

    r_acc1, r_acc5, r_loss = evaluate(model,
                                      criterion,
                                      val_dl,
                                      config,
                                      log_suffix='Regular |')

    print(f"Accuracy@1:{r_acc1:.2f} & Accuracy@5:{r_acc5:.2f}")

    nn_model = nn_transformation(model)

    nn_acc1, nn_acc5, nn_loss = evaluate(nn_model,
                                         criterion,
                                         val_dl,
                                         config,
                                         log_suffix='Non Negative |')
    print(f"NN Accuracy@1:{nn_acc1:.2f} & Accuracy@5:{nn_acc5:.2f}")

    if use_wandb is not None:
        wandb.log({'r_acc1': r_acc1, 'r_acc5': r_acc5})
        wandb.log({'nn_acc1': nn_acc1, 'nn_acc5': nn_acc5})
        wandb.finish()

    return (r_acc1, r_acc5, nn_acc1, nn_acc5)


def add_to_results(acc_results, architecture, dataset, activation, results):
    acc_results['architecture'].append(architecture)
    acc_results['dataset'].append(dataset)
    acc_results['activation'].append(activation)
    acc_results['r_acc1'].append(results[0])
    acc_results['r_acc5'].append(results[1])
    acc_results['nn_acc1'].append(results[2])
    acc_results['nn_acc5'].append(results[3])


def main() -> None:
    acc_results = {
        'architecture': [],
        'dataset': [],
        'activation': [],
        'r_acc1': [],
        'r_acc5': [],
        'nn_acc1': [],
        'nn_acc5': [],
    }

    args = get_imagenet_args()

    experiments = itertools.product(args.architecture, args.dataset,
                                    args.activation)
    for exp_param in experiments:
        results = neat_transformation(*exp_param, args.wandb)
        add_to_results(
            acc_results,
            *exp_param,
            results,
        )

    print(tabulate(acc_results, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
