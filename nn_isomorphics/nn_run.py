import itertools
import logging

import torch as T
from tabulate import tabulate

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

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

NN_OPTIMIZER = {
    'nnsgd':
    lambda model, config: NNSGD(model.parameters(),
                                u_func=M_ABS(),
                                lr_in=config['lr_in'],
                                lr_out=config['lr_out']),
    'csgd':
    lambda model, config: CustomSGD(
        model.parameters(), u_func=N_Clip(), lr=config['lr_nn'])
}


def post_train_transformation(architecture, dataset, activation, _):
    '''
    Trains a model using SGD and then is transformed to its non-negative isomorphic,
    compering their evaluation accuracy.

    :param architecture: The applied architecture among ['mlp', 'cnn']
    :param dataset: The used dataset among ['mnist', 'fashion_mnist', 'cifar10']
    :param activation: The employed activation function among ['photonic_sigmoid', 'photonic_sinusoidal']
    :param _: Receives the optimizer (ignored)
    :return: The evaluation accuracy of the non-negative isomorphic model
    '''
    logging.info(
        '==================== Non-negative Transformation ===================='
    )

    logging.info(
        '==================== {} - {} - {} ===================='.format(
            architecture, dataset, activation))

    config = get_transformation_config(architecture, dataset, activation)
    train_dl, test_dl = DATASET_LOADERS[architecture][dataset](config)
    model = get_model(config, architecture, dataset)

    runner = MulticlassRunner(criterion=T.nn.CrossEntropyLoss())

    model.train()

    optimizer = T.optim.SGD(model.parameters(), lr=config['lr'])

    runner.fit(model, optimizer, train_dl, config['epochs'], verbose=0)

    nn_model = nn_transformation(model)

    nn_model.eval()
    model.eval()

    _, eval_acc = runner.eval(model, test_dl, verbose=0)
    _, nn_eval_acc = runner.eval(nn_model, test_dl, verbose=0)

    logging.info(
        'Regular Accuracy: {} | Non-negative Isomorphic Accuracy: {} '.format(
            eval_acc, nn_eval_acc))

    match = (nn_eval_acc == nn_eval_acc)

    return match, eval_acc, nn_eval_acc


def non_negative_post_training(architecture, dataset, activation,
                               str_optimizer):
    '''
    Trains a model using SGD for T_r epoch and then it trains the non-negative
    isomorphic model for another T_nn epoch in a non-negative manner using
    a non-negative optimizer.

    :param architecture: The applied architecture among ['mlp', 'cnn']
    :param dataset: The used dataset among ['mnist', 'fashion_mnist', 'cifar10']
    :param activation: The employed activation function among ['photonic_sigmoid', 'photonic_sinusoidal']
    :param str_optimizer: The applied non-negative optimizer among ['csgd', 'nnsgd']
    :return: The evaluation accuracy of the non-negative isomorphic model
    '''

    logging.info(
        '==================== Non-negative Post Training ====================')
    logging.info(
        '==================== {} - {} - {} - {} ===================='.format(
            architecture, dataset, activation, str_optimizer))

    config = get_nn_post_config(architecture, dataset, activation,
                                str_optimizer)

    train_dl, test_dl = DATASET_LOADERS[architecture][dataset](config)

    runner = MulticlassRunner(criterion=T.nn.CrossEntropyLoss(), )

    model = get_model(config, architecture, dataset)

    optimizer = T.optim.SGD(model.parameters(), lr=config['lr'])

    model.train()
    model.to(DEVICE)

    for _ in range(config['T_r']):
        runner.run_epoch(model, optimizer, train_dl)

    nn_model = nn_transformation(model)

    model.eval()
    nn_model.eval()
    _, half_accuracy = runner.eval(model, test_dl, verbose=0)
    _, nn_half_accuracy = runner.eval(nn_model, test_dl, verbose=0)

    match = (half_accuracy == nn_half_accuracy)

    nn_optimizer = NN_OPTIMIZER[str_optimizer](nn_model, config)

    nn_model.train()
    for _ in range(config['T_nn']):
        runner.run_epoch(nn_model, nn_optimizer, train_dl)

    _, nn_eval_acc = runner.eval(nn_model, test_dl, verbose=0)

    logging.info(
        'Non-negative Post Training Accuracy: {} '.format(nn_eval_acc))

    return match, None, nn_eval_acc


def fully_non_negative_training(architecture, dataset, activation, optimizer):
    '''
    A non-negative training from scratch after obtaining the non-negative isomorphic
    of a model.

    :param architecture: The applied architecture among ['mlp', 'cnn']
    :param dataset: The used dataset among ['mnist', 'fashion_mnist', 'cifar10']
    :param activation: The employed activation function among ['photonic_sigmoid', 'photonic_sinusoidal']
    :param str_optimizer: The applied non-negative optimizer among ['csgd', 'nnsgd']
    :return: The evaluation accuracy of the non-negative isomorphic model
    '''

    logging.info(
        '==================== Fully Non-negative Training ===================='
    )
    logging.info(
        '==================== {} - {} - {} - {} ===================='.format(
            architecture, dataset, activation, optimizer))

    config = get_fully_nn_config(architecture, dataset, activation, optimizer)

    train_dl, test_dl = DATASET_LOADERS[architecture][dataset](config)

    runner = MulticlassRunner(criterion=T.nn.NLLLoss())

    model = get_model(config, architecture, dataset)

    nn_model = nn_transformation(model)

    nn_model.eval()
    model.eval()
    _, half_accuracy = runner.eval(model, test_dl, verbose=0)
    _, nn_half_accuracy = runner.eval(nn_model, test_dl, verbose=0)

    match = (half_accuracy == nn_half_accuracy)

    nn_optimizer = NN_OPTIMIZER[optimizer](nn_model, config)

    nn_model.train()

    runner.fit(nn_model, nn_optimizer, train_dl, config['epochs'], verbose=0)

    nn_model.eval()

    _, nn_eval_acc = runner.eval(nn_model, test_dl, verbose=0)

    logging.info(
        "Fully Non-negative Training Accuracy: {}".format(nn_eval_acc))

    return match, None, nn_eval_acc


def add_to_results(results, method, architecture, dataset, activation,
                   optimizer, match, r_acc, nn_acc):
    results['method'].append(method)
    results['architecture'].append(architecture)
    results['dataset'].append(dataset)
    results['activation'].append(activation)
    results['nn_optimizer'].append(optimizer)
    results['nn_match'].append('✓' if match else '✘')
    results['nn_acc'].append(nn_acc)
    results['r_acc'].append(r_acc)


if __name__ == '__main__':
    args = get_args()

    results = {
        'method': [],
        'architecture': [],
        'dataset': [],
        'activation': [],
        'nn_optimizer': [],
        'nn_match': [],
        'nn_acc': [],
        'r_acc': [],
    }

    experiments = itertools.product(args.architecture, args.dataset,
                                    args.activation, args.nn_optimizer)

    print('Running experiments...')
    for exp_param in experiments:
        if args.transformation:
            match, r_acc, nn_acc = post_train_transformation(*exp_param)
            add_to_results(results, 'transformation', *exp_param, match, r_acc,
                           nn_acc)

        if args.post_train_nn:
            match, r_acc, nn_acc = non_negative_post_training(*exp_param)
            add_to_results(results, 'post_train_nn', *exp_param, match, r_acc,
                           nn_acc)

        if args.fully_negative_training:
            match, r_acc, nn_acc = fully_non_negative_training(*exp_param)
            add_to_results(results, 'fully_nn_train', *exp_param, match, r_acc,
                           nn_acc)

    print(tabulate(results, headers="keys", tablefmt="github"))
