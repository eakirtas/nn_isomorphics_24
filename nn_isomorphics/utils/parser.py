import argparse
import logging


def get_args():
    parser = argparse.ArgumentParser(
        description=
        "Runs experiments using Non Negative Isomorphic Neural Networks")

    parser.add_argument("--transformation",
                        "-t",
                        help='transform network to non-negative isomorphic',
                        default=False,
                        action='store_true')

    parser.add_argument("--post_train_nn",
                        "-p",
                        help='post-training in non-negative amnner',
                        default=False,
                        action='store_true')

    parser.add_argument("--fully_negative_training",
                        "-f",
                        help='non-negative training from scratch',
                        default=False,
                        action='store_true')

    parser.add_argument('--architecture',
                        '-c',
                        help='Employed architecture',
                        type=str,
                        choices=['mlp', 'cnn', 'alexnet'],
                        nargs='+')

    parser.add_argument('--nn_optimizer',
                        '-o',
                        help='optimization algorithm',
                        choices=['nnsgd', 'csgd'],
                        default=['-'],
                        type=str,
                        nargs='+')

    parser.add_argument('--dataset',
                        '-d',
                        help='dataset',
                        type=str,
                        choices=[
                            'mnist',
                            'fashion_mnist',
                            'cifar10',
                        ],
                        nargs='+')

    parser.add_argument('--activation',
                        '-a',
                        help='the employed activation',
                        type=str,
                        choices=[
                            'photonic_sigmoid',
                            'photonic_sinusoidal',
                            'relu',
                            'relu14',
                            'relu12',
                            'relu10',
                            'relu8',
                            'relu6',
                        ],
                        nargs='+')

    parser.add_argument('--noise_aware',
                        help='noise aware training',
                        default=False,
                        action='store_true')

    parser.add_argument('--noise_inference',
                        help='noise inference',
                        default=False,
                        action='store_true')

    parser.add_argument(
        '--log',
        help='logging level',
        choices=['info', 'debug', 'warning', 'error', 'critical'],
        type=str,
        default='WARNING',
    )

    args = parser.parse_args()

    logging.basicConfig(
        format=
        '[LOG-%(levelname)s][%(module)s:%(lineno)d] %(asctime)s -- %(message)s',
        handlers=[logging.StreamHandler()],
        level=getattr(logging, args.log.upper(), None),
    )

    return args
