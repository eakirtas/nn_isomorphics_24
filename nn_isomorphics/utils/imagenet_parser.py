import argparse
import logging


def get_imagenet_args():
    parser = argparse.ArgumentParser(
        description=
        "Runs experiments using Non Negative Isomorphic Neural Networks")

    parser.add_argument('--activation',
                        '-a',
                        help='the employed activation',
                        type=str,
                        nargs='+')

    parser.add_argument('--architecture',
                        '-c',
                        help='employed architecture',
                        type=str,
                        nargs='+')

    parser.add_argument('--dataset',
                        '-d',
                        help='dataset',
                        type=str,
                        choices=[
                            'imagenet',
                            'cifar10',
                            'cifar100',
                            'mnist',
                            'fashion_mnist',
                        ],
                        default=['imagenet'],
                        nargs='+')

    parser.add_argument('--wandb',
                        '-w',
                        help='use wandb',
                        type=str,
                        default=None)

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
