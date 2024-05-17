import torch as T

from .alexnet import AlexNet
from .mnist_cnn import MnistCNN
from .simple_cnn import SimpleCNN, SimpleCNNStride
from .simple_cnn_new import SimpleCNN_v2
from .simple_mlp import SimpleMLP


def kaiming_init(m):
    if isinstance(m, T.nn.Conv1d) or isinstance(m, T.nn.Conv2d) or isinstance(
            m, T.nn.Linear):
        T.nn.init.kaiming_normal_(m.weight)


@T.no_grad()
def xavier_init(m, gain=1):
    if isinstance(m, T.nn.Conv1d) or isinstance(m, T.nn.Conv2d) or isinstance(
            m, T.nn.Linear):
        T.nn.init.xavier_normal_(m.weight, gain=gain)


def get_model(config, architecture, dataset):
    if architecture == 'mlp':
        model = SimpleMLP(
            config['input'],
            config['output'],
            config['activation'],
            config['nn_activation'],
            config['alpha'],
            config['sizes'],
        )
    elif architecture == 'cnn':
        if dataset == 'cifar10':
            model = SimpleCNN_v2(
                config['output'],
                config['activation'],
                config['nn_activation'],
                config['alpha'],
            )
        else:
            model = MnistCNN(
                config['output'],
                config['activation'],
                config['nn_activation'],
                config['alpha'],
            )
    elif architecture == 'alexnet':
        model = AlexNet(
            num_classes=config['output'],
            alpha=config['alpha'],
            act_func_cls=config['activation'],
        )

    model.apply(xavier_init)

    return model
