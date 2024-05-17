import random

import numpy as np
import torch as T
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import Compose, Grayscale, ToTensor


def seed_worker(worker_id):
    worker_seed = T.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


NUM_WORKERS = 0

BATCH = 256
MNIST_INPUT = 784
CIFAR10_GRAY_INPUT = 1024

OUTPUT = 10
CIFAR100_OUTPUT = 100

cifar10_alexnet_transform_train = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_alexnet_transform_test = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR100_DEFAULT_TRAIN_TRANFORM = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

CIFAR100_DEFAULT_TEST_TRANFORM = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

CIFAR100_DEFAULT_TRAIN_TRANFORM = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

CIFAR100_DEFAULT_TEST_TRANFORM = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

FASHION_MNIST_TRANSFORM = Compose([ToTensor()])

DATASET_LOADERS = {
    'mlp': {
        'mnist': lambda config: get_mnist_c(config),
        'fashion_mnist': lambda config: get_fmnist_c(config),
        'cifar10': lambda config: get_cifar10gray_c(config)
    },
    'cnn': {
        'mnist': lambda config: get_mnist_c(config),
        'fashion_mnist': lambda config: get_fmnist_c(config),
        'cifar10': lambda config: get_cifar10_c(config),
        'cifar100': lambda config: get_cifar100_c(config)
    },
    'alexnet': {
        'mnist':
        lambda config: get_mnist_c(config),
        'fashion_mnist':
        lambda config: get_fmnist_c(config),
        'cifar10':
        lambda config: get_cifar10_c(config, cifar10_alexnet_transform_train,
                                     cifar10_alexnet_transform_test),
        'cifar100':
        lambda config: get_cifar100_c(config)
    },
    'vgg': {
        'mnist': lambda config: get_mnist_c(config),
        'fashion_mnist': lambda config: get_fmnist_c(config),
        'cifar10': lambda config: get_cifar10_c(config),
        'cifar100': lambda config: get_cifar100_c(config)
    },
}


def get_cifar10_dl(batch_size=32,
                   train_transforms=cifar10_transform_train,
                   test_transforms=cifar10_transform_test,
                   num_workers=3,
                   download=True):

    GENERATOR = T.Generator()
    GENERATOR.manual_seed(0)

    trainset = torchvision.datasets.CIFAR10(
        root='./data/.',
        train=True,
        download=download,
        transform=train_transforms,
    )

    trainloader = T.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=GENERATOR,
    )

    if test_transforms is None:
        test_transforms = train_transforms

    testset = torchvision.datasets.CIFAR10(
        root='./data/.',
        train=False,
        download=download,
        transform=test_transforms,
    )

    testloader = T.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=GENERATOR,
    )

    return trainloader, testloader


def get_mnist(batch_size,
              transforms=Compose(
                  [
                      ToTensor(),
                  ]), num_workers=4):

    GENERATOR = T.Generator()
    GENERATOR.manual_seed(0)

    train_mnist = MNIST('./data/',
                        transform=transforms,
                        train=True,
                        download=True)
    test_mnist = MNIST('./data/',
                       transform=transforms,
                       train=False,
                       download=True)

    train_mnist = DataLoader(train_mnist,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             worker_init_fn=seed_worker,
                             generator=GENERATOR)

    test_mnist = DataLoader(test_mnist,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            worker_init_fn=seed_worker,
                            generator=GENERATOR)

    return train_mnist, test_mnist


def get_cifar100_dl(batch_size=32,
                    train_transforms=CIFAR100_DEFAULT_TRAIN_TRANFORM,
                    test_transforms=CIFAR100_DEFAULT_TEST_TRANFORM,
                    manual_seed=None,
                    num_workers=4):

    worker_init_fn = None
    if manual_seed is not None:
        worker_init_fn = np.random.seed(manual_seed)
        num_workers = 0

    trainset = torchvision.datasets.CIFAR100(root='./data/',
                                             train=True,
                                             download=True,
                                             transform=train_transforms)

    trainloader = T.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          worker_init_fn=worker_init_fn)
    if test_transforms is None:
        test_transforms = train_transforms

    testset = torchvision.datasets.CIFAR100(root='./data/',
                                            train=False,
                                            download=True,
                                            transform=test_transforms)
    testloader = T.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         worker_init_fn=worker_init_fn)

    return trainloader, testloader


def get_fashion_mnist(batch_size,
                      transforms=FASHION_MNIST_TRANSFORM,
                      num_workers=4):

    GENERATOR = T.Generator()
    GENERATOR.manual_seed(1)

    train_mnist = FashionMNIST(
        './data/',
        transform=transforms,
        train=True,
        download=True,
    )

    test_mnist = FashionMNIST(
        './data/',
        transform=transforms,
        train=False,
        download=True,
    )

    train_dl = DataLoader(train_mnist,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          worker_init_fn=seed_worker,
                          generator=GENERATOR)

    test_dl = DataLoader(test_mnist,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         worker_init_fn=seed_worker,
                         generator=GENERATOR)

    return train_dl, test_dl


def get_mnist_c(config):
    batch = config['batch']
    if batch is None:
        batch = BATCH

    train_dl, test_dl = get_mnist(batch,
                                  transforms=Compose([
                                      ToTensor(),
                                  ]),
                                  num_workers=NUM_WORKERS)

    config['input'] = MNIST_INPUT
    config['output'] = OUTPUT

    return train_dl, test_dl


def get_fmnist_c(config):
    batch = config['batch']
    if batch is None:
        batch = BATCH

    train_dl, test_dl = get_fashion_mnist(batch,
                                          transforms=FASHION_MNIST_TRANSFORM,
                                          num_workers=NUM_WORKERS)

    config['input'] = MNIST_INPUT
    config['output'] = OUTPUT

    return train_dl, test_dl


def get_cifar10gray_c(config):
    batch = config['batch']
    if batch is None:
        batch = BATCH

    train_dl, test_dl = get_cifar10_dl(batch,
                                       train_transforms=Compose([
                                           Grayscale(),
                                           ToTensor(),
                                       ]),
                                       test_transforms=Compose([
                                           Grayscale(),
                                           ToTensor(),
                                       ]),
                                       num_workers=NUM_WORKERS)

    config['input'] = CIFAR10_GRAY_INPUT
    config['output'] = OUTPUT

    return train_dl, test_dl


def get_cifar10_c(config,
                  train_transform=cifar10_transform_train,
                  test_transform=cifar10_transform_test):
    batch = config['batch']
    if batch is None:
        batch = BATCH

    train_dl, test_dl = get_cifar10_dl(batch,
                                       train_transforms=train_transform,
                                       test_transforms=test_transform,
                                       num_workers=NUM_WORKERS)

    config['input'] = CIFAR10_GRAY_INPUT
    config['output'] = OUTPUT

    return train_dl, test_dl


def get_cifar100_c(config):
    batch = config['batch']
    if batch is None:
        batch = BATCH

    train_dl, test_dl = get_cifar100_dl(batch, num_workers=NUM_WORKERS)

    config['output'] = 100

    return train_dl, test_dl
