DATASET_CONFIG = {
    'imagenet': {
        'batch_size': 32
    },
}

TRAIN_CONFIG = {
    'lr': 1e-4,
    'start_epoch': 0,
    'epochs': 30,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_step_size': 10,
    'lr_gamma': 0.1,
    'optimizer': 'sgd',
    'clip_grad_norm': None,
    'print_freq': 100,
    'label_smoothing': 0,
    'model_path': './exports/imagenets/fine_tune.pth'
}

MAX_BOUNDS = {
    'alexnet': 25,
    'vgg11': 25,
    'vgg13': 40,
    'vgg16': 40,
    'vgg19': 40,
    'resnet18': 25,
    'resnet34': 25,
    'resnet50': 100,
    'resnet101': 100,
    'resnet152': 100,
}


def get_finetuning_config(dataset):
    config = {}

    config.update(DATASET_CONFIG[dataset].copy())
    config.update(TRAIN_CONFIG.copy())

    return config
