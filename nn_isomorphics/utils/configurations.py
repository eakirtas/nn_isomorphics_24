import torch as T
from nn_isomorphics.utils.activations import (PhotonicSigmoid,
                                              PhotonicSinusoidal, ReLUN)

MAX_RELU_BOUND = 14
PH_SIGMOID_ALPHA = 1.005
PH_SINUSOIDAL_ALPHA = 1.0

ACTIVATIONS_CONFIG = {
    'photonic_sigmoid': {
        'activation': PhotonicSigmoid,
        'nn_activation': None,
        'alpha': PH_SIGMOID_ALPHA,
    },
    'photonic_sinusoidal': {
        'activation': PhotonicSinusoidal,
        'nn_activation': None,
        'alpha': PH_SINUSOIDAL_ALPHA
    },
    'relu': {
        'activation': T.nn.ReLU,
        'nn_activation': lambda: ReLUN(MAX_RELU_BOUND),
        'alpha': MAX_RELU_BOUND
    },
    'relu14': {
        'activation': lambda: ReLUN(14),
        'nn_activation': None,
        'alpha': 14
    },
    'relu12': {
        'activation': lambda: ReLUN(12),
        'nn_activation': None,
        'alpha': 12
    },
    'relu10': {
        'activation': lambda: ReLUN(10),
        'nn_activation': None,
        'alpha': 10
    },
    'relu8': {
        'activation': lambda: ReLUN(8),
        'nn_activation': None,
        'alpha': 8
    },
    'relu6': {
        'activation': lambda: ReLUN(6),
        'nn_activation': None,
        'alpha': 6
    },
    'relu4': {
        'activation': lambda: ReLUN(4),
        'nn_activation': None,
        'alpha': 4
    },
}

DATASET_CONFIG = {
    'mnist': {
        'batch': 256,
        'epochs': 50,
        'sizes': [100, 200],
        'T_r': 5,
        'T_nn': 5
    },
    'fashion_mnist': {
        'batch': 256,
        'epochs': 50,
        'sizes': [100, 200],
        'T_r': 5,
        'T_nn': 5
    },
    'cifar10': {
        'batch': 256,
        'epochs': 100,  # TODO: Changed for debug
        'sizes': [1024, 512],
        'T_r': 15,
        'T_nn': 15
    },
    'cifar100': {
        'batch': 256,
        'epochs': 60,
        'sizes': [1024, 512],
        'T_r': 30,
        'T_nn': 30
    },
}

TRANFORMATIONS_LR = {
    'mlp': {
        'photonic_sigmoid': {
            'mnist': {
                'lr': 0.092,
            },
            'fashion_mnist': {
                'lr': 0.01,
            },
            'cifar10': {
                'lr': 0.002,
            },
        },
        'photonic_sinusoidal': {
            'mnist': {
                'lr': 0.5,
            },
            'fashion_mnist': {
                'lr': 0.16,
            },
            'cifar10': {
                'lr': 0.047,
            }
        },
        'relu': {
            'cifar10': {
                'lr': 0.01,
            },
            'cifar100': {
                'lr': 0.01  # Default
            },
            'mnist': {
                'lr': 0.01,
            },
            'fashion_mnist': {
                'lr': 0.01,
            }
        }
    },
    'cnn': {
        'photonic_sigmoid': {
            'mnist': {
                'lr': 0.007,
            },
            'fashion_mnist': {
                'lr': 0.01,
            },
            'cifar10': {
                'lr': 0.026,
            },
            'cifar100': {
                'lr': 0.01  # Default
            }
        },
        'photonic_sinusoidal': {
            'mnist': {
                'lr': 0.5,
            },
            'fashion_mnist': {
                'lr': 0.056,
            },
            'cifar10': {
                'lr': 0.06,
            },
            'cifar100': {
                'lr': 0.01  # Default
            }
        },
        'relu': {
            'cifar10': {
                'lr': 0.01,
            },
            'cifar100': {
                'lr': 0.01  # Default
            },
            'mnist': {
                'lr': 0.01,
            },
            'fashion_mnist': {
                'lr': 0.01,
            }
        }
    },
    'alexnet': {  # TODO: Hyper parameter tuning
        'photonic_sigmoid': {
            'mnist': {
                'lr': 0.007,
            },
            'fashion_mnist': {
                'lr': 0.01,
            },
            'cifar10': {
                'lr': 0.026,
            },
            'cifar100': {
                'lr': 0.01  # Default
            }
        },
        'photonic_sinusoidal': {
            'mnist': {
                'lr': 0.5,
            },
            'fashion_mnist': {
                'lr': 0.056,
            },
            'cifar10': {
                'lr': 0.01,
            },
            'cifar100': {
                'lr': 0.01  # Default
            }
        },
        'relu': {
            'cifar10': {
                'lr': 0.01,
            },
            'cifar100': {
                'lr': 0.01  # Default
            }
        }
    }
}

POST_NN_LR = {
    'nnsgd': {
        'mlp': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.78,
                    'lr_out': 0.23,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.9,
                    'lr_out': 0.68,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.28,
                    'lr_out': 0.3,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.9,
                    'lr_out': 0.6,
                }
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_in': 0.37,
                    'lr_out': 0.3,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.91,
                    'lr_out': 0.086,
                }
            },
        },
        'cnn': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.93,
                    'lr_out': 0.21,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.56,
                    'lr_out': 0.69,
                }
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.61,
                    'lr_out': 0.082,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.58,
                    'lr_out': 0.41,
                }
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_in': 0.37,
                    'lr_out': 0.3,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.91,
                    'lr_out': 0.086,
                }
            },
        }
    },
    'csgd': {
        'mlp': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.002,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.009,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0027,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.008,
                },
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0034,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.002,
                },
            },
        },
        'cnn': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0024,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.04,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0035,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.024,
                },
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_nn': 0.00002,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.00001,
                },
            },
        }
    }
}

FULLY_NN_LR = {
    'nnsgd': {
        'mlp': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.64,
                    'lr_out': 0.6,
                },
                'photonic_sinusoidal': {
                    'lr_in': 1.0,
                    'lr_out': 1.0,
                },
                'relu': {
                    'lr_in': 0.0005,
                    'lr_out': 1.0,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.28,
                    'lr_out': 0.91,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.63,
                    'lr_out': 0.5,
                },
                'relu': {
                    'lr_in': 0.63,
                    'lr_out': 0.5,
                }
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_in': 0.5,
                    'lr_out': 0.5,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.86,
                    'lr_out': 0.68,
                },
                'relu': {
                    'lr_in': 0.86,
                    'lr_out': 0.68,
                }
            },
        },
        'cnn': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.39,
                    'lr_out': 0.63,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.83,
                    'lr_out': 0.78,
                },
                'relu': {
                    'lr_in': 0.83,
                    'lr_out': 0.78,
                }
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_in': 0.1,
                    'lr_out': 0.11,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.77,
                    'lr_out': 0.56,
                },
                'relu': {
                    'lr_in': 0.77,
                    'lr_out': 0.56,
                }
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_in': 0.28,
                    'lr_out': 0.38,
                },
                'photonic_sinusoidal': {
                    'lr_in': 0.71,
                    'lr_out': 0.37,
                },
                'relu': {
                    'lr_in': 0.71,
                    'lr_out': 0.37,
                }
            },
        }
    },
    'csgd': {
        'mlp': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0067,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.034,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0091,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.05,
                },
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_nn': 0.0049,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.024,
                },
            },
        },
        'cnn': {
            'mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.004,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.039,
                },
            },
            'fashion_mnist': {
                'photonic_sigmoid': {
                    'lr_nn': 0.004,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.024,
                },
            },
            'cifar10': {
                'photonic_sigmoid': {
                    'lr_nn': 0.002,
                },
                'photonic_sinusoidal': {
                    'lr_nn': 0.0034,
                },
            },
        }
    }
}


def get_transformation_config(architecture, dataset, activation):
    config = {}

    if activation == 'relu':
        config.update({
            'activation': T.nn.ReLU,
            'nn_activation': lambda: ReLUN(MAX_RELU_BOUND),
            'alpha': MAX_RELU_BOUND
        })

    elif 'relu' in activation:
        bound = int(activation[4:])
        config.update({
            'activation': T.nn.ReLU,
            'nn_activation': lambda: ReLUN(bound),
            'alpha': MAX_RELU_BOUND
        })
    else:
        config.update(ACTIVATIONS_CONFIG[activation].copy())

    config.update(DATASET_CONFIG[dataset].copy())

    if 'relu' in activation:
        config.update(
            TRANFORMATIONS_LR[architecture][activation[:4]][dataset].copy())
    else:
        config.update(
            TRANFORMATIONS_LR[architecture][activation][dataset].copy())

    return config


def get_nn_post_config(architecture, dataset, activation, optimizer):
    config = {}

    config.update(ACTIVATIONS_CONFIG[activation].copy())
    config.update(DATASET_CONFIG[dataset].copy())
    config.update(TRANFORMATIONS_LR[architecture][activation][dataset].copy())
    config.update(
        POST_NN_LR[optimizer][architecture][dataset][activation].copy())

    return config


def get_fully_nn_config(architecture, dataset, activation, optimizer):

    config = {}

    if activation == 'relu':
        config.update({
            'activation': T.nn.ReLU,
            'nn_activation': lambda: ReLUN(MAX_RELU_BOUND),
            'alpha': MAX_RELU_BOUND
        })

    elif 'relu' in activation:
        bound = int(activation[4:])
        config.update({
            'activation': T.nn.ReLU,
            'nn_activation': lambda: ReLUN(bound),
            'alpha': MAX_RELU_BOUND
        })
    else:
        config.update(ACTIVATIONS_CONFIG[activation].copy())

    config.update(DATASET_CONFIG[dataset].copy())
    if 'relu' in activation:
        config.update(FULLY_NN_LR[optimizer][architecture][dataset][
            activation[:4]].copy())
    else:
        config.update(
            FULLY_NN_LR[optimizer][architecture][dataset][activation].copy())

    return config
