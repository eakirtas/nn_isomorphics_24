from typing import Any, Callable, List, Optional, Type, Union

import torch as T
from nn_isomorphics.networks.nn_resnet import ResBlock, ResBottleneck

from .nn_conv1d import NNConv1d
from .nn_conv2d import NNConv2d
from .nn_layer import NNLinear
from .nn_resblocks import NNResBlock, NNResBottleneck

BASIC_MODULES = {
    'Linear': NNLinear,
    'Conv1d': NNConv1d,
    'Conv2d': NNConv2d,
    # 'Sequential': T.nn.Sequential
}
RES_MODULES = {
    'ResBlock': NNResBlock,
    'ResBottleneck': NNResBottleneck,
}

ACTIVATIONS = ['ReLUN', 'ReLU', 'GELU', 'GELUN']

LIST_CASES = ['AlexNet', 'VGG']
DICT_CASES = ['ResNet']


def nn_transform_sequential(
    sequential: T.nn.Sequential,
    alpha: T.Tensor,
) -> T.nn.Sequential:
    # print('Start Sequential')
    nn_modules = []
    for i, module in enumerate(sequential):
        if module.__class__.__name__ in BASIC_MODULES:
            # print('Transform Basic Module')
            # print("Alpha:", type(alpha[i]))
            nn_modules.append(nn_tranform_module(module, T.tensor(
                alpha[i])))  #TODO: Fix that!
        elif module.__class__.__name__ == 'Sequential':
            # print('Transform Sequential')
            nn_modules.append(nn_transform_sequential(module, alpha[i]))
        elif module.__class__.__name__ in RES_MODULES:
            # print('Transform ResNet Module')
            nn_modules.append(nn_res_transform(module))
        elif module.__class__.__name__ in 'BatchNorm2d':
            # print('Transform Batch Norm Module')
            nn_modules.append(module)
        elif module.__class__.__name__ in 'Dropout':
            nn_modules.append(module)
        elif module.__class__.__name__ in ACTIVATIONS:
            nn_modules.append(module)
        else:
            raise Exception("Sequential Transformation do not support " +
                            module.__class__.__name__)
    # print('End Sequential')
    return T.nn.Sequential(*nn_modules)


def nn_res_transform(module: Type[Union[ResBlock, ResBottleneck]]):
    nn_module = module.get_nn_module()
    alpha = module.alpha

    for i, (key, inner_module) in enumerate(module.layers_dict.items()):
        if inner_module.__class__.__name__ in BASIC_MODULES:
            # print('Transform basic Module - ', key)
            inner_nn_module = nn_tranform_module(inner_module, alpha[i])
            nn_module.add_layer(key, inner_nn_module)
        elif inner_module.__class__.__name__ in 'BatchNorm2d':
            # print('Transform Batch Norm Module - ', key)
            nn_module.add_layer(key, inner_module)
        elif inner_module.__class__.__name__ == 'Sequential':
            # print('Transform Sequential', key)
            nn_module.add_layer(key,
                                nn_transform_sequential(inner_module,
                                                        alpha))  # TODO: Fix
        else:
            print('Class name', inner_module.__class__.__name__)
            raise Exception("This layer type cannot be converted")
    return nn_module


def nn_transformation_dict(model: T.nn.Module) -> T.nn.Module:
    # Get the non-negative replica of model
    nn_model = model.get_nn_module()
    alpha = model.alpha

    for i, (key, module) in enumerate(model.layers_dict.items()):
        if module.__class__.__name__ in BASIC_MODULES:
            # print('Transform Basic Module')
            nn_module = nn_tranform_module(module, alpha[i])
            nn_model.add_layer(key, nn_module)
        elif module.__class__.__name__ == 'Sequential':
            # print('Transform Sequential')
            nn_module = nn_transform_sequential(module, alpha[i])
            nn_model.add_layer(key, nn_module)
        elif module.__class__.__name__ in RES_MODULES:
            # print('Transform ResNet Module')
            nn_module = nn_res_transform(module, alpha[i])
            nn_model.add_layer(key, nn_module)
        elif module.__class__.__name__ in 'BatchNorm2d':
            # print('Transform Batch Norm Module')
            nn_module = module
            nn_model.add_layer(key, nn_module)
        elif module.__class__.__name__ == 'LayerNorm':
            # print('Transform Layer Norm Module')
            nn_module = module
            nn_model.add_layer(key, nn_module)
        elif module.__class__.__name__ == 'Dropout':
            # print('Transform Dropout Module')
            nn_model.add_layer(key, module)
        else:
            raise Exception(
                "Sequential Transformation Support only Build-in and Res Modules"
            )

    return nn_model


def nn_transformation_list(module: T.nn.Module, alpha) -> T.nn.Module:
    """
    Receives a regular models and return a fully non-negative one.

    param model: The model that will be transformed into non-negative one
    type model: T.nn.Module
    param dtype: Type of model
    type mode: Type
    """
    # Get the non-negative replica of model
    nn_module = module.get_nn_module()

    for i, inner_module in enumerate(module.layers):
        if inner_module.__class__.__name__ in BASIC_MODULES:
            nn_layer = nn_tranform_module(module.layers[i], alpha[i])
            nn_module.add_layer(nn_layer)
        elif inner_module.__class__.__name__ in 'Dropout':
            nn_module.add_layer(inner_module)
        elif inner_module.__class__.__name__ in 'LayerNorm':
            nn_module.add_layer(inner_module)

    return nn_module


def nn_transformation(model):
    if model.__class__.__name__ in DICT_CASES:
        nn_model = nn_transformation_dict(model)
    else:
        nn_model = nn_transformation_list(model, model.alpha)

    return nn_model


def nn_tranform_module(
    module: T.nn.Module,
    alpha: T.Tensor,
) -> T.nn.Module:

    classname = module.__class__.__name__

    if classname.find('Linear') != -1:
        sum_dim = 1
    elif classname.find('Conv1d') != -1:
        sum_dim = (1, 2)
    elif classname.find('Conv2d') != -1:
        sum_dim = (1, 2, 3)
    else:
        raise Exception("This layer type cannot be converted")

    # Get the non-negative weights
    w_neg = T.clamp_max(module.weight, 0)

    # Calculated b_tilde
    if module.bias is not None:
        b_tilde = module.bias - alpha * T.sum(T.abs(w_neg), dim=sum_dim)
    else:
        b_tilde = -alpha * T.sum(T.abs(w_neg), dim=sum_dim)

    b_new, act_shift = calc_b_new(module, b_tilde)

    # Create two different tensors one for positive and one for negative biases
    w_neg_abs = T.abs(w_neg)
    w_pos = T.clamp_min(module.weight, 0)

    if classname.find('Linear') != -1:
        nn_layer = NNLinear(
            module.in_features,
            module.out_features,
            w_pos,
            w_neg_abs,
            b_new,
            alpha,
            act_shift,
        )
    elif classname.find('Conv2d') != -1:
        nn_layer = NNConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            w_pos,
            w_neg_abs,
            b_new,
            alpha,
            act_shift,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            padding_mode=module.padding_mode,
        )
    elif classname.find('Conv1d') != -1:
        nn_layer = NNConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            w_pos,
            w_neg_abs,
            b_new,
            alpha,
            act_shift,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            padding_mode=module.padding_mode,
            # device=module.device(),
            # dtype=module.dtype
        )
    else:
        raise Exception("This layer type cannot be converted")

    return nn_layer


def calc_b_new(layer, b_tilde):
    with T.no_grad():
        act_shift = T.max(T.abs(b_tilde))

        if layer.bias is not None:
            old_bias = layer.bias
        else:
            old_bias = T.zeros_like(b_tilde)

        # For b_tilde < 0 replace biases with b_tilde
        b_new = T.scatter(old_bias, 0,
                          T.where(b_tilde < 0)[0],
                          b_tilde[b_tilde < 0]).type(old_bias.dtype)

        # Add the activation shifting point to the b_tilde < 0 case
        b_new = T.scatter_add(
            -T.abs(b_new), 0,
            T.where(b_tilde < 0)[0],
            act_shift.expand_as(T.where(b_tilde < 0)[0])).type(old_bias.dtype)

        # Add the activation shifting point to b_tilde > 0
        b_new = T.scatter_add(
            b_new, 0,
            T.where(b_tilde > 0)[0],
            act_shift.expand_as(T.where(b_tilde > 0)[0])).type(old_bias.dtype)

    return b_new, act_shift
