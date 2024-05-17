import re
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from nn_isomorphics.nnn.nn_resblocks import (NNResBlock, NNResBottleneck,
                                             NNResNet)
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.utils import _log_api_usage_once


def set_activations(act_func_cls, nn_act_func_cls):
    activation = act_func_cls()
    act_func_cls = act_func_cls
    nn_act_func_cls = nn_act_func_cls
    if nn_act_func_cls is None:
        nn_act_func_cls = act_func_cls

    return activation, act_func_cls, nn_act_func_cls


def get_alpha(bound, layers):
    alpha = []
    for i in range(7):
        if (i == 0 or i == 5 or i == 6):
            alpha.append(torch.tensor(float(bound)))
        else:
            alpha.append(
                [torch.tensor(float(bound)) for _ in range(layers[i - 1] + 3)])
    return alpha


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 act_func_cls=lambda: nn.ReLU(inplace=True),
                 nn_act_func_cls=None,
                 alpha: list[float] = [1.0, 1.0]) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            print('Groups {} Base Width {}'.format(groups, base_width))
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stride = stride

        self.layers_dict = nn.ModuleDict({
            'conv1':
            conv3x3(inplanes, planes, stride),
            'bn1':
            norm_layer(planes),
            'conv2':
            conv3x3(planes, planes),
            'bn2':
            norm_layer(planes),
        })

        if downsample is not None:
            self.layers_dict.update({'downsample': downsample})

        self.activation, self.act_func_cls, self.nn_act_func_cls = set_activations(
            act_func_cls, nn_act_func_cls)

        self.alpha = alpha

    def get_nn_module(self):
        return NNResBlock(self.nn_act_func_cls)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.layers_dict['conv1'](x)
        out = self.layers_dict['bn1'](out)
        out = self.activation(out)

        out = self.layers_dict['conv2'](out)
        out = self.layers_dict['bn2'](out)

        if 'downsample' in self.layers_dict:
            identity = self.layers_dict['downsample'](x)

        out += identity
        out = self.activation(out)

        return out


class ResBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 act_func_cls=lambda: nn.ReLU(inplace=True),
                 nn_act_func_cls=None,
                 alpha=None) -> None:
        super().__init__()

        self.inplanes = inplanes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.stride = stride

        self.layers_dict = nn.ModuleDict({
            'conv1':
            conv1x1(inplanes, width),
            'bn1':
            norm_layer(width),
            'conv2':
            conv3x3(width, width, stride, groups, dilation),
            'bn2':
            norm_layer(width),
            'conv3':
            conv1x1(width, planes * self.expansion),
            'bn3':
            norm_layer(planes * self.expansion),
        })

        if downsample is not None:
            self.layers_dict.update({'downsample': downsample})

        self.activation, self.act_func_cls, self.nn_act_func_cls = set_activations(
            act_func_cls, nn_act_func_cls)

        self.alpha = alpha

    def get_nn_module(self):
        return NNResBottleneck(self.nn_act_func_cls)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.layers_dict['conv1'](x)
        out = self.layers_dict['bn1'](out)
        out = self.activation(out)

        out = self.layers_dict['conv2'](out)
        out = self.layers_dict['bn2'](out)
        out = self.activation(out)

        out = self.layers_dict['conv3'](out)
        out = self.layers_dict['bn3'](out)

        if 'downsample' in self.layers_dict.keys():
            identity = self.layers_dict['downsample'](x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[ResBlock, ResBottleneck]],
        layers_list: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act_func_cls=lambda: nn.ReLU(inplace=True),
        nn_act_func_cls=None,
        alpha: list[float] = [1.0, 1.0],
    ) -> None:

        super().__init__()

        _log_api_usage_once(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.init_inplanes = self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}")

        self.groups = groups

        self.base_width = width_per_group

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialization(zero_init_residual)

        self.layers_dict = nn.ModuleDict({
            'conv1':
            nn.Conv2d(3,
                      self.inplanes,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            'bn1':
            norm_layer(self.inplanes),
            'layer1':
            self._make_layer(
                block,
                64,
                layers_list[0],
                act_func_cls,
                nn_act_func_cls,
                alpha=alpha[1],
            ),
            'layer2':
            self._make_layer(
                block,
                128,
                layers_list[1],
                act_func_cls,
                nn_act_func_cls,
                stride=2,
                dilate=replace_stride_with_dilation[0],
                alpha=alpha[2],
            ),
            'layer3':
            self._make_layer(
                block,
                256,
                layers_list[2],
                act_func_cls,
                nn_act_func_cls,
                stride=2,
                dilate=replace_stride_with_dilation[1],
                alpha=alpha[3],
            ),
            'layer4':
            self._make_layer(
                block,
                512,
                layers_list[3],
                act_func_cls,
                nn_act_func_cls,
                stride=2,
                dilate=replace_stride_with_dilation[2],
                alpha=alpha[3],
            ),
            'fc':
            nn.Linear(512 * block.expansion, num_classes)
        })

        self.activation, self.act_func_cls, self.nn_act_func_cls = set_activations(
            act_func_cls, nn_act_func_cls)

        self.alpha = alpha

    def _initialization(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight,
                                      0)  # type: ignore[arg-type]
                elif isinstance(m, ResBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight,
                                      0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[ResBlock, ResBottleneck]],
        planes: int,
        blocks: int,
        act_func_cls,
        nn_act_func_cls,
        stride: int = 1,
        dilate: bool = False,
        alpha: list[float] = [1.0, 1.0],
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  self.base_width,
                  previous_dilation,
                  norm_layer,
                  act_func_cls=act_func_cls,
                  nn_act_func_cls=nn_act_func_cls,
                  alpha=alpha))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer,
                      act_func_cls=act_func_cls,
                      nn_act_func_cls=nn_act_func_cls,
                      alpha=alpha))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.layers_dict['conv1'](x)
        x = self.layers_dict['bn1'](x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layers_dict['layer1'](x)  # Sequential
        x = self.layers_dict['layer2'](x)  # Sequential
        x = self.layers_dict['layer3'](x)  # Sequential
        x = self.layers_dict['layer4'](x)  # Sequential

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.layers_dict['fc'](x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = {'layers_dict.' + k: v for k, v in state_dict.items()}

        old_keys = list(state_dict.keys())

        new_keys = [
            re.sub(r'layer(\d+.\d+).', r'layer\1.layers_dict.', k)
            for k in old_keys
        ]

        key_re = dict(map(lambda i, j: (i, j), old_keys, new_keys))

        state_dict = {key_re[k]: v for k, v in state_dict.items()}

        super().load_state_dict(state_dict, strict)

    def get_nn_module(self):
        return NNResNet(self.nn_act_func_cls)


def resnet(
    block: Type[Union[ResBlock, ResBottleneck]],
    layers: List[int],
    act_func_cls,
    nn_act_func_cls,
    bound,
    weights: Optional[WeightsEnum] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:

    print(kwargs)
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes",
                              len(weights.meta["categories"]))

    model = ResNet(
        block,
        layers,
        act_func_cls=act_func_cls,
        nn_act_func_cls=nn_act_func_cls,
        alpha=get_alpha(bound, layers),
    )

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True))

    return model


ALL_RESNETS = {
    'resnet18':
    lambda act_func_cls, nn_act_func_cls, bound: resnet(
        ResBlock,
        [2, 2, 2, 2],
        act_func_cls,
        nn_act_func_cls,
        bound,
    ),
    "resnet34":
    lambda act_func_cls, nn_act_func_cls, bound: resnet(
        ResBlock,
        [3, 4, 6, 3],
        act_func_cls,
        nn_act_func_cls,
        bound,
    ),
    "resnet50":
    lambda act_func_cls, nn_act_func_cls, bound: resnet(
        ResBottleneck,
        [3, 4, 6, 3],
        act_func_cls,
        nn_act_func_cls,
        bound,
    ),
    "resnet101":
    lambda act_func_cls, nn_act_func_cls, bound: resnet(
        ResBottleneck,
        [3, 4, 23, 3],
        act_func_cls,
        nn_act_func_cls,
        bound,
    ),
    "resnet152":
    lambda act_func_cls, nn_act_func_cls, bound: resnet(
        ResBottleneck,
        [3, 8, 36, 3],
        act_func_cls,
        nn_act_func_cls,
        bound,
    ),
}

####################### SHOULD BE CHECKED ###########################
# _ovewrite_named_param(kwargs, "groups", 32)
# _ovewrite_named_param(kwargs, "width_per_group", 4)
# ALL_RESNEXTS = {
#     "resnext50_32x4d": lambda: resnet(ResBottleneck, [3, 4, 6, 3]),
#     "resnext101_32x8d": lambda: resnet(ResBottleneck, [3, 4, 23, 3]),
#     "resnext101_64x4d": lambda: resnet(ResBottleneck, [3, 4, 23, 3]),
# }

# # _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
# ALL_WIDE_RESNERS = {
#     "wide_resnet50_2": lambda: resnet(ResBottleneck, [3, 4, 6, 3]),
#     "wide_resnet101_2": lambda: resnet(ResBottleneck, [3, 4, 23, 3]),
# }
#####################################################################
