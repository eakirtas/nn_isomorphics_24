from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


class NNResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, nn_act_func_cls) -> None:
        super().__init__()

        self.activation = nn_act_func_cls()

        self.module_dict = nn.ModuleDict()

    def add_layer(self, key, layer):
        self.module_dict.update({key: layer})

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.module_dict['conv1'](x)
        out = self.module_dict['bn1'](out)
        out = self.activation(out)

        out = self.module_dict['conv2'](out)
        out = self.module_dict['bn2'](out)

        if 'downsample' in self.module_dict:
            identity = self.module_dict['downsample'](x)

        out += identity
        out = self.activation(out)

        return out


class NNResBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, nn_act_func_cls) -> None:
        super().__init__()
        self.activation = nn_act_func_cls()

        self.module_dict = nn.ModuleDict()

    def add_layer(self, key, layer):
        self.module_dict.update({key: layer})

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.module_dict['conv1'](x)
        out = self.module_dict['bn1'](out)
        out = self.activation(out)

        out = self.module_dict['conv2'](out)
        out = self.module_dict['bn2'](out)
        out = self.activation(out)

        out = self.module_dict['conv3'](out)
        out = self.module_dict['bn3'](out)

        if 'downsample' in self.module_dict:
            identity = self.module_dict['downsample'](x)

        out += identity
        out = self.activation(out)

        return out


class NNResNet(nn.Module):

    def __init__(
        self,
        nn_act_func_cls,
    ) -> None:

        super().__init__()

        self.activation = nn_act_func_cls()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers_dict = nn.ModuleDict()

    def add_layer(self, key, module):
        self.layers_dict.update({key: module})

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
