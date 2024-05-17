# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch as T
from torch import Tensor, nn

__all__ = [
    "AlexNet",
    "alexnet",
]


class NNAlexNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 act_func_cls=nn.ReLU):

        super().__init__()

        self.features = nn.Sequential()
        self.features_list = [
            None,
            act_func_cls,
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
            None,
            act_func_cls,
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
            None,
            act_func_cls,
            None,
            act_func_cls,
            None,
            act_func_cls,
            lambda: nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential()
        self.classifier_list = [
            lambda: nn.Dropout(p=dropout),
            None,
            act_func_cls,
            lambda: nn.Dropout(p=dropout),
            None,
            act_func_cls,
            None,
            # lambda: T.nn.LogSoftmax(1),
        ]

        self.counter = 0

    def add_layer(self, layer):
        if self.counter < 5:
            i = len(self.features)
            while self.features_list[i] is not None:
                self.features.append(self.features_list[i]())
                i += 1
            self.features.append(layer)
            i += 1
            while i < len(
                    self.features_list) and self.features_list[i] is not None:
                self.features.append(self.features_list[i]())
                i += 1

        else:
            i = len(self.classifier)
            while i < len(self.classifier_list
                          ) and self.classifier_list[i] is not None:
                self.classifier.append(self.classifier_list[i]())
                i += 1
            self.classifier.append(layer)
            i += 1
            while i < len(self.classifier_list
                          ) and self.classifier_list[i] is not None:
                self.classifier.append(self.classifier_list[i]())
                i += 1

        self.counter += 1

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_all_modules(module):
    all_modules = []

    def add_module(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            all_modules.append(module)

    # Recursively travesrse the module and its submodules
    module.apply(add_module)

    return all_modules


class AlexNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.5,
                 act_func_cls=nn.ReLU,
                 nn_activation=None,
                 alpha=None) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            act_func_cls(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            act_func_cls(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_func_cls(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_func_cls(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_func_cls(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            act_func_cls(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            act_func_cls(),
            nn.Linear(4096, num_classes),
            # T.nn.LogSoftmax(1),
        )

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = T.tensor(alpha)
            else:
                self.alpha = T.tensor([6] + [alpha for _ in range(7)])

        self.act_func_cls = act_func_cls
        self.nn_activation = act_func_cls
        if nn_activation is None:
            self.nn_activation = act_func_cls

        self.num_classes = num_classes

        self.layers = get_all_modules(self)

    def get_nn_module(self):
        return NNAlexNet(num_classes=self.num_classes,
                         act_func_cls=self.nn_activation)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x
