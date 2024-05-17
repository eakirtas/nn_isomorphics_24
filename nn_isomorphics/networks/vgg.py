from typing import Dict, List, Union, cast

import torch
import torch.nn as nn


def get_all_modules(module):
    all_modules = []

    def add_module(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            all_modules.append(module)

    # Recursively traverse the module and its submodules
    module.apply(add_module)

    return all_modules


class NNVGG(nn.Module):

    def __init__(
        self,
        version: str = 'vgg19',
        act_func_cls=nn.ReLU,
        dropout: float = 0.5,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential()

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_list = [
            None,
            act_func_cls,
            lambda: nn.Dropout(p=dropout),
            None,
            act_func_cls,
            lambda: nn.Dropout(p=dropout),
            None,
        ]

        self.classifier = nn.Sequential()

        self.act_func_cls = act_func_cls
        self.batch_norm = batch_norm
        self.version = version

        self._c = 0

    def add_layer(
        self,
        layer,
    ) -> None:

        cfg = cfgs[self.version]

        if self._c < len(cfg):
            flag = False
            while self._c < len(cfg) and (not flag or cfg[self._c] == 'M'):
                if cfg[self._c] == 'M':
                    self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.features.append(layer)
                    v = cast(int, cfg[self._c])
                    if self.batch_norm:
                        self.features.append(nn.BatchNorm2d(v))
                    self.features.append(self.act_func_cls())
                    flag = True
                self._c += 1

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):

    def __init__(self,
                 version: str = 'vgg1',
                 act_func_cls=nn.ReLU,
                 nn_activation=None,
                 num_classes: int = 1000,
                 init_weights: bool = True,
                 dropout: float = 0.5,
                 batch_norm: bool = False,
                 alpha=None) -> None:
        super().__init__()

        self.features = self.make_layers(version, batch_norm, act_func_cls)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            act_func_cls(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            act_func_cls(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        self.layers = get_all_modules(self)

        # self.alpha = torch.tensor([alpha for _ in range(int(version[-2:]))])

        self.alpha = torch.tensor(
            [6] + [alpha for _ in range(int(version[-2:]) - 1)])

        self.act_func_cls = act_func_cls
        self.nn_activation = nn_activation
        if nn_activation is None:
            self.nn_activation = act_func_cls

        self.version = version
        self.dropout = dropout
        self.batch_norm = batch_norm

    def get_nn_module(self):
        return NNVGG(
            self.version,
            self.nn_activation,
            self.dropout,
            self.batch_norm,
        )

    def make_layers(
        self,
        version: str,
        batch_norm: bool = False,
        act_func_cls=torch.nn.ReLU,
    ) -> nn.Sequential:

        layers: List[nn.Module] = []
        in_channels = 3

        cfg = cfgs[version]
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), act_func_cls()]
                else:
                    layers += [conv2d, act_func_cls()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13":
    [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
        512, 512, 512, "M"
    ],
    "vgg19": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
        512, "M", 512, 512, 512, 512, "M"
    ],
}
