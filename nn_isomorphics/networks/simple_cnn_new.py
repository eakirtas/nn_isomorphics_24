import torch as T


def get_all_modules(module):
    all_modules = []

    def add_module(module):
        if isinstance(module, T.nn.Linear) or isinstance(module, T.nn.Conv2d):
            all_modules.append(module)

    # Recursively travesrse the module and its submodules
    module.apply(add_module)

    return all_modules


class SimpleCNN_v2(T.nn.Module):

    def __init__(self,
                 output_size,
                 activation,
                 nn_activation=None,
                 alpha=None):
        super().__init__()

        self.features = T.nn.Sequential(
            T.nn.Conv2d(3, 32, kernel_size=3),
            activation(),
            T.nn.Conv2d(32, 64, kernel_size=3),
            activation(),
            T.nn.AvgPool2d(2, 2),
            T.nn.Conv2d(64, 128, kernel_size=3),
            activation(),
            T.nn.Conv2d(128, 256, kernel_size=3),
            activation(),
            T.nn.AvgPool2d(2, 2),
        )

        self.classifier = T.nn.Sequential(
            T.nn.Linear(256 * 5 * 5, 512),
            activation(),
            T.nn.Linear(512, output_size),
            # T.nn.LogSoftmax(1),
        )

        self.output_size = output_size

        self.activation = activation
        self.nn_activation = nn_activation
        if nn_activation is None:
            self.nn_activation = activation

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = T.tensor(alpha)
            else:
                self.alpha = T.tensor([alpha for _ in range(6)])

        self.layers = get_all_modules(self)

    def get_nn_net(self):
        return NNSimpleCNN_v2(self.output_size, self.nn_activation)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)

        return x


class NNSimpleCNN_v2(T.nn.Module):

    def __init__(self, output_size, activation_function):
        super().__init__()

        self.features = T.nn.Sequential()
        self.features_list = [
            None,
            activation_function,
            None,
            activation_function,
            lambda: T.nn.AvgPool2d(2, 2),
            None,
            activation_function,
            None,
            activation_function,
            lambda: T.nn.AvgPool2d(2, 2),
        ]

        self.classifier = T.nn.Sequential()
        self.classifier_list = [
            None,
            activation_function,
            None,
            # lambda: T.nn.LogSoftmax(1),
        ]

        self.counter = 0

    def add_layer(self, layer):
        if self.counter < 4:
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)

        return x
