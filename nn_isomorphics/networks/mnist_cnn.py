import torch as T


class MnistCNN(T.nn.Module):

    def __init__(self,
                 output_size,
                 activation,
                 nn_activation,
                 alpha=None) -> None:
        super().__init__()

        self.output_size = output_size

        self.conv1 = T.nn.Conv2d(1, 32, 3)
        self.conv1_act = activation()

        self.conv2 = T.nn.Conv2d(32, 32, 3, 2)
        self.conv2_act = activation()

        self.conv3 = T.nn.Conv2d(32, 64, 3)
        self.conv3_act = activation()

        self.conv4 = T.nn.Conv2d(64, 64, 3, 2)
        self.conv4_act = activation()

        self.flatten = T.nn.Flatten(start_dim=1)
        self.fc1 = T.nn.Linear(1024, 128)
        self.fc1_act = activation()

        self.fc2 = T.nn.Linear(128, output_size)
        # self.fc2_act = T.nn.LogSoftmax(1)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1,
            self.fc2,
        ]

        self.activation = activation
        self.nn_activation = nn_activation
        if nn_activation is None:
            self.nn_activation = activation

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = T.tensor(alpha)
            else:
                self.alpha = T.tensor([alpha for _ in range(6)])

        # self.acts = [
        #     self.conv1_act,
        #     self.conv2_act,
        #     self.conv3_act,
        #     self.conv4_act,
        #     self.fc1_act,
        #     self.fc2_act,
        # ]

    def get_nn_net(self):
        return NNMnistCNN(self.output_size, self.nn_activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)

        x = self.conv2(x)
        x = self.conv2_act(x)

        x = self.conv3(x)
        x = self.conv3_act(x)

        x = self.conv4(x)
        x = self.conv4_act(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        # x = self.fc2_act(x)

        return x


class NNMnistCNN(T.nn.Module):

    def __init__(self, output_size, activation_function):
        super().__init__()

        self.layers = T.nn.ModuleList()

        self.conv1_act = activation_function()

        self.conv2_act = activation_function()

        self.conv3_act = activation_function()

        self.conv4_act = activation_function()

        self.flatten = T.nn.Flatten(start_dim=1)
        self.fc1_act = activation_function()

        # self.fc2_act = T.nn.LogSoftmax(1)

        self.layers = T.nn.ModuleList()

        # self.acts = [
        #     self.conv1_act,
        #     self.conv2_act,
        #     self.conv3_act,
        #     self.conv4_act,
        #     self.fc1_act,
        #     self.fc2_act,
        # ]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.conv1_act(x)

        x = self.layers[1](x)
        x = self.conv2_act(x)

        x = self.layers[2](x)
        x = self.conv3_act(x)

        x = self.layers[3](x)
        x = self.conv4_act(x)

        x = self.flatten(x)
        x = self.layers[4](x)
        x = self.fc1_act(x)

        x = self.layers[5](x)
        # x = self.fc2_act(x)

        return x
