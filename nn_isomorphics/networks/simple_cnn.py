import torch as T


class SimpleCNNStride(T.nn.Module):

    def __init__(self, output_size, activation_function, alpha=None):
        super().__init__()

        self.output_size = output_size
        self.activation_function = activation_function
        self.alpha = T.tensor(alpha)

        self.conv1 = T.nn.Conv2d(3, 32, kernel_size=3, padding=2)
        self.conv1_act = activation_function()

        self.conv2 = T.nn.Conv2d(32, 64, kernel_size=3, padding=2)
        self.conv2_act = activation_function()
        self.pool_2 = T.nn.AvgPool2d(2, 2)

        self.conv3 = T.nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_act = activation_function()

        self.conv4 = T.nn.Conv2d(128, 256, kernel_size=3)
        self.conv4_act = activation_function()
        self.pool_4 = T.nn.AvgPool2d(2, 2)

        self.fc1 = T.nn.Linear(12544, 512)
        self.fc1_act = activation_function()

        self.fc2 = T.nn.Linear(512, output_size)
        self.fc2_act = T.nn.LogSoftmax(1)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1,
            self.fc2,
        ]

        self.acts = [
            self.conv1_act,
            self.conv2_act,
            self.conv3_act,
            self.conv4_act,
            self.fc1_act,
            self.fc2_act,
        ]

    def get_nn_net(self):
        return NNSimpleCNN(self.output_size, self.activation_function)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)

        x = self.conv2(x)
        x = self.conv2_act(x)
        x = self.pool_2(x)

        x = self.conv3(x)
        x = self.conv3_act(x)

        x = self.conv4(x)
        x = self.conv4_act(x)
        x = self.pool_4(x)

        x = T.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)

        return x


class SimpleCNN(T.nn.Module):

    def __init__(self, output_size, activation_function, alpha=None):
        super().__init__()

        self.output_size = output_size
        self.activation_function = activation_function
        self.alpha = T.tensor(alpha)

        self.conv1 = T.nn.Conv2d(3, 32, kernel_size=3)
        self.conv1_act = activation_function()

        self.conv2 = T.nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_act = activation_function()
        self.pool_2 = T.nn.AvgPool2d(2, 2)

        self.conv3 = T.nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_act = activation_function()

        self.conv4 = T.nn.Conv2d(128, 256, kernel_size=3)
        self.conv4_act = activation_function()
        self.pool_4 = T.nn.AvgPool2d(2, 2)

        self.fc1 = T.nn.Linear(256 * 5 * 5, 512)
        self.fc1_act = activation_function()

        self.fc2 = T.nn.Linear(12544, output_size)
        self.fc2_act = T.nn.LogSoftmax(1)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1,
            self.fc2,
        ]

        self.acts = [
            self.conv1_act,
            self.conv2_act,
            self.conv3_act,
            self.conv4_act,
            self.fc1_act,
            self.fc2_act,
        ]

    def get_nn_net(self):
        return NNSimpleCNN(self.output_size, self.activation_function)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_act(x)

        x = self.conv2(x)
        x = self.conv2_act(x)
        x = self.pool_2(x)

        x = self.conv3(x)
        x = self.conv3_act(x)

        x = self.conv4(x)
        x = self.conv4_act(x)
        x = self.pool_4(x)

        x = T.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)

        return x


class NNSimpleCNN(T.nn.Module):

    def __init__(self, output_size, activation_function):
        super().__init__()

        self.layers = T.nn.ModuleList()

        self.conv1_act = activation_function()

        self.conv2_act = activation_function()
        self.pool_2 = T.nn.AvgPool2d(2, 2)

        self.conv3_act = activation_function()

        self.conv4_act = activation_function()
        self.pool_4 = T.nn.AvgPool2d(2, 2)

        self.fc1_act = activation_function()

        self.fc2_act = T.nn.LogSoftmax(1)

        self.layers = T.nn.ModuleList()

        self.acts = [
            self.conv1_act, self.conv2_act, self.conv3_act, self.conv4_act,
            self.fc1_act, self.fc2_act
        ]

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.conv1_act(x)

        x = self.layers[1](x)
        x = self.conv2_act(x)
        x = self.pool_2(x)

        x = self.layers[2](x)
        x = self.conv3_act(x)

        x = self.layers[3](x)
        x = self.conv4_act(x)
        x = self.pool_4(x)

        x = T.flatten(x, 1)

        x = self.layers[4](x)
        x = self.fc1_act(x)

        x = self.layers[5](x)
        x = self.fc2_act(x)

        return x
