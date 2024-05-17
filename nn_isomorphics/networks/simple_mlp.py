import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class NNSimpleMLP(T.nn.Module):

    def __init__(self, activation_function):
        super().__init__()
        super().cuda()

        self.a_func_1 = activation_function()
        self.a_func_2 = activation_function()
        self.a_func_3 = T.nn.LogSoftmax(1)

        self.layers = T.nn.ModuleList()

    def add_layer(self, nn_layer):
        self.layers.append(nn_layer)

    def forward(self, x):
        x = T.flatten(x, start_dim=1)

        x = self.layers[0](x)
        x = self.a_func_1(x)

        x = self.layers[1](x)
        x = self.a_func_2(x)

        x = self.layers[2](x)
        x = self.a_func_3(x)

        return x

    def __repr__(self):
        return "NNSimpleMLP"


class SimpleMLP(T.nn.Module):

    def __init__(self,
                 n_input,
                 n_output,
                 activation,
                 nn_activation=None,
                 alpha=None,
                 sizes=[100, 100]):
        super().__init__()
        super().cuda()

        self.activation_function = activation
        self.nn_activation = activation
        if nn_activation is None:
            self.nn_activation = activation

        self.nn_class = lambda: NNSimpleMLP(self.nn_activation)

        self.layer_1 = T.nn.Linear(n_input, sizes[0])
        self.act_1 = activation()

        self.layer_2 = T.nn.Linear(sizes[0], sizes[1])
        self.act_2 = activation()

        self.layer_3 = T.nn.Linear(sizes[1], n_output)
        self.act_3 = T.nn.LogSoftmax(1)

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = T.tensor(alpha)
            else:
                self.alpha = T.tensor([alpha for _ in range(3)])

        self.layers = [self.layer_1, self.layer_2, self.layer_3]

    def get_closs(self):
        self.closs = T.tensor([0.0]).to(DEVICE)
        for layer in self.layers:
            self.closs += T.norm(layer.bias).sum()
        return self.closs

    def get_nn_net(self) -> NNSimpleMLP:
        return self.nn_class()

    def forward(self, x):

        x = T.flatten(x, start_dim=1)

        x = self.act_1(self.layer_1(x))
        x = self.act_2(self.layer_2(x))
        x = self.act_3(self.layer_3(x))

        return x

    def __repr__(self):
        return "SimpleMLP_{}".format(self.activation_function.__name__)
