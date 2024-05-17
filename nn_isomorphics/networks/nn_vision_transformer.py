import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum, register_model
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import (_ovewrite_named_param,
                                       handle_legacy_interface)
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torchvision.transforms._presets import (ImageClassification,
                                             InterpolationMode)
from torchvision.utils import _log_api_usage_once


class NNEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, ):
        super().__init__()

        self.layers = nn.Sequential()

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        x = self.layers[0](input)

        x, _ = self.layers[1](x, x, x, need_weights=False)

        x = self.layers[2](x)
        x = x + input

        y = self.layers[3](x)
        y = self.layers[4](y)

        return x + y


class NNEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers,
        pos_embedding,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = pos_embedding
        self.num_layers = num_layers

        self.layers_dict = nn.ModuleDict()

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        input = input + self.pos_embedding

        x = self.layers_dict['ln'](input)
        for i in range(self.num_layers):
            x = self.layers_dict[f"encoder_layer_{i}"](x)
        x = self.layers_dict['dropout'](x)

        return x

    def add_layer(self, key, module):
        self.layers_dict.update({key: module})


class NNVisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, image_size: int, patch_size: int, hidden_dim: int,
                 class_token):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0,
                      "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.layers_dict = nn.ModuleDict()
        self.class_token = class_token

    def add_layer(self, key, module):
        self.layers_dict.update({key: module})

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.layers_dict['conv_proj'](x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.layers_dict['encoder'](x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.layers_dict['heads'](x)

        return x
