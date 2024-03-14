from math import floor
from ast import literal_eval
import torch
from torch import nn
from typing import Callable, List, Optional, Sequence, Tuple, Union



class LinearBlock(nn.Module):
    """
    A custom neural network module consisting of:
    - Linear layer ``nn.Linear``;
    - Normalization layer (as Batch Normalization or Layer Normalization);
    - Nonlinear activation function;
    - Dropout layer.
    
    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the linear layer will not learn an additive bias.
            Defaults to ``True``.
        norm_layer (Callable[..., torch.nn.Module]): Normalization layer.
            If set ``None`` this layer won't be used (Default).
        activation_function (Callable[..., torch.nn.Module]): Nonlinear activation function.
            If set ``None`` will use identity function (Default).
        dropout_prob (float): The probability for the dropout layer. Defaults to 0.0.
        layer_order (Union[str, List[str]]): Representation of the layer's ordering.
            Defaults to "lnad".
    """
    def __init__(self, *args, in_features: int, out_features: int, bias: bool = True,
                 norm_layer: Callable[..., nn.Module] = None,
                 activation_function: Callable[..., nn.Module] = None,
                 dropout_prob: float = 0.0,
                 layers_order: Union[str, List[str]] = "lnad",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        layers_dict = {}

        self.linear_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        layers_dict["l"] = self.linear_layer

        if norm_layer is not None:
            self.norm_layer = norm_layer
            layers_dict["n"] = nn.BatchNorm1d(out_features)
        else:
            layers_dict["n"] = nn.Identity()
            
        if activation_function is not None:
            if activation_function not in torch.nn.modules.activation.__all__:
                raise TypeError("That activation function doesn't exist")
            else:
                layers_dict["a"] = literal_eval("nn." + activation_function + "()")
        else:
            layers_dict["a"] = nn.Identity()

        if dropout_prob == 0.0:
            layers_dict["n"] = nn.Dropout(dropout_prob)
        else:
            layers_dict["n"] = nn.Identity()

        self.out_features = out_features

    def output_shape(self):
        return {"H_out": self.out_features}

    def forward(self, input):
        output = self.linear_layer(input)

        if self.norm:
            output = self.norm_layer(output)

        output = self.activation_layer(output)

        if self.dropout != 0.0:
            output = self.dropout_layer(output)

        return output


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 norm: bool = False,
                 activation_function: str = None,
                 dropout_prob: float = 0.0,
                 maxpool_kernel_size: int = None,
                 maxpool_stride: int = None, maxpool_padding: int = 0, maxpool_dilation: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, bias=bias,
                                      stride=stride, padding=padding, dilation=dilation)

        self.norm = norm
        if norm:
            self.norm_layer = nn.BatchNorm1d(out_channels)

        if activation_function not in torch.nn.modules.activation.__all__:
            raise TypeError("That activation function doesn't exist")
        self.activation_layer = eval("nn." + activation_function + "()")

        self.dropout = dropout_prob
        if dropout_prob != 0.0:
            self.dropout_layer = nn.Dropout1d(dropout_prob)

        self.maxpool_layer = nn.MaxPool1d(kernel_size=maxpool_kernel_size,
                                          stride=maxpool_stride, padding=maxpool_padding, dilation=maxpool_dilation)

        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.maxpool_kernel_size = maxpool_kernel_size
        if maxpool_stride is None:
            self.maxpool_stride = maxpool_kernel_size
        else:
            self.maxpool_stride = maxpool_stride
        self.maxpool_padding = maxpool_padding
        self.maxpool_dilation = maxpool_dilation

    def output_shape(self, L_in):
        L_out = floor((L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        L_out = floor((L_out + 2 * self.maxpool_padding - self.maxpool_dilation * (
                self.maxpool_kernel_size - 1) - 1) / self.maxpool_stride + 1)

        return {"C_out": self.out_channels, "L_out": L_out}

    def forward(self, input):
        output = self.conv_layer(input)

        if self.norm:
            output = self.norm_layer(output)

        output = self.activation_layer(output)

        if self.dropout != 0.0:
            output = self.dropout_layer(output)

        output = self.maxpool_layer(output)

        return output


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, bias: bool = True,
                 stride: int | tuple = 1, padding: int | tuple = 0, dilation: int | tuple = 1,
                 norm: bool = False,
                 activation_function: str = None,
                 dropout: float = 0.0,
                 maxpool_kernel_size: int | tuple = None,
                 maxpool_stride: int | tuple = None, maxpool_padding: int | tuple = 0,
                 maxpool_dilation: int | tuple = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, bias=bias,
                                      stride=stride, padding=padding, dilation=dilation)

        self.norm = norm
        if norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)

        if activation_function not in torch.nn.modules.activation.__all__:
            raise TypeError("That activation function doesn't exist")
        self.activation_layer = eval("nn." + activation_function + "()")

        self.dropout = dropout
        if dropout != 0.0:
            self.dropout_layer = nn.Dropout2d(dropout)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=maxpool_kernel_size,
                                          stride=maxpool_stride, padding=maxpool_padding, dilation=maxpool_dilation)

        self.out_channels = out_channels

        self.kernel_size = list(kernel_size) * 2
        self.stride = list(stride) * 2
        self.padding = list(padding) * 2
        self.dilation = list(dilation) * 2

        self.maxpool_kernel_size = list(maxpool_kernel_size) * 2
        if maxpool_stride is None:
            self.maxpool_stride = list(maxpool_kernel_size) * 2
        else:
            self.maxpool_stride = list(maxpool_stride) * 2
        self.maxpool_padding = list(maxpool_padding) * 2
        self.maxpool_dilation = list(maxpool_dilation) * 2

    def output_shape(self, H_in, W_in):
        H_out = floor(
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        H_out = floor(
            (H_out + 2 * self.maxpool_padding[0] - self.maxpool_dilation[0] * (self.maxpool_kernel_size[0] - 1) - 1) /
            self.maxpool_stride[0] + 1)

        W_out = floor(
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        W_out = floor(
            (W_out + 2 * self.maxpool_padding[1] - self.maxpool_dilation[1] * (self.maxpool_kernel_size[1] - 1) - 1) /
            self.maxpool_stride[1] + 1)

        return {"C_out": self.out_channels, "H_out": H_out, "W_out": W_out}

    def forward(self, input):
        output = self.conv_layer(input)

        if self.norm:
            output = self.norm_layer(output)

        output = self.activation_layer(output)

        if self.dropout != 0.0:
            output = self.dropout_layer(output)

        output = self.maxpool_layer(output)

        return output
