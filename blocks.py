from math import floor
import torch
from torch import nn
from typing import Callable, List, Union, Tuple


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
                 norm_layer: Callable[..., nn.Module] = None, norm_layer_args: dict = {},
                 activation_function: Callable[..., nn.Module] = None, activation_function_args: dict={},
                 dropout_prob: float = 0.0,
                 layers_order: Union[str, List[str]] = "lnad",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_dict = {"l": None, "n": None, "a": None, "d": None}

        self.layers_dict["l"] = nn.Linear(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)

        if norm_layer is not None:
            self.layers_dict["n"] = norm_layer(**norm_layer_args)

        if activation_function is not None:
            self.layers_dict["a"] = activation_function(**activation_function_args)

        if dropout_prob != 0.0:
            self.layers_dict["d"] = nn.Dropout(dropout_prob)

        self.out_features = out_features
        
        self.sequential = nn.Sequential()
        for layer in layers_order:
            if self.layers_dict[layer] is not None:            
                self.sequential = self.sequential.append(self.layers_dict[layer])

    def output_shape(self):
        return {"H_out": self.out_features}

    def forward(self, input):
        return self.sequential(input)


class Conv1dBlock(nn.Module):
    """
    A custom neural network module consisting of:
    - One dimensional convulational layer ``nn.Conv1d``;
    - Normalization layer (as Batch Normalization or Layer Normalization);
    - Nonlinear activation function;
    - Dropout layer.
    - Pooling layer (maxpool or averagepool).

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
        pooling_layer (Callable[..., torch.nn.Module]): Pooling layer.
            If set ``None`` this layer won't be used (Default).
        layer_order (Union[str, List[str]]): Representation of the layer's ordering.
            Defaults to "cnadp".
    """
    def __init__(self, *args, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 norm_layer: Callable[..., nn.Module] = None, norm_layer_args: dict = {},
                 activation_function: Callable[..., nn.Module] = None, activation_function_args: dict = {},
                 dropout_prob: float = 0.0,
                 pooling_layer: Callable[..., nn.Module] = None, pooling_layer_args: dict = {},
                 layers_order: Union[str, List[str]] = "cnadp",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers_dict = {"c": None, "n": None, "a": None, "d": None, "p": None}

        self.layers_dict["c"] = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        if norm_layer is not None:
            self.layers_dict["n"] = norm_layer(**norm_layer_args)

        if activation_function is not None:
            self.layers_dict["a"] = activation_function(**activation_function_args)

        if dropout_prob != 0.0:
            self.layers_dict["d"] = nn.Dropout(dropout_prob)

        if pooling_layer is not None:
            self.layers_dict["p"] = pooling_layer(**pooling_layer_args)
            
            self.pool_kernel_size = self.layers_dict["p"].kernel_size
            if isinstance(self.pool_kernel_size, tuple):
                self.pool_kernel_size = self.pool_kernel_size[0]
                
            self.pool_stride = self.layers_dict["p"].stride
            if isinstance(self.pool_stride, tuple):
                self.pool_stride = self.pool_stride[0]
                
            try:
                self.pool_padding = self.layers_dict["p"].padding
                if isinstance(self.pool_padding, tuple):
                    self.pool_padding = self.pool_padding[0]
            except AttributeError:
                self.pool_padding = 0
                
            try:
                self.pool_dilation = self.layers_dict["p"].dilation
                if isinstance(self.pool_dilation, tuple):
                    self.pool_dilation = self.pool_dilation[0]
            except AttributeError:
                self.pool_dilation = 1

        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.sequential = nn.Sequential()
        for layer in layers_order:
            if self.layers_dict[layer] is not None:
                self.sequential = self.sequential.append(self.layers_dict[layer])

    def output_shape(self, L_in):
        L_out = floor((L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        if self.layers_dict["p"] is not None:
            L_out = floor((L_out + 2 * self.pool_padding - self.pool_dilation * (
                    self.pool_kernel_size - 1) - 1) / self.pool_stride + 1)

        return {"C_out": self.out_channels, "L_out": L_out}

    def forward(self, input):
        return self.sequential(input)
    

class Conv2dBlock(nn.Module):
    """
    A custom neural network module consisting of:
    - Two dimensional convulational layer ``nn.Conv1d``;
    - Normalization layer (as Batch Normalization or Layer Normalization);
    - Nonlinear activation function;
    - Dropout layer.
    - Pooling layer (maxpool or averagepool).

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
        pooling_layer (Callable[..., torch.nn.Module]): Pooling layer.
            If set ``None`` this layer won't be used (Default).
        layer_order (Union[str, List[str]]): Representation of the layer's ordering.
            Defaults to "cnadp".
    """
    def __init__(self, *args, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], bias: bool = True,
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1,
                 norm_layer: Callable[..., nn.Module] = None, norm_layer_args: dict = {},
                 activation_function: Callable[..., nn.Module] = None, activation_function_args: dict = {},
                 dropout_prob: float = 0.0,
                 pooling_layer: Callable[..., nn.Module] = None, pooling_layer_args: dict = {},
                 layers_order: Union[str, List[str]] = "cnadp",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.layers_dict = {"c": None, "n": None, "a": None, "d": None, "p": None}

        self.layers_dict["c"] = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, bias=bias,
            stride=stride, padding=padding, dilation=dilation)

        if norm_layer is not None:
            self.layers_dict["n"] = norm_layer(**norm_layer_args)

        if activation_function is not None:
            self.layers_dict["a"] = activation_function(**activation_function_args)

        if dropout_prob != 0.0:
            self.layers_dict["d"] = nn.Dropout2d(dropout_prob)

        if pooling_layer is not None:
            self.layers_dict["p"] = pooling_layer(**pooling_layer_args)
            
            self.pool_kernel_size = self.layers_dict["p"].kernel_size
            if isinstance(self.pool_kernel_size, tuple):
                self.pool_kernel_size = self.pool_kernel_size[0]
                
            self.pool_stride = self.layers_dict["p"].stride
            if isinstance(self.pool_stride, tuple):
                self.pool_stride = self.pool_stride[0]
                
            try:
                self.pool_padding = self.layers_dict["p"].padding
                if isinstance(self.pool_padding, tuple):
                    self.pool_padding = self.pool_padding[0]
            except AttributeError:
                self.pool_padding = 0
                
            try:
                self.pool_dilation = self.layers_dict["p"].dilation
                if isinstance(self.pool_dilation, tuple):
                    self.pool_dilation = self.pool_dilation[0]
            except AttributeError:
                self.pool_dilation = 1

        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.sequential = nn.Sequential()
        for layer in layers_order:
            if self.layers_dict[layer] is not None:
                self.sequential = self.sequential.append(self.layers_dict[layer])

    def output_shape(self, L_in):
        L_out = floor((L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1)
        if self.layers_dict["p"] is not None:
            L_out = floor((L_out + 2 * self.pool_padding - self.pool_dilation * (
                    self.pool_kernel_size - 1) - 1) / self.pool_stride + 1)

        return {"C_out": self.out_channels, "L_out": L_out}

    def forward(self, input):
        return self.sequential(input)



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
