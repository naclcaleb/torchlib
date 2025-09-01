import math
import torch.nn as nn
from torch.nn.modules.utils import _pair

class Module(nn.Module):
    _input_shape: tuple

    def augmented_initialize(self, input_shape):
        self._input_shape = input_shape

    def augmented_forward(self, input):
        if not self._input_shape:
            raise Exception('Module was forwarded before initialization')
        result = self.forward(input)
        return result, self._calculate_output_shape()

    def _calculate_output_shape(self):
        raise NotImplementedError()

class Conv2d(nn.Conv2d, Module):
    def __init__(self, *args, input_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        if input_shape:
            self.augmented_initialize(_pair(input_shape))

    def _calculate_output_shape(self):
        height = math.ceil((self._input_shape[0] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        width = math.ceil((self._input_shape[1] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        return (self.out_channels, height, width)

class Linear(nn.Linear, Module):
    def __init__(self,*, out_features, bias=True):
        self._out_features = out_features
        self._should_have_bias = bias

    def augmented_initialize(self, input_shape):
        self._input_shape = input_shape
        assert len(self._input_shape) == 1, 'Input shape should be 1-dimensional for linear layer'
        super().__init__(self._input_shape[0], self._out_features, self._should_have_bias)
    
    def _calculate_output_shape(self):
        return (self._out_features,)
    
class BatchNorm2d(nn.BatchNorm2d, Module):
    def _calculate_output_shape(self):
        return self._input_shape
    
class ReLU(nn.ReLU, Module):
    def _calculate_output_shape(self):
        return self._input_shape
    
class Sigmoid(nn.Sigmoid, Module):
    def _calculate_output_shape(self):
        return self._input_shape
    
class MaxPool2d(nn.MaxPool2d, Module):
    def _calculate_output_shape(self):
        padding = _pair(self.padding)
        dilation = _pair(self.dilation)
        kernel_size = _pair(self.kernel_size)
        stride = _pair(self.stride)
        height = math.ceil((self._input_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        width = math.ceil((self._input_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        return (self._input_shape[0], height, width)
    
class Flatten(nn.Flatten, Module):
    def _calculate_output_shape(self):
        value = 1
        for dim in self._input_shape:
            value *= dim
        return (value,)

class Sequential(Module):
    def augmented_initialize(self, input_shape):
        self._input_shape = input_shape
        self._initialize_modules()
        
    def _initialize_modules(self):
        current_shape = self._input_shape

        for module in self._all_modules:
            module.augmented_initialize(current_shape)
            current_shape = module._calculate_output_shape()

        self._output_shape = current_shape

    def __init__(self, *args: Module, input_shape=None):

        self._all_modules = args
        
        if input_shape:
            self.augmented_initialize(input_shape)

        super().__init__()

    def _calculate_output_shape(self):
        return self._output_shape


