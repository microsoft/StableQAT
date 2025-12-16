# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def clipping(x, upper, lower):
    # clip lower
    x = x + F.relu(lower - x)
    # clip upper
    x = x - F.relu(x - upper)

    return x

def phi_function(x_div_s, mi_div_s, alpha):

    # alpha should less than 2 or log will be None
    # alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
    # s = 1/(1-alpha)
    # k = (2/alpha - 1).log() * (1/delta)
    # x = (((x - mi) *k ).tanh()) * s 
    
    alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
    s = 1/(1-alpha)
    k = (2/alpha - 1).log()
    x = (((x_div_s - mi_div_s) * k).tanh()) * s 
    
    return x	

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        return x.round() * 2 - 1
    @staticmethod
    def backward(ctx, g):
        return g 
    
def sgn(x):
    x = RoundWithGradient.apply(x)

    return x

def dequantize(x_div_2, lower_bound_div_s, delta, interval):

    # save mem
    x =  ((x+1)/2 + interval) * delta + lower_bound
    # x = x * delta / 2 + (interval + 0.5) * delta + lower_bound
    
    return x

def LsqBinaryTernaryExtension(input, step_size, num_bits, layerwise, alpha):
        """
        :param input: input to be quantized
        :param step_size: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :param alpha: parameter in DSQ
        :return: quantized output
        """
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=step_size.device).float()

        step_size = torch.where(step_size > eps, step_size, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        
        step_size = ScaleGrad.apply(step_size, grad_scale)
            
        # dsq 
        Qweight = input / step_size
        cur_min = Qn 
        interval = torch.floor(Qweight - cur_min)  
        Qweight = phi_function(Qweight, interval + 0.5 + cur_min, alpha)
        Qweight = sgn(Qweight)
        Qweight = Qweight / 2 + 0.5 + interval + cur_min
        Qweight = clipping(Qweight, Qp, Qn)
        # dequantize
        Qweight = Qweight * step_size
        
        return Qweight


class DSQLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        sine_soft_q=False
    ):
        super(DSQLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.alpha_dsq = nn.Parameter(torch.tensor([0.2]))
        # self.alpha_dsq = torch.tensor([0.2]).cuda()
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        # elif self.w_bits == 2 or self.w_bits == 0:
        #     weight = StretchedElasticQuant(
        #         real_weights,
        #         self.weight_clip_val,
        #         self.w_bits,
        #         self.weight_layerwise,
        #     ).to(input_.dtype)
        elif self.w_bits <= 4:
            weight = LsqBinaryTernaryExtension(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
                # self.sine_soft_q,
                self.alpha_dsq
            ).to(input_.dtype)
        else:
            raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
