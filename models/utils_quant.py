import math

import torch
import torch.nn as nn

import torch
import math

@torch.compile
def lsq_forward_kernel(input, alpha, qn, qp, num_bits):
    inv_alpha = 1.0 / alpha
    if num_bits == 1:
        q_w = input.sign()
    else:
        q_w = (input * inv_alpha).round_().clamp_(qn, qp)
    return q_w * alpha

@torch.compile
def lsq_backward_kernel(grad_output, input_, alpha, qn, qp, grad_scale, num_bits, layerwise, sine_config):
    inv_alpha = 1.0 / alpha
    v = input_ * inv_alpha
    
    mask_middle = (v >= qn) & (v <= qp)
    v_bar = v.clamp(qn, qp)
    
    if num_bits > 1:
        diff = v_bar.round().sub_(v)
    else:
        diff = v.sign().sub_(v)
    
    diff = torch.where(mask_middle, diff, v_bar)
    grad_alpha_base = diff * grad_output * grad_scale
    
    if layerwise:
        grad_alpha = grad_alpha_base.sum().view_as(alpha)
    else:
        grad_alpha = grad_alpha_base.flatten(1).sum(dim=1).view_as(alpha)

    # Sine Soft Quantization
    if sine_config['enable']:
        PI = math.pi
        coeff = sine_config['amplitude'] * 4.442882938158366 #sqrt(2) * pi
        temp = (v + v.round()).mul_(PI)
        sum_term = torch.zeros_like(temp)
        for idx in range(coeff.shape[0]):
            term = torch.cos(temp * (2 * idx + 1))
            sum_term += (term * coeff[idx])
        sum_term.add_(1.0)
        grad_input = sum_term.reciprocal_().mul_(2.0).sub_(1.0)
        grad_input = grad_input * mask_middle * grad_output
    else:
        grad_input = grad_output * mask_middle

    return grad_input, grad_alpha

class LsqBinaryTernaryExtensionEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, sine_soft_q):
        if num_bits >= 16:
            return input
        
        qn = -1.0 if num_bits <= 1 else float(-(2 ** (num_bits - 1)))
        qp = 1.0 if num_bits <= 1 else float(2 ** (num_bits - 1) - 1)
        
        alpha = torch.clamp(alpha, min=1e-5)
        grad_scale = 1.0 / math.sqrt(input.numel() * max(qp, 1.0))

        output = lsq_forward_kernel(input, alpha, qn, qp, num_bits)
        
        ctx.save_for_backward(input, alpha)
        ctx.num_bits = num_bits
        ctx.layerwise = layerwise
        # ctx.sine_config = {
        #     'enable': sine_soft_q['enable'],
        #     'amplitude': sine_soft_q['amplitude'] if sine_soft_q['enable'] else None,
        # }
        ctx.sine_config = sine_soft_q
        ctx.constants = (grad_scale, qn, qp)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, qn, qp = ctx.constants
        
        grad_input, grad_alpha = lsq_backward_kernel(
            grad_output, input_, alpha, qn, qp, grad_scale, 
            ctx.num_bits, ctx.layerwise, ctx.sine_config
        )

        return grad_input, grad_alpha, None, None, None

class LsqBinaryTernaryExtensionRegular(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, sine_soft_q):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        # ctx.x, ctx.y = (input / alpha), (input / alpha).round()
        # print("sine_soft_q:", sine_soft_q)
        # print("sine_soft_q type:", type(sine_soft_q))
        ctx.sine_soft_q = sine_soft_q
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        if ctx.sine_soft_q['enable']:
            '''
            modified grad_x
            '''
            alpha = ctx.sine_soft_q['amplitude']
            item = torch.pi * (q_w + q_w.round())
            sum = 0
            for idx in range(len(alpha)):
                sum += alpha[idx] * torch.cos((2*idx+1) * item)            
            grad_x = (1 - pow(2, 0.5) * torch.pi * sum) / (1 + pow(2, 0.5) * torch.pi * sum)
            grad_input = indicate_middle * grad_output * grad_x
        else:
            grad_input = indicate_middle * grad_output
        
        return grad_input, grad_alpha, None, None, None

class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        # ctx.x, ctx.y = (torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift), q_w * n_levels
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None



class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        sine_soft_q=dict(),
        efficient=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.sine_soft_q = sine_soft_q
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
        self.efficient = efficient
        self.sine_soft_q = {
            'enable': sine_soft_q['enable'],
            'amplitude': torch.tensor(sine_soft_q['amplitude']).cuda() if sine_soft_q['enable'] else None
        }

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight
        else:
            if self.efficient:
                weight = LsqBinaryTernaryExtensionEfficient.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                    self.sine_soft_q
                ).to(input_.dtype)
            else:
                weight = LsqBinaryTernaryExtensionRegular.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                    self.sine_soft_q
                ).to(input_.dtype)
    
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
