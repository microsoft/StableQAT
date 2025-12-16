

import torch
from torch.optim.optimizer import required
from .base_optimizer import BaseOptimizer


class QOptimizer(BaseOptimizer):
    """
    GETA: General and Efficient Training framework that Automates
    joint structured pruning and quantization.
    """

    def __init__(
        self,
        params,
        variant="adam",
        lr=required,
        lr_quant=required,
        first_momentum=None,
        second_momentum=None,
        dampening=None,
        weight_decay=None,
        additional_defaults=dict(),
    ):      
        self.params = params
        self.variant = variant
        self.lr = lr
        self.lr_quant = lr_quant
        self.first_momentum = first_momentum
        self.second_momentum = second_momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        defaults = dict(
            variant=variant,
            lr=lr,
            lr_quant=lr_quant,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        defaults.update(additional_defaults)
        
        super(QOptimizer, self).__init__(params, defaults)
        
    def step(self, loss=None, closure=None):
        """
        Core function.
        """

        if closure is not None:
            loss = closure()

        self.num_steps += 1
        
        self.compute_grad_variant()

        for group in self.param_groups:
            for p_name, p in zip(group["p_names"], group["params"]):
                if p_name not in group["grad_variant"]:
                    continue
                if "weight_clip_value" in p_name:
                    p.data.add_(group["grad_variant"][p_name], alpha=-group["lr_quant"])
                else:
                    p.data.add_(group["grad_variant"][p_name], alpha=-group["lr"])