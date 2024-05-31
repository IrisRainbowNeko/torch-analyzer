from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.autograd import Function


class RecordFlowContext:
    def __init__(self, model):
        self.model = model
        self.module_record: List[Tuple[str, str, nn.Module]] = []
        self.original_forwards = {}

    def __enter__(self):
        def make_forward_hook(module, name):
            def forward_hook(*args, ori_forward=module.forward, **kwargs):
                self.module_record.append((name, 'i', module))
                outputs = ori_forward(*args, **kwargs)
                self.module_record.append((name, 'o', module))
                return outputs

            return forward_hook

        for name, module in self.model.named_modules():
            self.original_forwards[name] = module.forward
            module.forward = make_forward_hook(module, name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            module.forward = self.original_forwards[name]


# 方便在backward中记录Module区间
class BackRecoder(Function):
    @staticmethod
    def forward(ctx, x, func):
        ctx.constant = func
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        func = ctx.constant
        func()
        return grad_outputs, None


class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def _get_device(self, input_args: Tuple, input_kwargs: Dict):
        for arg in input_args:
            if isinstance(arg, torch.Tensor):
                return arg.device

        for k, v in input_kwargs.items():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device('cpu')

    def _enable_grad(self, inputs):
        if isinstance(inputs, (tuple, list)):
            return [(input.requires_grad_(True) if isinstance(input, torch.Tensor) else input) for input in inputs]
        else:
            return {k: (v.requires_grad_(True) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    def analyze(self, input_args, input_kwargs) -> List[Tuple[str, str, nn.Module, List]]:
        raise NotImplementedError

    def _prof_extra_repr(self, info_input, info_output, ori_func):
        sin = info_input + '\n' if info_input is not None else ''
        sout = '\n' + info_output if info_output is not None else ''
        return sin + ori_func() + sout

    def show_with(self, viser, flow, **kwargs):
        viser.show(self.model, flow, **kwargs)
