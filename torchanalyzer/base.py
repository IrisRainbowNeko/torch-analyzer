from typing import List, Tuple

from torch import nn


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


class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze(self, inputs) -> List[Tuple[str, str, nn.Module, List]]:
        raise NotImplementedError

    def _prof_extra_repr(self, info_input, info_output, ori_func):
        sin = info_input + '\n' if info_input is not None else ''
        sout = '\n' + info_output if info_output is not None else ''
        return sin + ori_func() + sout

    def show_with(self, viser, flow, **kwargs):
        viser.show(self.model, flow, **kwargs)
