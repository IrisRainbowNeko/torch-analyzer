import inspect
from typing import Union, Dict, Iterable

import torch
from torch import nn

from .base import ModelAnalyzer, RecordFlowContext
from .utils import Color


class ModuleIOContext:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_forwards = {}
        self.io_infos = {}

    def _analyze_io(self, inputs: Union[Dict, Iterable]):
        info_dict = {}
        extract_data = lambda v: (list(v.shape), v.max().item(), v.min().item(), v.mean().item())
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            for i, v in enumerate(inputs):
                if isinstance(v, torch.Tensor):
                    info_dict[str(i)] = extract_data(v)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    info_dict[k] = extract_data(v)
        elif isinstance(inputs, torch.Tensor):
            info_dict['0'] = extract_data(inputs)
        return info_dict

    def __enter__(self):
        def make_io_hook(ori_forward, name):
            arg_names = inspect.getfullargspec(ori_forward).args

            def io_hook(*args, **kwargs):
                arg_dict = {(arg_names[i + 1] if i + 1 < len(arg_names) else i): v for i, v in enumerate(args)}  # skip self
                arg_dict.update(kwargs)
                self.io_infos[f'{name}$i'] = self._analyze_io(arg_dict)
                outputs = ori_forward(*args, **kwargs)
                self.io_infos[f'{name}$o'] = self._analyze_io(outputs)
                return outputs

            return io_hook

        for name, module in self.model.named_modules():
            self.original_forwards[name] = module.forward
            module.forward = make_io_hook(module.forward, name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            module.forward = self.original_forwards[name]


class ModelIOAnalyzer(ModelAnalyzer):
    def __init__(self, model):
        super().__init__(model)
        self.info_names = ['shape', 'max', 'min', 'mean']
        self.colors = [Color.CYAN, Color.GREEN, Color.YELLOW, Color.MAGENTA]

    def analyze(self, input_args, input_kwargs=None, prefix='layer:'):
        if input_kwargs is None:
            input_kwargs = {}
        if not isinstance(input_args, (tuple, list)):
            input_args = [input_args]

        with ModuleIOContext(self.model) as module_io, RecordFlowContext(self.model) as module_flow:
            out = self.model(*input_args, **input_kwargs)
        self.module_io = module_io

        flow = self.add_info_to_flow(module_flow.module_record)
        return flow

    def add_info_to_flow(self, flow):
        new_flow = []
        for i, item in enumerate(flow):
            name, io_type, module = item
            new_flow.append(item + (self.get_module_io(name, io_type, module),))
        return new_flow

    def _format_info(self, info):
        if isinstance(info, float):
            return f'{info:.4f}'
        return info

    def get_module_io(self, name, io_type, module):
        '''
        :return: {arg_name: [(name:str, info, color:str)]}
        '''
        info_dict = {
            k: [(iname, self._format_info(item), c_i) for iname, item, c_i in zip(self.info_names, v, self.colors)]
            for k, v in self.module_io.io_infos[f'{name}${io_type}'].items()
        }

        return info_dict
