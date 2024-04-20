import inspect
from typing import Union, Dict, Iterable

import torch
from torch import nn

from .base import ModelAnalyzer
from .utils import Color
from prettytable import PrettyTable


class ModuleIOContext:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_forwards = {}
        self.io_infos = {}

    def _analyze_io(self, inputs: Union[Dict, Iterable]):
        info_dict = {}
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            for i, v in enumerate(inputs):
                if isinstance(v, torch.Tensor):
                    info_dict[str(i)] = (list(v.shape), v.max().item(), v.min().item(), v.mean().item())
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    info_dict[k] = (list(v.shape), v.max().item(), v.min().item(), v.mean().item())
        elif isinstance(inputs, torch.Tensor):
            info_dict['0'] = (list(inputs.shape), inputs.max().item(), inputs.min().item(), inputs.mean().item())
        return info_dict

    def __enter__(self):
        def make_io_hook(ori_forward, name):
            arg_names = inspect.getfullargspec(ori_forward).args

            def io_hook(*args, **kwargs):
                arg_dict = {arg_names[i + 1]: v for i, v in enumerate(args)}  # skip self
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


class ModelIOAnalysis(ModelAnalyzer):
    def __init__(self, model):
        super().__init__(model)
        self.info_names = ['shape', 'max', 'min', 'mean']
        self.colors = [Color.CYAN, Color.GREEN, Color.YELLOW, Color.MAGENTA]

    def analyze(self, inputs, prefix='layer:'):
        self.model(inputs)  # warmup

        with ModuleIOContext(self.model) as module_io:
            out = self.model(inputs)
        self.module_io = module_io

    def _format_info(self, info):
        if isinstance(info, float):
            return f'{info:.4f}'
        return info

    def get_module_io(self, name, module):
        info_list_i = [[f'{Color.RED}{k}{Color.RESET}'] + [f'{c_i}{{}}{self._format_info(item)}{Color.RESET}' for
                                                               item, c_i in zip(v, self.colors)]
                           for k, v in self.module_io.io_infos[f'{name}$i'].items()]
        info_list_o = [[f'{Color.BLUE}{k}{Color.RESET}'] + [f'{c_i}{{}}{self._format_info(item)}{Color.RESET}' for
                                                                 item, c_i in zip(v, self.colors)]
                           for k, v in self.module_io.io_infos[f'{name}$o'].items()]
        info_names = ['Layer', 'arg name']+self.info_names

        return info_names, info_list_i, info_list_o

    def show_table(self, model, info_call_back, **kwargs):
        table = PrettyTable()
        table.field_names = ['Layer', 'arg name']+self.info_names
        for name, info in self.module_io.io_infos.items():
            layer_name, io_type = name.split('$')
            c_type = Color.RED if io_type=='i' else Color.BLUE
            info_list_i = [[f'{c_type}{k}{Color.RESET}'] + [f'{c_i}{{}}{self._format_info(item)}{Color.RESET}' for
                                                 item, c_i in zip(v, self.colors)] for k, v in info.items()]
            self._proc_str_table(table, layer_name, info_list_i)

        print(table)

    def show_flow(self, model, info_call_back, **kwargs):
        flow_str=''
        intend = -2
        for name, info in self.module_io.io_infos.items():
            layer_name, io_type = name.split('$')
            if len(layer_name)==0:
                layer_name='all'
            if io_type == 'i':
                c_type = Color.RED
                intend += 2
                flow_str += ' ' * intend + f'{layer_name}:\n'
            else:
                c_type = Color.BLUE

            info_list_i = [[f'{c_type}{k}{Color.RESET}'] + [f'{c_i}{{}}{self._format_info(item)}{Color.RESET}' for
                                                 item, c_i in zip(v, self.colors)] for k, v in info.items()]
            flow_str += self._proc_str_flow(info_list_i, ['arg name']+self.info_names, intend+2)

            if io_type == 'o':
                intend -= 2

        print(flow_str)