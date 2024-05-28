from typing import List, Tuple

from torch import nn
from torch.profiler import profile, record_function

from .base import ModelAnalyzer, RecordFlowContext
from .flops_kernel import flops_op_map
from .memops_kernel import memops_op_map
from .utils import Color, format_flops, format_percent


class ProfContext:
    def __init__(self, model, prefix='layer:'):
        self.model = model
        self.prefix = prefix
        self.original_forwards = {}

    def __enter__(self):
        def make_prof_hook(ori_forward, name):
            def prof_hook(*args, **kwargs):
                with record_function(self.prefix + name + '$i'):
                    pass
                outputs = ori_forward(*args, **kwargs)
                with record_function(self.prefix + name + '$o'):
                    pass
                return outputs

            return prof_hook

        for name, module in self.model.named_modules():
            self.original_forwards[name] = module.forward
            module.forward = make_prof_hook(module.forward, name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            module.forward = self.original_forwards[name]


class ModelFlopsAnalyzer(ModelAnalyzer):

    def analyze(self, input_args, input_kwargs=None) -> List[Tuple[str, str, nn.Module, List]]:
        if input_kwargs is None:
            input_kwargs = {}
        if not isinstance(input_args, (tuple, list)):
            input_args = [input_args]

        self.model(*input_args, **input_kwargs)
        self.model: nn.Module
        self.model.named_modules()
        self.model.parameters()

        with (RecordFlowContext(self.model) as module_flow, ProfContext(self.model, prefix=''),
              profile(record_shapes=True, use_cuda=True) as prof):
            out = self.model(*input_args, **input_kwargs)

        self.flops_dict = self.summary_events(prof.events(), flops_op_map)
        self.flops_all = self.flops_dict['']

        self.memops_dict = self.summary_events(prof.events(), memops_op_map)
        self.memops_all = self.memops_dict['']

        self.param_dict = {}
        self.count_params(self.model, self.param_dict)
        self.param_all = self.param_dict['']

        flow = self.add_info_to_flow(module_flow.module_record)
        return flow

    def summary_events(self, events, op_map):
        flops_dict = {}
        blocks = []
        for event in events:
            if event.name.endswith('$i'):
                blocks.append([event.name[:-2], 0])
            elif event.name.endswith('$o'):
                block = blocks.pop()
                flops_dict[block[0]] = block[1]
                if len(blocks) > 0:
                    blocks[-1][1] += block[1]
            elif event.name in op_map:
                flops = op_map[event.name](event.input_shapes, event.concrete_inputs)
                blocks[-1][1] += flops
        return flops_dict

    def count_params(self, module, param_dict, prefix=''):
        param_count = 0
        for name, child in module._modules.items():
            if child is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            param_count += self.count_params(child, param_dict, submodule_prefix)
        param_count += sum(p.numel() for p in module.parameters(recurse=False))
        param_dict[prefix] = param_count
        return param_count

    def add_info_to_flow(self, flow):
        new_flow = []
        for i, item in enumerate(flow):
            name, io_type, module = item
            if io_type == 'i':
                new_flow.append(item + (self.get_flops(name, module),))
            else:
                new_flow.append(item + (None,))
        return new_flow

    def get_flops(self, name, module):
        '''
        :return: [(name:str, info, color:str)]
        '''
        flops = self.flops_dict[name]
        memops = self.memops_dict[name]
        params = self.param_dict[name]

        info_list = [
            # ('Layer', f'{format_time(event.cpu_time)}, {format_percent(event.cpu_time / self.cpu_time_all)}', None, Color.CYAN),
            ('FLOPs', f'{format_flops(flops)}, {format_percent(flops / self.flops_all)}', Color.CYAN),
            ('MemOPs', f'{format_flops(memops)}, {format_percent(memops / self.memops_all)}', Color.MAGENTA),
            ('Parameters', f'{format_flops(params)}, {format_percent(params / self.param_all)}', Color.YELLOW),
        ]

        return {'_one_': info_list}
