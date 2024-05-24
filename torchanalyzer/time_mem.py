from typing import List, Tuple, Iterable

import numpy as np
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from .base import ModelAnalyzer, RecordFlowContext
from .utils import Color, format_memory, format_time, format_percent

from torch.autograd import Function

# 方便在backward中记录Module区间
class BackContext(Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.constant = name
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        name = ctx.constant
        with record_function(name):
            return grad_outputs, None

class ProfContext:
    def __init__(self, model, prefix='layer:', func_name='forward', with_backward=False):
        self.model = model
        self.prefix = prefix
        self.func_name = func_name
        self.original_forwards = {}
        self.with_backward = with_backward

        if with_backward:
            self.make_prof_hook = self._make_prof_back_hook
        else:
            self.make_prof_hook = self._make_prof_hook
    def _make_prof_back_hook(self, ori_forward, name):
        def prof_hook(*args, **kwargs):
            with record_function(self.prefix + name):
                args_0 = args[0]
                args_0 = BackContext.apply(args_0, f'{self.prefix}{name}$i')
                out = ori_forward(args_0, *args[1:], **kwargs)
                return BackContext.apply(out, f'{self.prefix}{name}$o')
        return prof_hook

    def _make_prof_hook(self, ori_forward, name):
        def prof_hook(*args, **kwargs):
            with record_function(self.prefix + name):
                return ori_forward(*args, **kwargs)
        return prof_hook

    def __enter__(self):
        for name, module in self.model.named_modules():
            self.original_forwards[name] = getattr(module, self.func_name)
            setattr(module, self.func_name, self.make_prof_hook(getattr(module, self.func_name), name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            setattr(module, self.func_name, self.original_forwards[name])

class ModelTimeMemAnalyzer(ModelAnalyzer):

    def analyze(self, inputs, prefix='layer:', with_init=True, with_backward=True) -> List[Tuple[str, str, nn.Module, List]]:
        if with_init:
            self.model.to('cpu')
            with ProfContext(self.model, prefix=prefix, func_name='_apply'):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_to:
                    self.model.to(inputs.device)
            self.filtered_events_to = {event.key[len(prefix):]: event for event in prof_to.key_averages() if
                                       event.key.startswith(prefix)}
            self.cpu_mem_all_to = max(1, self.filtered_events_to[''].cpu_memory_usage)
            self.cuda_mem_all_to = max(1, self.filtered_events_to[''].cuda_memory_usage)

        with RecordFlowContext(self.model) as module_flow:
            self.model(inputs)  # warmup

        inputs.requires_grad = True # for backward
        with ProfContext(self.model, prefix=prefix, func_name='forward', with_backward=with_backward):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                out = self.model(inputs)

        if with_backward:
            out = out.mean()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_back:
                out.backward()
            # for event in prof_back.events():
            #     print(event.name, format_memory(abs(event.cuda_memory_usage)), format_time(event.cuda_time))
            self.events_back = self.summary_events(prof_back.events())

            events_back_all = self.events_back['']
            self.back_cpu_time_all = max(1, events_back_all[0])
            self.back_cuda_time_all = max(1, events_back_all[1])
            self.back_cpu_mem_all = max(1, events_back_all[2])
            self.back_cuda_mem_all = max(1, events_back_all[3])

        self.filtered_events = {event.key[len(prefix):]: event for event in prof.key_averages() if
                                event.key.startswith(prefix)}
        self.cpu_time_all = max(1, self.filtered_events[''].cpu_time)
        self.cuda_time_all = max(1, self.filtered_events[''].cuda_time)
        self.cpu_mem_all = max(1, self.filtered_events[''].cpu_memory_usage)
        self.cuda_mem_all = max(1, self.filtered_events[''].cuda_memory_usage)

        flow = self.add_info_to_flow(module_flow.module_record)
        return flow

    def summary_events(self, events: Iterable, start_key='$o', end_key='$i', prefix='layer:',
                       keys=('cpu_time', 'cuda_time', 'cpu_memory_usage', 'cuda_memory_usage')):
        event_dict = {}
        blocks = []
        len_prefix = len(prefix)
        for event in events:
            if event.name.endswith(start_key):
                blocks.append([event.name[len_prefix:-2], np.zeros(4, dtype=np.int64)])
            elif event.name.endswith(end_key):
                block = blocks.pop()
                event_dict[block[0]] = block[1]
                if len(blocks) > 0:
                    blocks[-1][1] += block[1]
            elif len(blocks)>0 and event.name.startswith('aten::'):
                blocks[-1][1] += np.array([getattr(event, k) for k in keys], dtype=np.int64)
        return event_dict


    def add_info_to_flow(self, flow):
        new_flow = []
        for i, item in enumerate(flow):
            name, io_type, module = item
            if io_type == 'i':
                new_flow.append(item + (self.get_module_time(name, module),))
            else:
                new_flow.append(item + (None,))
        return new_flow

    def get_module_time(self, name, module):
        '''
        :return: [(name:str, info, color:str)]
        '''
        event = self.filtered_events[name]

        time = [
            ('CPU', f'{format_time(event.cpu_time)}, {format_percent(event.cpu_time / self.cpu_time_all)}', Color.CYAN),
            ('CUDA', f'{format_time(event.cuda_time)}, {format_percent(event.cuda_time / self.cuda_time_all)}', Color.GREEN),
        ]

        mem = [
            ('CPU', f'{format_memory(event.cpu_memory_usage)}, {format_percent(event.cpu_memory_usage / self.cpu_mem_all)}', Color.YELLOW),
            ('CUDA', f'{format_memory(event.cuda_memory_usage)}, {format_percent(event.cuda_memory_usage / self.cuda_mem_all)}', Color.MAGENTA),
        ]

        if hasattr(self, 'filtered_events_to'):
            event_to = self.filtered_events_to[name]
            # mem.append(('CPU(init)', f'{format_memory(event_to.cpu_memory_usage)}, {format_percent(event_to.cpu_memory_usage / self.cpu_mem_all_to)}', Color.YELLOW))
            mem_rate = event_to.cuda_memory_usage / self.cuda_mem_all_to
            mem.append(('CUDA(init)', f'{format_memory(event_to.cuda_memory_usage)}, {format_percent(mem_rate)}', Color.BLUE))

        info_dict = {'Forward Time': time, 'Forward Memory': mem}

        if hasattr(self, 'events_back'):
            event_back = self.events_back[name]

            time = [
                ('CPU', f'{format_time(event_back[0])}, {format_percent(event_back[0] / self.back_cpu_time_all)}', Color.CYAN),
                ('CUDA', f'{format_time(event_back[1])}, {format_percent(event_back[1] / self.back_cuda_time_all)}', Color.GREEN),
            ]

            mem = [
                ('CPU', f'{format_memory(event_back[2])}, {format_percent(event_back[2] / self.back_cpu_mem_all)}', Color.YELLOW),
                ('CUDA', f'{format_memory(event_back[3])}, {format_percent(event_back[3] / self.back_cuda_mem_all)}', Color.MAGENTA),
            ]

            info_dict['Backward Time'] = time
            info_dict['Backward Memory'] = mem

        return info_dict
