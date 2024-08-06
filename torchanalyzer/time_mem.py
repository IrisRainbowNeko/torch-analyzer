from typing import List, Tuple

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from .base import ModelAnalyzer, RecordFlowContext, BackRecoder
from .utils import Color, format_memory, format_time, format_percent


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
            args_0 = args[0]
            rec = record_function(f'back:{name}')
            args_0 = BackRecoder.apply(args_0, lambda: rec.__exit__(None, None, None))
            with record_function(self.prefix + name):
                out = ori_forward(args_0, *args[1:], **kwargs)
            return BackRecoder.apply(out, rec.__enter__)

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

    def analyze(self, input_args, input_kwargs=None, prefix='layer:', with_init=True, with_backward=False) -> List[Tuple[str, str, nn.Module, List]]:
        if input_kwargs is None:
            input_kwargs = {}
        if not isinstance(input_args, (tuple, list)):
            input_args = [input_args]

        device = self._get_device(input_args, input_kwargs)
        if with_init:
            self.model.to('cpu')
            with ProfContext(self.model, prefix=prefix, func_name='_apply'):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_to:
                    self.model.to(device)
            self.filtered_events_to = {event.key[len(prefix):]: event for event in prof_to.key_averages() if
                                       event.key.startswith(prefix)}
            self.cpu_mem_all_to = max(1, self.filtered_events_to[''].cpu_memory_usage)
            self.cuda_mem_all_to = max(1, self.filtered_events_to[''].cuda_memory_usage)

        with RecordFlowContext(self.model) as module_flow:
            self.model(*input_args, **input_kwargs)  # warmup

        self._enable_grad(input_args)  # for backward
        self._enable_grad(input_kwargs)  # for backward
        with ProfContext(self.model, prefix=prefix, func_name='forward', with_backward=with_backward):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                out = self.model(*input_args, **input_kwargs)

        if with_backward:
            out :torch.Tensor = out.mean()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof_back:
                out.backward()
            # for event in prof_back.events():
            #     print(event.name, format_memory(abs(event.cuda_memory_usage)), format_time(event.cuda_time))
            #self.events_back = self.summary_events(prof_back.events())
            self.events_back = {event.name[5:]: event for event in prof_back.events() if event.name.startswith('back:')}

            events_back_all = self.events_back['']
            self.back_cpu_time_all = max(1, events_back_all.cpu_time)
            self.back_cuda_time_all = max(1, events_back_all.cuda_time)
            self.back_cpu_mem_all = max(1, events_back_all.cpu_memory_usage)
            self.back_cuda_mem_all = max(1, events_back_all.cuda_memory_usage)

        self.filtered_events = {event.key[len(prefix):]: event for event in prof.key_averages() if
                                event.key.startswith(prefix)}
        self.cpu_time_all = max(1, self.filtered_events[''].cpu_time)
        self.cuda_time_all = max(1, self.filtered_events[''].cuda_time)
        self.cpu_mem_all = max(1, self.filtered_events[''].cpu_memory_usage)
        self.cuda_mem_all = max(1, self.filtered_events[''].cuda_memory_usage)

        flow = self.add_info_to_flow(module_flow.module_record)
        return flow

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
                ('CPU', f'{format_time(event_back.cpu_time)}, {format_percent(event_back.cpu_time / self.back_cpu_time_all)}', Color.CYAN),
                ('CUDA', f'{format_time(event_back.cuda_time)}, {format_percent(event_back.cuda_time / self.back_cuda_time_all)}', Color.GREEN),
            ]

            mem = [
                ('CPU', f'{format_memory(event_back.cpu_memory_usage)}, {format_percent(event_back.cpu_memory_usage / self.back_cpu_mem_all)}', Color.YELLOW),
                ('CUDA', f'{format_memory(event_back.cuda_memory_usage)}, {format_percent(event_back.cuda_memory_usage / self.back_cuda_mem_all)}', Color.MAGENTA),
            ]

            info_dict['Backward Time'] = time
            info_dict['Backward Memory'] = mem

        return info_dict
