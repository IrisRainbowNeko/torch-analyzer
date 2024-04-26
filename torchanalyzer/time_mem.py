from typing import List, Tuple

from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from .base import ModelAnalyzer, RecordFlowContext
from .utils import Color, format_memory, format_time, format_percent


class ProfContext:
    def __init__(self, model, prefix='layer:', func_name='forward'):
        self.model = model
        self.prefix = prefix
        self.func_name = func_name
        self.original_forwards = {}

    def __enter__(self):
        def make_prof_hook(ori_forward, name):
            def prof_hook(*args, **kwargs):
                with record_function(self.prefix + name):
                    return ori_forward(*args, **kwargs)

            return prof_hook

        for name, module in self.model.named_modules():
            self.original_forwards[name] = getattr(module, self.func_name)
            setattr(module, self.func_name, make_prof_hook(getattr(module, self.func_name), name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            setattr(module, self.func_name, self.original_forwards[name])


class ModelTimeMemAnalyzer(ModelAnalyzer):

    def analyze(self, inputs, prefix='layer:', with_init=True) -> List[Tuple[str, str, nn.Module, List]]:
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

        with ProfContext(self.model, prefix=prefix):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                out = self.model(inputs)

        self.filtered_events = {event.key[len(prefix):]: event for event in prof.key_averages() if
                                event.key.startswith(prefix)}
        self.cpu_time_all = self.filtered_events[''].cpu_time
        self.cuda_time_all = self.filtered_events[''].cuda_time
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
            ('CUDA', f'{format_time(event.cuda_time)}, {format_percent(event.cuda_time / self.cuda_time_all)}',
             Color.GREEN),
        ]

        mem = [
            ('CPU',
             f'{format_memory(event.cpu_memory_usage)}, {format_percent(event.cpu_memory_usage / self.cpu_mem_all)}',
             Color.YELLOW),
            ('CUDA',
             f'{format_memory(event.cuda_memory_usage)}, {format_percent(event.cuda_memory_usage / self.cuda_mem_all)}',
             Color.MAGENTA),
        ]

        if hasattr(self, 'filtered_events_to'):
            event_to = self.filtered_events_to[name]
            # mem.append(('CPU(init)', f'{format_memory(event_to.cpu_memory_usage)}, {format_percent(event_to.cpu_memory_usage / self.cpu_mem_all_to)}', Color.YELLOW))
            mem.append(('CUDA(init)',
                        f'{format_memory(event_to.cuda_memory_usage)}, {format_percent(event_to.cuda_memory_usage / self.cuda_mem_all_to)}',
                        Color.BLUE))

        return {'Time': time, 'Memory': mem}
