from torch.profiler import profile, record_function, ProfilerActivity

from .utils import Color, format_memory, format_time, format_percent
from .base import ModelAnalyzer, RecordFlowContext

class ProfContext:
    def __init__(self, model, prefix='layer:'):
        self.model = model
        self.prefix = prefix
        self.original_forwards = {}

    def __enter__(self):
        def make_prof_hook(ori_forward, name):
            def prof_hook(*args, **kwargs):
                with record_function(self.prefix+name):
                    return ori_forward(*args, **kwargs)

            return prof_hook

        for name, module in self.model.named_modules():
            self.original_forwards[name] = module.forward
            module.forward = make_prof_hook(module.forward, name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            module.forward = self.original_forwards[name]


class ModelTimeMemAnalysis(ModelAnalyzer):

    def analyze(self, inputs, prefix='layer:'):
        with RecordFlowContext(self.model) as self.module_flow:
            self.model(inputs)  # warmup

        with ProfContext(self.model, prefix=prefix):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         profile_memory=True) as prof:
                out = self.model(inputs)

        self.filtered_events = {event.key[len(prefix):]: event for event in prof.key_averages() if
                           event.key.startswith(prefix)}
        self.cpu_time_all = self.filtered_events[''].cpu_time
        self.cuda_time_all = self.filtered_events[''].cuda_time
        self.cpu_mem_all = max(1, self.filtered_events[''].cpu_memory_usage)
        self.cuda_mem_all = max(1, self.filtered_events[''].cuda_memory_usage)

    def get_module_time(self, name, module, show_self=False, **kwargs):
        head_names = ['Layer', 'CPU Time', 'CUDA Time', 'CPU Mem', 'CUDA Mem']
        event = self.filtered_events[name]
        info_list = [
            f"{Color.CYAN}{{}}{format_time(event.cpu_time)}, {format_percent(event.cpu_time / self.cpu_time_all)}{Color.RESET}",
            f"{Color.GREEN}{{}}{format_time(event.cuda_time)}, {format_percent(event.cuda_time / self.cuda_time_all)}{Color.RESET}",
            f"{Color.YELLOW}{{}}{format_memory(event.cpu_memory_usage)}, {format_percent(event.cpu_memory_usage / self.cpu_mem_all)}{Color.RESET}",
            f"{Color.MAGENTA}{{}}{format_memory(event.cuda_memory_usage)}, {format_percent(event.cuda_memory_usage / self.cuda_mem_all)}{Color.RESET}",
        ]
        if show_self:
            head_names += ['Self CPU Time', 'Self CUDA Time']
            info_list.append(f"{Color.BLUE}Self CPU Time:{event.self_cpu_time_total} ns{Color.RESET}")
            info_list.append(f"{Color.RED}Self CUDA Time:{event.self_cuda_time_total} ns{Color.RESET}")
        return head_names, info_list, None

