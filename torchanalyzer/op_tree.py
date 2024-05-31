from torch.profiler import profile, record_function, ProfilerActivity

class OpTreeAnalyzer:
    def __init__(self, model):
        self.model = model

    def print_tree(self, events):
        event_stack = []
        pad=''

        for event in events:
            while len(event_stack) > 0 and event.time_range.end >= event_stack[-1].time_range.end:
                event_stack.pop()
                pad = pad[:-2]

            print(f'{pad}{event.name} - CUDA time: {event.cuda_time}, CUDA mem:{event.cuda_memory_usage}/{event.self_cuda_memory_usage}'
                  f'cpu time:{event.cpu_time}, cpu mem:{event.cpu_memory_usage}/{event.self_cpu_memory_usage}')

            pad+='  '
            event_stack.append(event)

    def analyze(self, input_args, input_kwargs=None, with_backward=False, filter=None):
        if input_kwargs is None:
            input_kwargs = {}
        if not isinstance(input_args, (tuple, list)):
            input_args = [input_args]

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            out = self.model(*input_args, **input_kwargs)


        print('Forward pass:')
        self.print_tree(prof.events())

        if with_backward:
            print()
            print('Backward pass:')
            out = out.sum()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof_back:
                out.backward()
            self.print_tree(prof_back.events())