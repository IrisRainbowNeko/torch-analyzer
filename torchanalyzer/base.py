from contextlib import contextmanager

from torch import nn
from torch.nn.modules.module import _addindent
from prettytable import PrettyTable

def repr_patch(self):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in self._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)

    if getattr(self, '_info_input', None) is not None:
        extra_lines.insert(0, self._info_input)
    if getattr(self, '_info_output', None) is not None:
        child_lines.append(self._info_output)

    lines = extra_lines + child_lines

    main_str = self._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


@contextmanager
def module_print_context():
    repr_raw = nn.Module.__repr__
    nn.Module.__repr__ = repr_patch
    yield None
    nn.Module.__repr__ = repr_raw

class RecordFlowContext:
    def __init__(self, model):
        self.model = model
        self.module_record = {}
        self.original_forwards = {}

    def __enter__(self):
        def make_forward_hook(module, name):
            def forward_hook(*args, ori_forward=module.forward, **kwargs):
                self.module_record[f'{name}$i'] = module
                outputs = ori_forward(*args, **kwargs)
                self.module_record[f'{name}$o'] = module
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

    def analyze(self, inputs):
        raise NotImplementedError

    def _prof_extra_repr(self, info_input, info_output, ori_func):
        sin = info_input + '\n' if info_input is not None else ''
        sout = '\n' + info_output if info_output is not None else ''
        return sin + ori_func() + sout

    def _proc_str_torch(self, info_list, info_names):
        if info_list is None:
            return None
        elif isinstance(info_list[0], str):
            info_list = [x.format(iname+':') for iname, x in zip(info_names[1:], info_list)]
            return ' | '.join(info_list)
        elif isinstance(info_list[0], list):
            return '\n'.join([' | '.join([x.format(iname+':') for iname, x in zip(info_names[1:], info_i)]) for info_i in info_list])

    def show_torch(self, model, info_call_back, **kwargs):
        for name, module in model.named_modules():
            field_names, info_list_input, info_list_output = info_call_back(name, module, **kwargs)
            info_input = self._proc_str_torch(info_list_input, field_names)
            info_output = self._proc_str_torch(info_list_output, field_names)

            module._info_input = info_input
            module._info_output = info_output

        with module_print_context():
            print(model)

    def _proc_str_table(self, table, name, info_list):
        if info_list is None:
            pass
        elif isinstance(info_list[0], str):
            info_list = [x.format('') for x in info_list]
            table.add_row([name, *info_list])
        elif isinstance(info_list[0], list):
            for i, info_i in enumerate(info_list):
                info_i = [x.format('') for x in info_i]
                table.add_row([name, *info_i])

    def show_table(self, model, info_call_back, **kwargs):
        table = PrettyTable()
        flag = True
        for name, module in model.named_modules():
            field_names, info_list_input, info_list_output = info_call_back(name, module, **kwargs)
            if flag:
                table.field_names = field_names
                flag = False

            self._proc_str_table(table, name, info_list_input)
            self._proc_str_table(table, name, info_list_output)

        print(table)

    def _proc_str_flow(self, info_list, info_names, intend):
        flow_str=''
        if info_list is None:
            pass
        elif isinstance(info_list[0], str):
            info_list = [x.format(iname + ':') for iname, x in zip(info_names, info_list)]
            flow_str += ' '*intend+' | '.join(info_list) + '\n'
        elif isinstance(info_list[0], list):
            for i, info_i in enumerate(info_list):
                info_i = [x.format(iname + ':') for iname, x in zip(info_names, info_i)]
                flow_str += ' ' * intend + ' | '.join(info_i) + '\n'
        return flow_str

    def show_flow(self, flow, info_call_back, **kwargs):
        flow_str=''
        intend = -2
        for name, module in flow.items():
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