from contextlib import contextmanager
from typing import List, Tuple, Callable, Dict

from torch import nn
from torch.nn.modules.module import _addindent
from prettytable import PrettyTable
from .utils import Color


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
        lines = self._info_input.split('\n')
        extra_lines = lines + extra_lines
    if getattr(self, '_info_output', None) is not None:
        lines = self._info_output.split('\n')
        child_lines += lines

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


class Viser:
    def show(self, model: nn.Module, flow: List[Tuple[str, str, nn.Module, Dict]]):
        raise NotImplementedError


class TorchViser(Viser):
    def _proc_str(self, info_dict: Dict[str, List], io_type):
        '''
        :param info_dict: {arg_name: [name, info, color]}
        '''
        if info_dict is None:
            return None
        elif '_one_' in info_dict:
            info_list = [f'{color}{name}:{info}{Color.RESET}' for name, info, color in info_dict['_one_']]
            return ' | '.join(info_list)
        else:
            arg_color = Color.RED if io_type == 'i' else Color.BLUE
            return '\n'.join(
                [f'{arg_color}{arg_name}{Color.RESET}: ' +
                 ' | '.join([f'{color}{name}:{info}{Color.RESET}' for name, info, color in info_list])
                 for arg_name, info_list in info_dict.items()]
            )

    def show(self, model: nn.Module, flow: List[Tuple[str, str, nn.Module, Dict]], **kwargs):
        for name, io_type, module, info_dict in flow:
            if info_dict is not None:
                info = self._proc_str(info_dict, io_type)

                if io_type == 'i':
                    module._info_input = info
                else:
                    module._info_output = info

        with module_print_context():
            print(model)


class TableViser(Viser):
    def _proc_str(self, table, name, info_dict, io_type):
        if info_dict is None:
            pass
        elif '_one_' in info_dict:
            info_list = [f'{color}{info}{Color.RESET}' for iname, info, color in info_dict['_one_']]
            table.add_row([name, *info_list])
        else:
            arg_color = Color.RED if io_type == 'i' else Color.BLUE

            for arg_name, info_list in info_dict.items():
                info_i = [f'{color}{info}{Color.RESET}' for iname, info, color in info_list]
                table.add_row([name, f'{arg_color}{arg_name}{Color.RESET}', *info_i])

    def _get_field_names(self, info_dict):
        if '_one_' in info_dict:
            info_first = info_dict['_one_']
            return ['Layer', *[x[0] for x in info_first]]
        else:
            info_first = next(iter(info_dict.values()))
            return ['Layer', 'arg name', *[x[0] for x in info_first]]

    def show(self, model: nn.Module, flow: List[Tuple[str, str, nn.Module, Dict]], **kwargs):
        table = PrettyTable()

        table.field_names = self._get_field_names(flow[0][3])

        for name, io_type, module, info_dict in flow:
            if info_dict is not None:
                self._proc_str(table, name, info_dict, io_type)

        print(table)


class FlowViser(Viser):
    def _proc_str(self, info_dict, io_type, intend):
        flow_str = ''
        if info_dict is None:
            pass
        elif '_one_' in info_dict:
            info_list = [f'{color}{name}:{info}{Color.RESET}' for name, info, color in info_dict['_one_']]
            flow_str += ' ' * intend + ' | '.join(info_list) + '\n'
        else:
            arg_color = Color.RED if io_type == 'i' else Color.BLUE
            for arg_name, info_list in info_dict.items():
                info_i = [f'{color}{name}:{info}{Color.RESET}' for name, info, color in info_list]
                flow_str += ' ' * intend + f'{arg_color}{arg_name}{Color.RESET}: ' + ' | '.join(info_i) + '\n'
        return flow_str

    def show(self, model: nn.Module, flow: List[Tuple[str, str, nn.Module, Dict]], with_module_name=True, **kwargs):
        flow_str = ''
        intend = -2
        for name, io_type, module, info_dict in flow:
            if info_dict is not None:
                layer_name = 'all' if len(name) == 0 else name
                if io_type == 'i':
                    intend += 2
                    if with_module_name:
                        flow_str += ' ' * intend + f'{layer_name} {Color.GREEN}({type(module).__name__}){Color.RESET}:\n'
                    else:
                        flow_str += ' ' * intend + f'{layer_name}:\n'

                flow_str += self._proc_str(info_dict, io_type, intend + 2)

            if io_type == 'o':
                intend -= 2

        print(flow_str)
