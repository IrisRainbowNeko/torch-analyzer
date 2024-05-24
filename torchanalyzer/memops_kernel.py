import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import profile


def tensor_context(func):
    def wrapper(*args, **kwargs):
        args = [(np.array(x) if isinstance(x, list) else x) for x in args]
        kwargs = {k: (np.array(v) if isinstance(v, list) else v) for k, v in kwargs}
        return func(*args, **kwargs)

    return wrapper


def flops_single_ops(input_shapes, concrete_inputs):
    return np.prod(input_shapes[0])

def flops_elem_ops(input_shapes, concrete_inputs):
    if len(input_shapes[0])>len(input_shapes[1]):
        return int(np.prod(input_shapes[0]))
    else:
        return int(np.prod(input_shapes[1]))


def flops_expand(input_shapes, concrete_inputs):
    return np.prod(concrete_inputs[1])

memops_op_map = {
    # memory ops
    'aten::transpose': flops_single_ops,
    'aten::transpose_': flops_single_ops,
    'aten::reshape': flops_single_ops,
    'aten::resize': flops_single_ops,
    'aten::resize_': flops_single_ops,
    'aten::resolve_conj': flops_single_ops,
    'aten::select': flops_single_ops,
    'aten::expand': flops_expand,
    'aten::expand_': flops_expand,
    'aten::copy': flops_single_ops,
    'aten::copy_': flops_single_ops,
    'aten::permute': flops_single_ops,
    'aten::permute_': flops_single_ops,

    'aten::zeros': flops_single_ops,
    'aten::zeros_': flops_single_ops,
    'aten::ones': flops_single_ops,
    'aten::ones_': flops_single_ops,
    'aten::zero': flops_single_ops,
    'aten::zero_': flops_single_ops,
    'aten::fill': flops_single_ops,
    'aten::fill_': flops_single_ops,
    'aten::empty': flops_single_ops,
    'aten::empty_': flops_single_ops,
}
