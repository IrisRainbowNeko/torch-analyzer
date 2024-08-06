import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import profile


def tensor_context(func):
    def wrapper(*args, **kwargs):
        args = [(np.array(x, dtype=np.int64) if isinstance(x, list) else x) for x in args]
        kwargs = {k: (np.array(v, dtype=np.int64) if isinstance(v, list) else v) for k, v in kwargs}
        return func(*args, **kwargs)

    return wrapper


# -----------------------model ops-----------------------
def flops_convnd(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_input, s_weight, s_bias, stride, padding, dilation, groups):
        s_out = (s_input[2:] + 2 * padding - s_weight[2:] - (s_weight[2:] - 1) * (dilation - 1)) / stride + 1

        if len(s_bias) == 0:
            flops = s_input[0] * (2 * s_weight[2:].prod() * (s_input[1] / groups) - 1) * s_weight[1] * s_out.prod()
        else:
            flops = s_input[0] * 2 * s_weight[2:].prod() * (s_input[1] / groups) * s_weight[1] * s_out.prod()
        return int(flops)

    return _flops(*input_shapes[:3], *concrete_inputs[3:])


def flops_single_ops(input_shapes, concrete_inputs):
    return np.prod(input_shapes[0])


def flops_pool(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_input, kernel_size, stride, padding, dilation):
        s_out = (s_input[2:] + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride + 1
        return int(kernel_size.prod() * s_out.prod() * s_input[:2].prod())

    return _flops(input_shapes[0], *concrete_inputs[1:5])

def flops_batch_norm(input_shapes, concrete_inputs):
    return (4+5+2)*np.prod(input_shapes[0])

def flops_attention(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_q, s_k, s_v):
        f_mat = 2*s_q.prod()*s_k[2]
        qk = s_k.prod() + f_mat # k/sqrt(h) + qk
        softmax = flops_softmax([[*s_q[:3], s_k[2]]], [None, -1])
        att_v = f_mat # [b,h,lq,lk]x[b,h,lv,d]
        return int(qk+softmax+att_v)

    return _flops(*input_shapes[:3])

# -----------------------mat ops-----------------------
def flops_addmm(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_v, s_mat1, s_mat2, *no_use):
        if len(s_v) > 0:
            return 2 * s_mat1[0] * s_mat1[1] * s_mat2[1] + s_mat1[0] * s_mat2[1]
        else:
            return 2 * s_mat1[0] * s_mat1[1] * s_mat2[1]

    return _flops(*input_shapes[:3])


def flops_mm(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_mat1, s_mat2, *no_use):
        return 2 * np.prod(s_mat1) * s_mat2[1]

    return _flops(*input_shapes)

def flops_bmm(input_shapes, concrete_inputs):
    @tensor_context
    def _flops(s_mat1, s_mat2, *no_use):
        return s_mat1[0] * 2 * s_mat1[1] * s_mat1[2] * s_mat2[2]

    return _flops(*input_shapes)

def flops_elem_ops(input_shapes, concrete_inputs):
    if len(input_shapes[0])>len(input_shapes[1]):
        return int(np.prod(input_shapes[0]))
    else:
        return int(np.prod(input_shapes[1]))

# -----------------------func ops-----------------------
def flops_softmax(input_shapes, concrete_inputs):
    NC = np.prod(input_shapes[0])
    N = NC/input_shapes[0][concrete_inputs[1]]
    return int(3*NC-N) # NC+N(C-1)+NC

def flops_sigmoid(input_shapes, concrete_inputs):
    return 4*np.prod(input_shapes[0])

def flops_gelu(input_shapes, concrete_inputs):
    # tanh take 7 flops
    return (4+7+3)*np.prod(input_shapes[0])

def flops_silu(input_shapes, concrete_inputs):
    return 5*np.prod(input_shapes[0])

def flops_tanh(input_shapes, concrete_inputs):
    return 8*np.prod(input_shapes[0])

# -----------------------tensor ops-----------------------
def flops_sum(input_shapes, concrete_inputs):
    s_in = np.array(input_shapes[0])
    if len(s_in)==0:
        return 1
    s_in[concrete_inputs[1]] -= 1
    return s_in.prod()

def flops_var(input_shapes, concrete_inputs):
    return 4*np.prod(input_shapes[0])

def flops_std(input_shapes, concrete_inputs):
    return 5*np.prod(input_shapes[0])

# -----------------------memory ops-----------------------


flops_op_map = {
    # model ops
    'aten::conv2d': flops_convnd,
    'aten::relu': flops_single_ops,
    'aten::relu_': flops_single_ops,
    'aten::gelu': flops_gelu,
    'aten::gelu_': flops_gelu,
    'aten::silu': flops_silu,
    'aten::silu_': flops_silu,
    'aten::tanh': flops_tanh,
    'aten::tanh_': flops_tanh,
    'aten::max_pool2d': flops_pool,
    'aten::adaptive_avg_pool2d': flops_single_ops,
    'aten::batch_norm': flops_batch_norm,
    'aten::layer_norm': flops_batch_norm,

    'aten::scaled_dot_product_attention': flops_attention,
    # mat ops
    'aten::addmm': flops_addmm,
    'aten::addmm_': flops_addmm,
    'aten::mm': flops_mm,
    'aten::mm_': flops_mm,
    'aten::bmm': flops_bmm,
    'aten::bmm_': flops_bmm,
    # number ops
    'aten::add': flops_elem_ops,
    'aten::add_': flops_elem_ops,
    'aten::mul': flops_elem_ops,
    'aten::mul_': flops_elem_ops,
    'aten::sub': flops_elem_ops,
    'aten::sub_': flops_elem_ops,
    'aten::div': flops_elem_ops,
    'aten::div_': flops_elem_ops,
    'aten::exp': flops_single_ops,
    'aten::exp_': flops_single_ops,
    'aten::log': flops_single_ops,
    'aten::log_': flops_single_ops,
    # tensor ops
    'aten::mean': flops_elem_ops,
    'aten::sum': flops_sum,
    'aten::var': flops_var,
    'aten::std': flops_std,
    'aten::clamp_min': flops_single_ops,
    'aten::clamp_min_': flops_single_ops,
    # func ops
    'aten::_softmax': flops_softmax,
    'aten::sigmoid': flops_sigmoid,
    # memory ops
    #'aten::transpose': flops_single_ops,
    #'aten::reshape': flops_single_ops,
    #'aten::resolve_conj': flops_single_ops,
}

def add_op(op_name, flops_counter):
    flops_counter[op_name] = flops_counter

if __name__ == '__main__':
    from torch import nn
    from torch.profiler import record_function
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10, bias=False)

        def attn(self, query, key, value, attn_bias=None, p=0.1):
            scale = 1.0 / query.shape[-1] ** 0.5
            query = query * scale - 1
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            attn = query @ key.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(-1)
            attn = F.dropout(attn, p)
            attn = attn @ value
            return attn.transpose(1, 2)

        def forward(self, x):
            # x = self.pool(torch.relu(self.conv1(x)))
            # with record_function('pool$i'):
            #     pass
            # x = self.pool(torch.relu(self.conv2(x)))
            # with record_function('pool$o'):
            #     pass
            # x = torch.flatten(x, 1)
            # x = torch.relu(self.fc1(x))
            # x = self.fc2(x)

            with record_function('attn$i'):
                pass
            # x = self.fc2(pp)
            # x = pp@self.fc2.weight.t()
            # x = self.attn(pp, pp, pp)
            x = torch.nn.functional.scaled_dot_product_attention(x, x, x)
            x = x.sum(dim=(1, 2)).tanh()
            with record_function('attn$o'):
                pass

            return x

    from dcn import DeformableConv2d
    class MNISTClassifier(nn.Module):
        def __init__(self, deformable=True):
            super(MNISTClassifier, self).__init__()

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
            conv = nn.Conv2d if deformable == False else DeformableConv2d
            self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

            self.pool = nn.MaxPool2d(2)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)  # [14, 14]
            x = torch.relu(self.conv2(x))
            x = self.pool(x)  # [7, 7]
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = torch.relu(self.conv5(x))
            x = self.gap(x)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x

    import timm

    device = 'cpu'
    #model = SimpleCNN().to(device)
    #model = MNISTClassifier().to(device)
    #model = models.resnet18().cuda()
    model = timm.create_model('vit_base_patch16_224').to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)

    with profile(record_shapes=True, use_cuda=True) as prof:
        model(inputs)

    # 遍历每个事件
    for event in prof.events():
        if event.name.startswith('cuda'):
            continue
        print(
            f"{event.name} - CUDA time: {event.cuda_time}, Input shapes: {event.input_shapes}, input:{event.concrete_inputs}")
        if event.name in flops_op_map:
            print(f"{event.name} - FLOPs: {flops_op_map[event.name](event.input_shapes, event.concrete_inputs)}")
