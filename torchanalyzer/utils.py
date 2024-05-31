import math
import torch
from enum import Enum


class Color(Enum):
    """
    控制台颜色枚举类
    """
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    def __str__(self):
        return self.value


def format_memory(size_bytes):
    """辅助函数，将字节转换为更高阶的单位（KB, MB, GB）。"""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    sig=''
    if size_bytes<0:
        size_bytes = -size_bytes
        sig='-'
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{sig}{s} {size_name[i]}"


def format_time(time_ns):
    """辅助函数，将时间从纳秒转换为毫秒或秒。"""
    if time_ns < 1000:
        return f"{time_ns} ns"
    elif time_ns < 1e6:
        return f"{time_ns / 1000:.2f} µs"
    elif time_ns < 1e9:
        return f"{time_ns / 1e6:.2f} ms"
    else:
        return f"{time_ns / 1e9:.2f} s"

def format_flops(flops):
    if flops < 1000:
        return f"{flops}"
    elif flops < 1e6:
        return f"{flops / 1000:.2f} K"
    elif flops < 1e9:
        return f"{flops / 1e6:.2f} M"
    elif flops < 1e12:
        return f"{flops / 1e9:.2f} G"
    else:
        return f"{flops / 1e12:.2f} T"

def format_percent(data):
    return f'{data * 100:.2f}%'