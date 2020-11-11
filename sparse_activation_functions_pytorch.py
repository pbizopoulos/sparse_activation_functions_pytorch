import torch

from torch import nn
from torch.nn import functional as F


def topk_absolutes_1d(x, k):
    primary_extrema = torch.zeros_like(x)
    topk, extrema_indices = torch.topk(abs(x), k)
    return primary_extrema.scatter(-1, extrema_indices, x.gather(-1, extrema_indices))

def extrema_pool_indices_1d(x, kernel_size):
    primary_extrema = torch.zeros_like(x)
    output, extrema_indices = F.max_pool1d_with_indices(abs(x), kernel_size)
    return primary_extrema.scatter(-1, extrema_indices, x.gather(-1, extrema_indices))

def extrema_1d(x, minimum_extrema_distance):
    primary_extrema = torch.zeros_like(x)
    dx = x[:, :, 1:] - x[:, :, :-1]
    dx_padright_greater = F.pad(dx, [0, 1]) > 0
    dx_padleft_less = F.pad(dx, [1, 0]) <= 0
    sign = (1 - torch.sign(x)).bool()
    valleys = dx_padright_greater & dx_padleft_less & sign
    peaks = ~dx_padright_greater & ~dx_padleft_less & ~sign
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for index, (x_, e_) in enumerate(zip(x, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices][:, 0]
        is_secondary_extrema = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for i, extrema_index in enumerate(extrema_indices_sorted):
            if not is_secondary_extrema[i]:
                extrema_indices_r = extrema_indices_sorted >= extrema_index - minimum_extrema_distance
                extrema_indices_l = extrema_indices_sorted <= extrema_index + minimum_extrema_distance
                extrema_indices_m = extrema_indices_r & extrema_indices_l
                is_secondary_extrema = is_secondary_extrema | extrema_indices_m
                is_secondary_extrema[i] = False
        primary_extrema_indices = extrema_indices_sorted[~is_secondary_extrema]
        primary_extrema[index, :, primary_extrema_indices] = x_[0, primary_extrema_indices]
    return primary_extrema

def topk_absolutes_2d(x, k):
    x_flattened = x.view(x.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    topk, extrema_indices = torch.topk(abs(x_flattened), k)
    return primary_extrema.scatter(-1, extrema_indices, x_flattened.gather(-1, extrema_indices)).view(x.shape)

def extrema_pool_indices_2d(x, kernel_size):
    x_flattened = x.view(x.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    output, extrema_indices = F.max_pool2d_with_indices(abs(x), kernel_size)
    return primary_extrema.scatter(-1, extrema_indices[..., 0, 0], x_flattened.gather(-1, extrema_indices[..., 0, 0])).view(x.shape)

def extrema_2d(x, minimum_extrema_distance):
    primary_extrema = torch.zeros_like(x)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx_padright_greater = F.pad(dx, [0, 1, 0, 0]) > 0
    dx_padleft_less = F.pad(dx, [1, 0, 0, 0]) <= 0
    dy_padright_greater = F.pad(dy, [0, 0, 0, 1]) > 0
    dy_padleft_less = F.pad(dy, [0, 0, 1, 0]) <= 0
    sign = (1 - torch.sign(x)).bool()
    valleys_x = dx_padright_greater & dx_padleft_less & sign
    valleys_y = dy_padright_greater & dy_padleft_less & sign
    peaks_x = ~dx_padright_greater & ~dx_padleft_less & ~sign
    peaks_y = ~dy_padright_greater & ~dy_padleft_less & ~sign
    peaks = peaks_x & peaks_y
    valleys = valleys_x & valleys_y
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for index, (x_, e_) in enumerate(zip(x, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices]
        is_secondary_extrema = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for i, (extrema_index_x, extrema_index_y) in enumerate(extrema_indices_sorted):
            if not is_secondary_extrema[i]:
                extrema_indices_r = extrema_indices_sorted[:, 0] >= extrema_index_x - minimum_extrema_distance[0]
                extrema_indices_l = extrema_indices_sorted[:, 0] <= extrema_index_x + minimum_extrema_distance[0]
                extrema_indices_t = extrema_indices_sorted[:, 1] >= extrema_index_y - minimum_extrema_distance[1]
                extrema_indices_b = extrema_indices_sorted[:, 1] <= extrema_index_y + minimum_extrema_distance[1]
                extrema_indices_m = extrema_indices_r & extrema_indices_l & extrema_indices_t & extrema_indices_b
                is_secondary_extrema = is_secondary_extrema | extrema_indices_m
                is_secondary_extrema[i] = False
        primary_extrema_indices = extrema_indices_sorted[~is_secondary_extrema]
        for primary_extrema_index in primary_extrema_indices:
            primary_extrema[index, :, primary_extrema_index[0], primary_extrema_index[1]] = x_[0, primary_extrema_index[0], primary_extrema_index[1]]
    return primary_extrema


class TopKAbsolutes1D(nn.Module):
    '''
    Finds the top-k absolutes in a 1D input.
    Shape:
        - Input: (B, C, W)
        - Output: (B, C, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> k = 10
        >>> topkabsolutes1d = TopKAbsolutes1D(k)
        >>> input = torch.randn(1, 1, 100)
        >>> output = topkabsolutes1d(input)

    '''
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input):
        return topk_absolutes_1d(input, self.k)

class ExtremaPoolIndices1D(nn.Module):
    '''
    Finds the extrema pool indices in a 1D input.
    Shape:
        - Input: (B, C, W)
        - Output: (B, C, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> k = 10
        >>> extremapoolindices1d = ExtremaPoolIndices1D(k)
        >>> input = torch.randn(1, 1, 100)
        >>> output = extremapoolindices1d(input)

    '''
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input):
        return extrema_pool_indices_1d(input, self.k)

class Extrema1D(nn.Module):
    '''
    Finds extrema in a 1D input using a minimum extrema distance.
    Shape:
        - Input: (B, C, W)
        - Output: (B, C, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> minimum_extrema_distance = 10
        >>> extrema1d = Extrema1D(minimum_extrema_distance)
        >>> input = torch.randn(1, 1, 100)
        >>> output = extrema2d(input)

    '''
    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input):
        return extrema_1d(input, self.minimum_extrema_distance)

class TopKAbsolutes2D(nn.Module):
    '''
    Finds the top-k absolutes in a 2D input.
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> k = 10
        >>> topkabsolutes2d = TopKAbsolutes2D(k)
        >>> input = torch.randn(1, 1, 100, 100)
        >>> output = topkabsolutes2d(input)

    '''
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input):
        return topk_absolutes_2d(input, self.k)

class ExtremaPoolIndices2D(nn.Module):
    '''
    Finds the extrema pool indices in a 2D input.
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> k = 10
        >>> extremapoolindices2d = ExtremaPoolIndices2D(k)
        >>> input = torch.randn(1, 1, 100, 100)
        >>> output = extremapoolindices2d(input)

    '''
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input):
        return extrema_pool_indices_2d(input, self.k)

class Extrema2D(nn.Module):
    '''
    Finds extrema in a 2D input using a minimum extrema distance.
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    References:
        -  Related paper:
            https://arxiv.org/abs/1907.06592

    Examples:
        >>> minimum_extrema_distance = [10, 10]
        >>> extrema2d = Extrema2D(minimum_extrema_distance)
        >>> input = torch.randn(1, 1, 100, 100)
        >>> output = extrema2d(input)

    '''
    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input):
        return extrema_2d(input, self.minimum_extrema_distance)
