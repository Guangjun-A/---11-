import sys

sys.path.insert(0, '')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from net.subnet.ms_tcn import MultiScale_TemporalConv as MS_TCN
from net.subnet.mlp import MLP
from net.subnet.activation import activation_factory
from utils.graph.tools import k_adjacency, normalize_adjacency_matrix


class UnfoldTemporalWindows(nn.Layer):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_sizes=[self.window_size, 1],
                                dilations=[self.window_dilation, 1],
                                paddings=[self.padding, 0],
                                strides=[self.window_stride, 1],
                                )
        '''
        nn.Unfold(kernel_size, dilation=1, padding=0, stride=1):
        将输入形为（N, C, H, W）的数据输出为（N, C * kernel_size, L） L区块的数量
        '''

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)  # 调用了nn.Unfold函数 # （N, C * window_size, T * V）
        # print('x', x.shape)

        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        # （N, C * window_size, T * V） -> (N, C, window_size, T, V) -> (N, C, T, window_size, V)
        x = paddle.reshape(x, (N, C, self.window_size, -1, V))
        x = paddle.transpose(x, [0, 1, 3, 2, 4])
        # x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()

        # (N, C, T, window_size, V) -> (N, C, T, window_size * V)  # 选了T个窗口，每个窗口有window_size帧，每帧有V个节点
        x = paddle.reshape(x, (N, C, -1, self.window_size * V))
        # x = x.view(N, C, -1, self.window_size * V)
        # print(x.shape)
        return x


# a = UnfoldTemporalWindows(3, 1, 1)
# a(torch.randn(32, 16, 100, 25))

class SpatialTemporal_MS_GCN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        # (window_size * V, window_size * V) 并加上自环
        A = self.build_spatial_temporal_graph(A_binary, window_size)  # 创建一张大图

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]  # 计算k=1到k=num_scales的所有值
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])  # 将A_scales的多个元素进行堆叠
            # print('这是', A_scales.shape)
            # 此时A_scales shape 为 (window_size * V * K, window_size * V)
        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = paddle.Tensor(A_scales)
        self.A_scales.stop_gradient = True
        self.V = len(A_binary)

        if use_Ares:
            # self.A_res = nn.initializer.Uniform(-1e-6, 1e-6)(paddle.Tensor(paddle.randn(self.A_scales.shape)))
            # self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)  # 初始化正太分布参数，下界-1x(10^-6)上界1x(10^6)
            self.A_res = paddle.uniform(self.A_scales.shape, np.float32(-1e-6), np.float32(1e-6))

            self.A_res.stop_gradient = False
        else:
            self.A_res = paddle.Tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0  # 定义了一个匿名函数，其实就是x = 0
        elif in_channels == out_channels:
            self.residual = lambda x: x  # 定义了一个匿名函数，其实就是x = x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

    def build_spatial_temporal_graph(self, A_binary, window_size):  # window_size = 3
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()  # 沿x轴将a复制两倍

        return A_large

    def forward(self, x):
        N, C, T, V = x.shape  # T = number of windows  # x 形状 (N, C, T, window_size * V)
        # Build graphs  # A 形状 (window_size * V * K, window_size * V), k为尺度即A^k
        if paddle.device.is_compiled_with_cuda():
            A = self.A_scales.cuda() + self.A_res.cuda()
        else:
            A = self.A_scales + self.A_res

        # Perform Graph Convolution
        res = self.residual(x)
        # agg = torch.einsum('vu,nctu->nctv', A, x)  # agg 形状 (N, C, T, window_size * V * K)
        agg = paddle.matmul(x, A, transpose_y=True)

        # (N, C, T, window_size * V * K) -> (N, C, T, K, window_size * V)
        # agg = agg.view(N, C, T, self.num_scales, V)
        agg = paddle.reshape(agg, (N, C, T, self.num_scales, V))
        # print('agg形状', agg.shape)
        # (N, C, T, K, window_size * V) -> (N, K, C, T, window_size * V) -> (N, K * C, T, window_size * V)
        # agg = agg.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)
        agg = paddle.transpose(agg, (0, 3, 1, 2, 4))
        agg = paddle.reshape(agg, (N, self.num_scales * C, T, V))
        # (N, K * C, T, window_size * V) -> (N, C, T, window_size * V) # 聚合k阶信息
        out = self.mlp(agg)
        out += res
        return self.act(out)  # 激活输出 (N, C, T, window_size * V)
