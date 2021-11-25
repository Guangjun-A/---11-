import sys

sys.path.insert(0, '')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from utils.graph.tools import k_adjacency, normalize_adjacency_matrix
from net.subnet.mlp import MLP
from net.subnet.activation import activation_factory


class MultiScale_GraphConv(nn.Layer):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
            # print(A_powers.shape)
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)
            # print(A_powers.shape)

        self.A_powers = paddle.Tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            # init = nn.initializer.uniform.Uniform(-1e-6, 1e-6)
            # self.A_res = init(paddle.Tensor(paddle.Tensor(paddle.randn(self.A_powers.shape))))
            # self.A_res = paddle.uniform(self.A_powers.shape, np.float32(-1e-6), np.float32(1e-6))
            # self.A_res = paddle.create_parameter(self.A_powers.shape,dtype=paddle.float32)
            self.A_res = paddle.create_parameter(self.A_powers.shape, dtype=paddle.float32,
                                                 default_initializer=paddle.nn.initializer.Normal())

            # self.A_res.stop_gradient = False
        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        N, C, T, V = x.shape
        # print('x', x.shape)
        if paddle.device.is_compiled_with_cuda():
            self.A_powers = self.A_powers.cuda()
        else:
            self.A_powers = self.A_powers
        A = self.A_powers  # .type(x.dtype)  # .to(x.dtype)  # (Scale * 25, N)
        # print('A', A.shape)
        if self.use_mask:
            A = A + self.A_res  # .type(x.dtype)  # .to(x.dtype)
        # print(x.shape)
        # print(A.shape)
        # support = torch.einsum('vu,nctu->nctv', A, x)  # (N, C, T, V)
        # X N, C, T, V
        # A V, V
        # print(x.shape, A.shape)
        support = paddle.matmul(x, A, transpose_y=True)

        # support = paddle.matmul(A, x, transpose_y=True)

        # support = support.view(N, C, T, self.num_scales, V)  # 按照行优先存储，然后重排
        support = paddle.reshape(support, (N, C, T, self.num_scales, V))

        # (N, C, T, k, V) -> (N, k, C, T, V)
        # support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)

        support = paddle.transpose(support, (0, 3, 1, 2, 4))
        support = paddle.reshape(support, (N, self.num_scales * C, T, V))
        # print('support', support.shape)

        out = self.mlp(support)
        # print('output', out.shape)
        return out  # (N, 96, T, V)


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=2, out_channels=64, A_binary=A_binary)

    msgcn(torch.randn(8, 2, 20, 17))
