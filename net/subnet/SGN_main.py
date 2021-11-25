# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from net.subnet.ms_tcn import MultiScale_TemporalConv
from utils.tools import Identity, ZeroLayer
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math
import paddle
import copy
import paddle.nn.initializer as init
from scipy import special


def zero(x):
    return 0


def iden(x):
    return x


def einsum(x, A):
    """paddle.einsum will be implemented in release/2.2.
    """
    x = x.transpose((0, 2, 3, 1, 4))
    n, c, t, k, v = x.shape
    k2, v2, w = A.shape
    assert (k == k2 and v == v2), "Args of einsum not match!"
    x = x.reshape((n, c, t, k * v))
    A = A.reshape((k * v, w))
    y = paddle.matmul(x, A)
    return y


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


class Graph():
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class ConvTemporalGraphical(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2D(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1))

    def forward(self, x, A):
        assert A.shape[0] == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.reshape((n, self.kernel_size, kc // self.kernel_size, t, v))
        x = einsum(x, A)

        return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SGN(nn.Layer):
    def __init__(self, seg, channel=4, out_channel=32, dim_embed=128, **kwargs):
        super(SGN, self).__init__()
        bias = True
        # self.dim1 = 256
        self.dim_embed = dim_embed
        self.seg = seg
        self.num_joints = 25  # 75/3 = 25
        self.spa_embed = embed(self.num_joints, self.dim_embed, norm=False, bias=bias)  # 64
        self.joint_embed = embed(channel, self.dim_embed, norm=True, bias=bias)
        self.dif_embed = embed(channel, self.dim_embed, norm=True, bias=bias)  # 64

        self.gcn1 = gcn_spa(self.dim_embed * 2, self.dim_embed * 2, bias=bias)
        self.compute_g1 = compute_g_spa(self.dim_embed * 2, self.dim_embed * 2, bias=bias)
        self.ms_tcn1 = MultiScale_TemporalConv(self.dim_embed * 2, self.dim_embed * 2)

        self.gcn2 = gcn_spa(self.dim_embed * 2, self.dim_embed * 2, bias=bias)
        self.compute_g2 = compute_g_spa(self.dim_embed * 2, self.dim_embed * 2, bias=bias)
        self.ms_tcn2 = MultiScale_TemporalConv(self.dim_embed * 2, self.dim_embed * 2)
        self.tem_embed2 = embed(self.seg, self.dim_embed * 2, norm=False, bias=bias)  # 64 * 4 # 和out_channel一致

        self.gcn3 = gcn_spa(self.dim_embed * 2, self.dim_embed * 4, bias=bias)
        self.compute_g3 = compute_g_spa(self.dim_embed * 2, self.dim_embed * 2, bias=bias)
        self.ms_tcn3 = MultiScale_TemporalConv(self.dim_embed * 4, self.dim_embed * 4)

        self.gcn4 = gcn_spa(self.dim_embed * 4, self.dim_embed * 4, bias=bias)
        self.compute_g4 = compute_g_spa(self.dim_embed * 2, self.dim_embed * 4, bias=bias)
        self.tem_embed4 = embed(self.seg, self.dim_embed * 4, norm=False, bias=bias)  # 64 * 4 # 和out_channel一致

        self.ms_tcn4 = MultiScale_TemporalConv(self.dim_embed * 4, out_channel)

    def forward(self, input):
        N, C, T, V = input.shape
        input = input.transpose((0, 2, 3, 1)).reshape((N, T, V * C))
        # Dynamic Representation
        bs, step, dim = input.shape  # （num_skes, max_num_frames, 75)

        self.spa = self.one_hot(bs, self.num_joints, self.seg)  # (N, T, V, V) N = batch_size V = 节点数 T = 帧数
        self.spa = self.spa.transpose((0, 3, 2, 1))  # .to(device)  # (N, V, V, T)
        self.tem = self.one_hot(bs, self.seg, self.num_joints)  # (N, V, T, T)
        self.tem = self.tem.transpose((0, 3, 1, 2))  # .to(device)  # (N, T, V, T)
        channel = dim // 25
        input = input.reshape((bs, step, self.num_joints, channel))  # (N, T, V, C)
        input = input.transpose((0, 3, 2, 1))  # (N, C, V, T)
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]  # 相邻两帧做差，太牛逼了
        dif = paddle.concat([paddle.zeros((bs, dif.shape[1], self.num_joints, 1)), dif],
                            axis=-1)  # (N, 64, V, T) 因为相邻两帧做差，会得到t-1个数据，为了对齐原数据，concatenate一帧0张量
        pos = self.joint_embed(input)  # (N, 64, V, T)
        spa1 = self.spa_embed(self.spa)  # (N, 64, V, T)
        dif = self.dif_embed(dif)  # (N, 64, V, T)
        dy = pos + dif  # (N, 64, V, T)
        # Joint-level Module
        input = paddle.concat([dy, spa1], 1)  # (N, 128, V, T)  concatention joint index
        g1 = self.compute_g1(input)  # (N, T, V, V)  # 邻接矩阵
        g2 = self.compute_g2(input)  # (N, T, V, V)  # 邻接矩阵
        g3 = self.compute_g3(input)  # (N, T, V, V)  # 邻接矩阵
        g4 = self.compute_g4(input)  # (N, T, V, V)  # 邻接矩阵

        input = self.gcn1(input, g1)  # (N, 128, V, T)
        input = self.ms_tcn1(input)

        input = self.gcn2(input, g2)  # (N, 256, V, T)
        tem2 = self.tem_embed2(self.tem)  # (N, 256, V, T)
        input = input + tem2  # (N, 256, V, T)  sum frame index
        input = self.ms_tcn2(input)

        input = self.gcn3(input, g3)  # (N, 256, V, T)
        input = self.ms_tcn3(input)

        input = self.gcn4(input, g4)  # (N, 256, V, T)
        tem4 = self.tem_embed4(self.tem)  # (N, 256, V, T)
        input = input + tem4  # (N, 256, V, T)  sum frame index
        input = self.ms_tcn4(input)
        input = input.transpose((0, 1, 3, 2))

        return input

    def one_hot(self, bs, spa, tem):  # (N, V, T)  (N, T, V)

        # y = paddle.arange(spa).unsqueeze(-1)  # 相当于转置
        # y_onehot = paddle.to_tensor(spa, spa)
        #
        # y_onehot.zero_()
        # y_onehot.scatter_(1, y, 1)  # 构建一个对角线元素为1的矩阵(25, 25), (T, T)
        y_onehot = paddle.eye(spa, spa)  # 构建一个对角线元素为1的矩阵(25, 25), (T, T)
        y_onehot = paddle.unsqueeze(y_onehot, axis=0)
        y_onehot = paddle.unsqueeze(y_onehot, axis=0)  # 增加两个维度

        # y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)  # 增加两个维度
        # y_onehot = y_onehot.repeat(bs, tem, 1, 1)  # 在第一维重复bs倍，第二维重复tem倍，第三维第四维1倍
        y_onehot = paddle.tile(y_onehot, [bs, tem, 1, 1])  # 在第一维重复bs倍，第二维重复tem倍，第三维第四维1倍

        return y_onehot  # (N, T, V, V)  (N, V, T, T)


class norm_data(nn.Layer):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1D(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.shape
        x = x.reshape((bs, -1, step))
        if self.bn:
            x.stop_gradient = False
        x = self.bn(x)
        x = x.reshape((bs, -1, num_joints, step))
        return x


class embed(nn.Layer):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)  # （N, T, V, T）
        return x


class cnn1x1(nn.Layer):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2D(dim1, dim2, kernel_size=1, bias_attr=True)
        self.bn = nn.BatchNorm2D(dim2)

    def forward(self, x):
        x = self.bn(self.cnn(x))
        return x


class gcn_spa(nn.Layer):
    def __init__(self, in_feature, out_feature, residual=True, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2D(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)
        self.w_res = cnn1x1(in_feature, out_feature, bias=bias) if residual else ZeroLayer()
        self.graph = Graph(
            layout='fsd10',
            strategy='spatial',
        )
        A = paddle.to_tensor(self.graph.A, dtype='float32')
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.shape[0]

        self.gcn = ConvTemporalGraphical(in_feature, out_feature, spatial_kernel_size)

    def forward(self, x1, g):  # X: N, C, V ,T
        x = x1.transpose((0, 3, 2, 1))  # x_input gcn1 (N, T, V, 128), gcn2 (N, T, V, 128)
        x_physical_A = x1.transpose((0, 1, 3, 2))  # N, C, T, V
        res = self.w_res(x_physical_A).transpose((0, 1, 3, 2))  # N, C, T, V
        x_physical_A = self.gcn(x_physical_A, self.A).transpose((0, 1, 3, 2))

        x = g.matmul(x)  # x shape gcn1 (N, T, V, 128), gcn2 (N, T, V, 128)
        x = x.transpose((0, 3, 2, 1))  # gcn1 x shape (N, 128, V, T), gcn2 (N, 128, V, T)
        x = self.w(x) + self.w1(x1) + x_physical_A + res  # + res  # gcn1 (N, 128, V, T), gcn2 (N, 256, V, T)
        x = self.relu(self.bn(x))
        return x


class compute_g_spa(nn.Layer):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x1):
        g1 = self.g1(x1).transpose((0, 3, 2, 1))  # (N, C, V, T) -> (N, T, V, C)
        g2 = self.g2(x1).transpose((0, 3, 1, 2))  # (N, C, V, T) -> (N, T, C, V)
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g


if __name__ == '__main__':
    from easydict import EasyDict

    args = EasyDict({'seg': 20, 'batch_size': 2})
    # args = True
    # print(args.train)
    model = SGN(500, channel=128, out_channel=256)
    # print(model)
    bs, step, dim, V = 2, 500, 128, 25  # （num_skes, max_num_frames, 75)
    # [10, 144, 20, 25]
    x = paddle.randn((bs, dim, step, V))
    y = model(x)
    print(y.shape)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

#
# class compute_g_spa(nn.Module):
#     def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
#         super(compute_g_spa, self).__init__()
#         self.dim1 = dim1
#         self.dim2 = dim2
#         self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
#         self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#
#         g1 = self.g1(x1).permute(0, 2, 3, 1).contiguous()
#         g2 = self.g2(x2).permute(0, 2, 1, 3).contiguous()
#         g3 = g1.matmul(g2)
#         g = self.softmax(g3)
#         return g


# if __name__ == "__main__":
#     import time
#     N, C, T, V, M = 6, 2, 50, 17, 1
#     # x = torch.randn(N, C, T, V, M)
#     K = 3
#     # (N, C, V, T)
#     input = torch.randn(N, M, C, T, V).permute(0, 2, 3, 4, 1).contiguous()
#     input = input.view(N, C, T, M * V)
#     input1 = input[:, :, :, :V]
#     input2 = input[:, :, :, V: M * V]
#     compute_g_spa = compute_g_spa(dim1=3, dim2=3)
#     g1 = compute_g_spa(input1, input2)
#     a = g1[N-1, T-1, :, :]
#     print(a.shape)
