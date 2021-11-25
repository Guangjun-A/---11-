import paddle
from paddle import nn
import numpy as np


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Basic_Layer(nn.Layer):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2D(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2D(out_channel)

        self.residual = Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Basic_Tem_Layer(nn.Layer):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Tem_Layer, self).__init__()

        self.conv = nn.Conv2D(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2D(out_channel)
        self.tem_embed = embed(500, out_channel, norm=False, bias=bias)
        # self.residual = nn.Sequential(
        #     nn.Conv2D(in_channel, in_channel, 1, (1, 1), groups=group),
        #     nn.BatchNorm2D(in_channel),
        # )
        self.residual = Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        N, C, T, V = x.shape
        self.tem = self.one_hot(N, T, 25)  # (N, V, T, T)
        self.tem = self.tem.transpose((0, 3, 1, 2))  # .to(device)  # (N, T, V, T)
        tem1 = self.tem_embed(self.tem)  # (N, 128, V, T)
        tem1 = tem1.transpose((0, 1, 3, 2))
        # x = paddle.concat([x, tem1], axis=1)
        res = self.residual(x)
        x = self.bn(self.conv(x) + tem1)
        x = self.act(x + res)
        return x

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


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)  # 对父类初始化

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channel, out_channel, 1),
                nn.BatchNorm2D(out_channel),
            )


class Spatial_Tem_Graph_Layer(Basic_Tem_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Tem_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)  # 对父类初始化

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channel, out_channel, 1),
                nn.BatchNorm2D(out_channel),
            )


class Spatial_Graph_Joint_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Joint_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)  # 对父类初始化

        self.conv = SpatialGraphConvJoint(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channel, out_channel, 1),
                nn.BatchNorm2D(out_channel),
            )


class Spatial_Graph_Joint_Tem_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Joint_Tem_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)  # 对父类初始化

        self.conv = SpatialGraphConvJoint(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channel, out_channel, 1),
                nn.BatchNorm2D(out_channel),
            )


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2D(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0))
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )


class Temporal_Tem_Basic_Layer(Basic_Tem_Layer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Tem_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2D(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0))
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )


class Temporal_Bottleneck_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2D(channel, inner_channel, 1),
            nn.BatchNorm2D(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2D(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0)),
            nn.BatchNorm2D(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2D(channel, inner_channel, 1),
                nn.BatchNorm2D(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2D(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel),
            nn.BatchNorm2D(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1),
            nn.BatchNorm2D(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_Tem_Sep_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, seg=500,residual=True, **kwargs):
        super(Temporal_Tem_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2D(channel, inner_channel, 1),
                nn.BatchNorm2D(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2D(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel),
            nn.BatchNorm2D(inner_channel),
        )

        self.point_conv = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1),
            nn.BatchNorm2D(channel),
        )
        self.stride = stride
        # self.tem_embed = embed(500, channel, norm=False, bias=bias)
        self.tem_embed = embed(seg, channel, norm=False, bias=bias)

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        N, C, T, V = x.shape
        res = self.residual(x)
        self.tem = self.one_hot(N, T // self.stride, 25)  # (N, V, T, T)
        self.tem = self.tem.transpose((0, 3, 1, 2))  # .to(device)  # (N, T, V, T)
        # if T == 500:

        tem1 = self.tem_embed(self.tem)  # (N, 128, V, T)
        tem1 = tem1.transpose((0, 1, 3, 2))
        # tem1 = self.ada_pool(tem1)
        # else:
        #     print('====')
        #     tem1 = 0
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))

        x = self.point_conv(x) + tem1

        return x + res

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


class Temporal_SG_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2D(channel, channel, (temporal_window_size, 1), 1, (padding, 0), groups=channel),
            nn.BatchNorm2D(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2D(channel, inner_channel, 1),
            nn.BatchNorm2D(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1),
            nn.BatchNorm2D(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2D(channel, channel, (temporal_window_size, 1), (stride, 1), (padding, 0), groups=channel),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride, 1)),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Zero_Layer(nn.Layer):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


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


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2D(in_channel, out_channel * self.s_kernel_size, 1)
        self.A = paddle.to_tensor(A[:self.s_kernel_size], dtype=paddle.float32, stop_gradient=True)
        if edge:
            self.edge = paddle.ones_like(A[:self.s_kernel_size], dtype=paddle.float32)

        else:
            self.edge = 1

    def forward(self, x):

        x = self.gcn(x)
        n, kc, t, v = x.shape
        x = x.reshape((n, self.s_kernel_size, kc // self.s_kernel_size, t, v))
        x = einsum(x, self.A * self.edge)  # 'nkctv,kvw->nctw',
        return x


class SpatialGraphConvJoint(nn.Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConvJoint, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        # self.gcn = nn.Conv2D(in_channel, out_channel * self.s_kernel_size, 1)
        self.spa_embed = embed(25, 16, norm=False, bias=False)

        self.gcn = nn.Conv2D(16 + in_channel, out_channel * self.s_kernel_size, 1)
        self.A = paddle.to_tensor(A[:self.s_kernel_size], dtype=paddle.float32, stop_gradient=True)
        if edge:
            self.edge = paddle.ones_like(A[:self.s_kernel_size], dtype=paddle.float32)

        else:
            self.edge = 1

    def forward(self, x):
        N, C, T, V = x.shape
        spa = self.one_hot(N, V, T)  # (N, T, V, V) N = batch_size V = 节点数 T = 帧数
        spa = paddle.transpose(spa, [0, 3, 2, 1])  # (N, V, V, T)
        spa1 = self.spa_embed(spa)  # (N, 16, V, T)
        spa1 = paddle.transpose(spa1, [0, 1, 3, 2])  # (n, 16, t, v)
        x = paddle.concat([x, spa1], axis=1)  # (n, 16+2, t, v)
        x = self.gcn(x)
        n, kc, t, v = x.shape
        x = x.reshape((n, self.s_kernel_size, kc // self.s_kernel_size, t, v))
        x = einsum(x, self.A * self.edge)  # 'nkctv,kvw->nctw',
        return x

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


class embed(nn.Layer):
    def __init__(self, dim=3, dim1=128, stride=1, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias, stride=stride),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias, stride=stride),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)  # （N, T, V, T）
        return x


class norm_data(nn.Layer):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1D(dim * 25)

    def forward(self, x):
        # N, C, V, T
        bs, c, num_joints, step = x.shape
        x = paddle.reshape(x, [bs, -1, step])
        # x = x.view(bs, -1, step)
        x = self.bn(x)
        x = paddle.reshape(x, [bs, -1, num_joints, step])
        # x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class cnn1x1(nn.Layer):
    def __init__(self, dim1=3, dim2=3, stride=1, bias=True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2D(dim1, dim2, kernel_size=1, stride=stride, bias_attr=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x


class TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=9,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2D(branch_channels),
                nn.ReLU(),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2D(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2D(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = nn.ReLU()

    def forward(self, x):
        # Input dim: (N,C,V,T)
        x = x.transpose((0, 1, 3, 2))  # .contiguous()  # N,C,T,V
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        out = self.act(out)
        out = out.transpose((0, 1, 3, 2))  # .contiguous()
        return out
