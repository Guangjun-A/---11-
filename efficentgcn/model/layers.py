import paddle
from paddle import nn


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


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)  # 对父类初始化

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
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
