import sys
sys.path.insert(0, '')

import paddle
import paddle.nn as nn

from net.subnet.activation import activation_factory


class TemporalConv(nn.Layer):  # （N, 16, T, V）
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,  # 16
            out_channels,  # 16
            kernel_size=(kernel_size, 1),  # (3, 1)
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
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4, 5, 6],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu'):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2  # 6
        branch_channels = out_channels // self.num_branches  # 96//6 = 16
        # 输入： (N, 96, T, V)
        # Temporal Convolution branches
        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels,  # 96
                    branch_channels,  # 16
                    kernel_size=1,
                    padding=0),  # （N, 16, T, V）
                nn.BatchNorm2D(branch_channels),  # 16
                activation_factory(activation),
                TemporalConv(
                    branch_channels,  # 16
                    branch_channels,  # 16
                    kernel_size=kernel_size,  # 3
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            activation_factory(activation),
            nn.MaxPool2D(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2D(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2D(branch_channels)
        ))

        # Residual connection
        if not residual:  # 步长为2，类似池化
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:  # 写法简洁，优秀
            out = tempconv(x)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        out = self.act(out)
        return out


if __name__ == "__main__":
    mstcn = MultiScale_TemporalConv(96, 96)
    print(mstcn)
    x = paddle.randn((32, 96, 100, 20))
    mstcn.forward(x)
    # for name, param in mstcn.named_parameters():
        # print(f'{name}: {param.numel()}')
    # print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))