# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from paddle import nn
import paddle
import math
from utils.graph.fsd import AdjMatrixGraph

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SGN(nn.Layer):
    def __init__(self, seg, channel=4, out_channel=128, **kwargs):
        super(SGN, self).__init__()
        bias = True
        # self.dataset = dataset
        self.seg = seg
        num_joint = 25        # if args:  # （num_skes, max_num_frames, 150)

        # else:
        #     self.spa = self.one_hot(32 * 5, num_joint, self.seg)
        #     self.spa = self.spa.permute(0, 3, 2, 1)#.to(device)
        #     self.tem = self.one_hot(32 * 5, self.seg, num_joint)
        #     self.tem = self.tem.permute(0, 3, 1, 2)#.to(device)
        self.graph = AdjMatrixGraph()
        self.A = paddle.to_tensor(self.graph.A, dtype='float32')
        self.tem_embed = embed(self.seg, 64, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(48 * channel, 64, norm=True, bias=bias)
        self.dif_embed = embed(48 * channel, 64, norm=True, bias=bias)
        # self.maxpool = nn.AdaptiveAvgPool2D((1, 1))  # 池化后后两维度大小为（1， 1）
        self.compute_g1 = compute_g_spa(128, 128, bias=bias)
        self.gcn1 = gcn_spa(128, 256, bias=bias)
        self.gcn2 = gcn_spa(256, 128, bias=bias)
        self.gcn3 = gcn_spa(128, 64, bias=bias)
        self.gcn4 = gcn_spa(64, 128, bias=bias)
        self.gcn5 = gcn_spa(128, 256, bias=bias)
        self.cnn = local(256, 128, bias=bias)

        # self.fc = nn.Linear(self.dim1 * 2, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2D):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #
        # nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        # nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        # nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        N, C, T, V = input.shape
        input = input.transpose((0, 2, 3, 1)).reshape((N, T, V * C))
        # Dynamic Representation
        bs, step, dim = input.shape  # （num_skes, max_num_frames, 75)
        self.spa = self.one_hot(bs, 25, self.seg)  # (N, T, V, V) N = batch_size V = 节点数 T = 帧数
        self.spa = self.spa.transpose((0, 3, 2, 1))  # .to(device)  # (N, V, V, T)
        self.tem = self.one_hot(bs, self.seg, 25)  # (N, V, T, T)
        self.tem = self.tem.transpose((0, 3, 1, 2))  # .to(device)  # (N, T, V, T)
        num_joints = 25  # 75/3
        input = input.reshape((bs, step, num_joints, -1))  # (N, T, V, C)
        input = input.transpose((0, 3, 2, 1))  # (N, C, V, T)

        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]  # 相邻两帧做差，太牛逼了
        dif = paddle.concat([paddle.zeros((bs, dif.shape[1], num_joints, 1)), dif],
                            axis=-1)  # (N, 64, V, T) 因为相邻两帧做差，会得到t-1个数据，为了对齐原数据，concatenate一帧0张量
        pos = self.joint_embed(input)  # (N, 64, V, T)

        tem1 = self.tem_embed(self.tem)  # (N, 128, V, T)
        spa1 = self.spa_embed(self.spa)  # (N, 64, V, T)
        dif = self.dif_embed(dif)  # (N, 64, V, T)
        dy = pos + dif  # (N, 64, V, T)
        # Joint-level Module
        input = paddle.concat([dy, spa1], 1)  # (N, 128, V, T)  concatention joint index

        g = self.compute_g1(input) + self.A  # (N, T, V, V)  # 邻接矩阵

        input = self.gcn1(input, g)  # (N, 128, V, T)

        input = self.gcn2(input, g)  # (N, 256, V, T)
        input = self.gcn3(input, g)  # (N, 256, V, T)

        # Frame-level Module
        input = input + tem1  # (N, 256, V, T)  sum frame index
        input = self.gcn4(input, g)  # (N, 128, V, T)

        input = self.gcn5(input, g)  # (N, 256, V, T)
        input = self.cnn(input)  # （N, 512, 1, 20）  (N, C, V, T)
        input = input.transpose((0, 1, 3, 2))
        # # Classification
        # output = self.maxpool(input)   # （N, 512, 1, 1）
        # output = paddle.flatten(output, 1)  #
        # output = self.fc(output)

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

    def forward(self, x):
        x = self.cnn(x)
        return x


class local(nn.Layer):
    def __init__(self, dim1=3, dim2=48, bias=False):
        super(local, self).__init__()
        # self.maxpool = nn.AdaptiveMaxPool2D((1, 20))  # 自适应最大池化，输出就是后两维的尺寸大小
        self.cnn1 = nn.Conv2D(dim1, dim1, kernel_size=(1, 1), bias_attr=bias)
        self.bn1 = nn.BatchNorm2D(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2D(dim1, dim2, kernel_size=1, bias_attr=bias)
        self.bn2 = nn.BatchNorm2D(dim2)
        # self.dropout = nn.Dropout2D(0.2)

    def forward(self, x1):
        # x1 = self.maxpool(x1)  # （N, 256, 1, 20）
        x = self.cnn1(x1)  # （N, 256, 1, 20）
        x = self.bn1(x)  # （N, 256, 1, 20）
        x = self.relu(x)  # （N, 256, 1, 20）
        # x = self.dropout(x)  # （N, 256, 1, 20）
        x = self.cnn2(x)  # （N, 512, 1, 20）
        x = self.bn2(x)  # （N, 512, 1, 20）
        x = self.relu(x)  # （N, 512, 1, 20）

        return x  # （N, 512, 1, 20）


class gcn_spa(nn.Layer):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2D(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)

    def forward(self, x1, g):
        x = x1.transpose((0, 3, 2, 1))  # x_input gcn1 (N, T, V, 128), gcn2 (N, T, V, 128)
        x = g.matmul(x)  # x shape gcn1 (N, T, V, 128), gcn2 (N, T, V, 128)
        x = x.transpose((0, 3, 2, 1))  # gcn1 x shape (N, 128, V, T), gcn2 (N, 128, V, T)
        x = self.w(x) + self.w1(x1)  # gcn1 (N, 128, V, T), gcn2 (N, 256, V, T)
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
    model = SGN(2, 20)
    # print(model)
    bs, step, dim, V = 2, 20, 144, 25  # （num_skes, max_num_frames, 75)
    A = paddle.randn((bs, 25, 25))

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
