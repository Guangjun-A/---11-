#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 0:05
# @Author  : Jun
import paddle
import paddle.nn as nn
from net.subnet.SGN import SGN as SGN


class Model(nn.Layer):
    def __init__(self, seg, branch_out_channel=128, input='JVB'):
        super(Model, self).__init__()
        self.branch_out_channel = branch_out_channel
        self.inputs_branch = nn.LayerList()
        self.joint_branch = SGN(seg, channel=4, out_channel=self.branch_out_channel, )
        self.vec_branch = SGN(seg, channel=4, out_channel=self.branch_out_channel, )
        self.bone_branch = SGN(seg, channel=4, out_channel=self.branch_out_channel, )
        self.inputs_branch.append(self.joint_branch)
        self.inputs_branch.append(self.vec_branch)
        self.inputs_branch.append(self.bone_branch)
        # self.angle_branch = SGN(seg, channel=9, out_channel=self.branch_out_channel, )
        # self.main_branch = SGN_main(seg, channel=21, out_channel=128 * 4, dim_embed=64)
        self.maxpool = nn.AdaptiveAvgPool2D((1, 1))  # 池化后后两维度大小为（1， 1）
        self.fc = nn.Linear(128 * 4, 30)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        N, I, C, T, V, M = x.shape  # I为三种输入
        x = x.transpose((1, 0, 5, 2, 3, 4)).reshape((I, N * M, C, T, V))

        # input branches
        x = paddle.concat([branch(x[i]) for i, branch in enumerate(self.input_branches)], axis=1)

        # x = self.main_branch(x)
        x = self.dropout(x)
        x = self.maxpool(x).reshape((N, -1))
        x = self.fc(x)
        return x


if __name__ == "__main__":
    N, I, C, T, V, M = 4, 3, 13, 500, 25, 1
    x = paddle.randn((N, I, C, T, V, M))
    model = Model(T)
    y = model(x)
    print(y.shape)