#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/10 21:34
# @Author  : Jun
# -*-coding:utf-8-*-
# https://blog.csdn.net/qq_36982160/article/details/80038380
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# labels表示你不同类别的代号，比如这里的demo中有13个类别
# labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'E']
# labels = [i for i in range(61)]
'''
具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个数字）。

同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
你训练好的网络预测出来的预测label。

这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，
然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式
的文件读入的方式，只要你最后将你的真实label和预测label分别保存到y_true和y_pred这两个变量中即可。
'''

# y_true = np.loadtxt('../Data/re_label.txt')
# y_pred = np.loadtxt('../Data/pr_label.txt')

#
# y_true = np.random.choice(labels, 10000, replace=True)
# y_pred = np.random.choice(labels, 10000, replace=True)
# tick_marks = np.array(range(len(labels))) + 1
action = ["sneeze/cough", "staggering", "falling down", "headache",
          "chest pain", " back pain", "neck pain", "nausea/vomiting", "punch/slap", "kicking", "pushing",
          "daily action"]


def plot_confusion_matrix(cm, num_class, title='Confusion Matrix', cmap='Blues'):  # map=plt.cm.binary
    action = ["sneeze/cough", "staggering", "falling down", "headache",
              "chest pain", " back pain", "neck pain", "nausea/vomiting", "punch/slap", "kicking", "pushing",
              "daily action"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(num_class))
    plt.xticks(xlocations, [i for i in range(num_class)], rotation=45, fontsize=6)
    plt.yticks(xlocations, [i for i in range(num_class)], fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#
# cm = confusion_matrix(y_true, y_pred)
# np.set_printoptions(precision=2)
#
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm_normalized)
# plt.figure(figsize=(12, 8), dpi=120)
#
# ind_array = np.arange(len(labels))
#
# # 显示归一化数字
# # x, y = np.meshgrid(ind_array, ind_array)
#
# # for x_val, y_val in zip(x.flatten(), y.flatten()):
# #     c = cm_normalized[y_val][x_val]
# #     if c > 0.01:
# #         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
#
# # offset the tick
# plt.gca().set_xticks(tick_marks, minor=True)
# plt.gca().set_yticks(tick_marks, minor=True)
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().yaxis.set_ticks_position('left')
# # plt.grid(True, which='minor', linestyle='-')
#
# # plt.gcf().subplots_adjust(left=0.001, bottom=0.001, right=0.01, top=0.01,
# #                         wspace=0.01, hspace=0.01)
#
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# # show confusion matrix
# plt.savefig('fig/confusion_matrix.png', format='png')
# plt.show()

def draw_confusion(y_pred, y_true, work_dir, num_class=30, show_num=False, show_fig=False):
    tick_marks = np.array(range(num_class)) + 1  # action # np.array(range(num_class)) + 1
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(num_class)
    res = []
    if show_num:
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            # res.append(cm_normalized[x_val][x_val])
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # print(res)
    # plt.xticks(tick_marks, action)
    # plt.yticks(tick_marks,action)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

    # 显示边框函数
    # plt.grid(True, which='minor', linestyle='-')

    # plt.gcf().subplots_adjust(left=0.001, bottom=0.001, right=0.01, top=0.01,
    #                         wspace=0.01, hspace=0.01)

    plot_confusion_matrix(cm_normalized, num_class, title='Normalized confusion matrix')
    if not os.path.exists(f'{work_dir}/fig'):
        os.makedirs(f'{work_dir}/fig')
    # show confusion matrix

    plt.savefig(f'{work_dir}/fig/confusion_matrix.png', format='png')
    # plt.savefig(f'{work_dir}/fig/confusion_matrix.png', format='png')
    if show_fig:
        plt.show()
    plt.clf()
    plt.close()
# ls = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
# y_true = np.random.choice(ls, 10000, replace=True)
# y_pred = np.random.choice(ls, 10000, replace=True)
# print(y_true.shape)
# draw_confusion(y_pred, y_true, 12)
