#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/23 19:57
# @Author  : Jun
import numpy as np
from paddle.io import Dataset
import paddle
import pandas as pd


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()

        train_data_path = 'G:\花样滑冰\\new_data\\train_data.npy'
        train_label_path = 'G:\花样滑冰\\new_data\\train_label.npy'

        self.train_data = np.load(train_data_path)
        print(len(self.train_data))

        self.label_data = np.load(train_label_path)
        self.N = len(self.train_data)

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.train_data[item], self.label_data[item]


class MytestDataset(Dataset):
    def __init__(self):
        super().__init__()

        train_data_path = 'G:\花样滑冰\\test_A_data.npy'
        train_label_path = 'G:\花样滑冰\\test_A_label.npy'
        self.train_data = np.load(train_data_path)
        self.label_data = np.load(train_label_path)
        self.N = len(self.train_data)

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.train_data[item], self.label_data[item]


dataset = MyDataset()
print(len(dataset))
test_dataset = MytestDataset()
# print(type(dataset))
# print(paddle.io.Subset(dataset, [0, 1]))
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# print(f'train size {train_size}, val_size {val_size}')
# train_dataset, val_dataset = paddle.io.random_split(dataset, [train_size, val_size])
batch_size = 16
train_loader = paddle.io.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False)
# val_loader = paddle.io.DataLoader(
#             dataset=val_dataset,
#             batch_size=1,
#             shuffle=True,
#             drop_last=True)
# test_loader = paddle.io.DataLoader(
#             dataset=test_dataset,
#             batch_size=2,
#             shuffle=True,
#             drop_last=True)
train_labels = {k:0 for k in range(30)}
# val_labels = {k:0 for k in range(10)}

from tqdm import tqdm

labels = []
results = {'video_len': [], 'video_label': []}
for i, (data, label) in enumerate(tqdm(train_loader)):
    # print(label)
    # label = int(label)
    N, C, T, V, M = data.shape
    for n in range(N):
        train_labels[label[n].item()] += 1
        for t in range(T - 1, -1, -1):
            tmp = paddle.sum(data[n, :, t, :, :])
            # print(tmp)
            if tmp > 0:
                # train_labels[label[n].item()] += 1
                train_labels[label[n].item()] += 1
                results['video_label'].append(label[n].item())
                results['video_len'].append(t)
                break
                # print(i*N + n, t)
    #             break
    # c = paddle.sum(data[n, 2, t, :, :])
# la = np.load()
print(train_labels)
print(len(labels))
# print(train_labels)
data_frame = pd.DataFrame(data=results)
data_frame.to_csv('data1.csv')
# print(t, c.item())
# print(data[:, 2, :, :, :])
# print(data.shape)
# for n in range(N):
#     for m in range(M):
#         for t in range(T):
#             if paddle.sum(data[n, :, t:, :, m])== 0:
#                 print(i, t)
# for n in range(N):
#     for i in range(T - 1, -1, -1):
#         tmp = paddle.sum(data[:, :, i, :, :])
#         if tmp > 0:
#             T = i + 1
#         if tmp == 0:
#             print(i)

# for i, (data, label) in enumerate(val_loader):
#     label = int(label)
#     val_labels[label] += 1
#
# print(train_labels)
# print(val_labels)
