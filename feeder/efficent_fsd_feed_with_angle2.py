#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/9 12:13
# @Author  : Jun
import pickle, logging, numpy as np
from paddle.io import Dataset
import os
import random
from efficentgcn.reader.transformer import pre_normalization
import paddle
from paddle.io import Subset
# from data.remove_zero_frame import create_shutil_without_Null_dataset
from feeder.tools import _rot, _transform
from utils.tools import log_info
from tqdm import tqdm


class FsdDataset(Dataset):
    def __init__(self, data_path, label_path, angle_path, window_size=350, inputs='JVB', center_point=1, PCA=False,
                 debug=False, **kwargs):
        super().__init__()
        self.T = window_size
        self.inputs = inputs
        self.conn = np.array([1, 8, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 18, 19, 14, 11, 22, 11])
        self.window_size = window_size
        # self.conn = connect_joint  # np.array([1, 8, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 18,
        # 19, 14, 11, 22, 11])

        self.data = np.load(data_path)

        self.label = np.load(label_path) if label_path is not None else None
        if 'A' in inputs:
            if PCA:
                self.angle = np.load(angle_path)
                self.angle = self.angle[:, :4, :, :]
            else:
                self.angle = self.angle[:, [1, 3, 4, 8], :, :]
            self.angle = self.normalization(self.angle)
            self.angle = self.angle[:20] if debug else self.angle

        if debug:
            self.data = self.data[:20]
            self.label = self.label[:20] if label_path is not None else None
            # self.seq_len = self.seq_len[:100]

        # 归一化处理
        self.data = self.data[:, :2, :, :, :]
        self.data = pre_normalization(self.data, center_point=center_point)

        # 抽样
        if 'A' in inputs:

            self.data, self.angle = self.sample_data(self.data, self.angle, window_size=window_size)
        else:
            self.data = self.sample_data_for_data(self.data, window_size=window_size)
        N, C, T, V, M = self.data.shape
        self.seq_len = N
        self.T = T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        # data = _transform(data, self.rota) if self.rota != 0 else data
        label = None if self.label is None else self.label[idx].astype('int64')

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        # joint, velocity, bone = self.multi_input(data[:, :self.T, :, :])
        joint, velocity, bone = self.multi_input(data)  # 2, T, V, M
        # joint = np.concatenate([joint, angle], axis=0)
        # velocity = np.concatenate([velocity, angle], axis=0)
        # bone = np.concatenate([bone, angle], axis=0)

        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        if 'A' in self.inputs:
            angle = self.angle[idx]  # 9, T, V, M
            data_new.append(angle)
        data_new = np.stack(data_new, axis=0)

        if label is None:
            return data_new.astype('float32'), 0
        else:
            return data_new, label
        # return data_new, label

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M))
        velocity = np.zeros((C * 2, T, V, M))
        bone = np.zeros((C * 2, T, V, M))
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
            velocity[C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        return joint, velocity, bone

    @staticmethod
    def sample_data_for_one_data(data, window_size, random_pad=False):

        C, T, V, M = data.shape
        # batch_T = FsdDataset.get_frame_num(data)
        T = FsdDataset.get_frame_num_for_one_data(data)
        data_pad = np.zeros((C, window_size, V, M))

        # for n, T in enumerate(batch_T):
        if T == window_size:
            data_pad[:, :, :, :] = data[:, :window_size, :, :]

        elif T < window_size:
            begin = random.randint(0, window_size - T) if random_pad else 0
            # data_pad = np.zeros((C, window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]

        else:
            if random_pad:
                index = np.random.choice(T, window_size, replace=False).astype('int64')
                index = sorted(index)
            else:
                index = np.linspace(0, T - 1, window_size).astype("int64")
            data_pad[:, :, :, :] = data[:, index, :, :]

        return data_pad

    @staticmethod
    def get_frame_num_for_one_data(data):
        C, T, V, M = data.shape
        T_temp = 0
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T_temp = i + 1

        return T_temp

    @staticmethod
    def normalization(data_input):  # N, C, T, V
        data_norm = np.zeros_like(data_input, dtype=float)
        for i, data in enumerate(tqdm(data_input, desc='angle normalization!')):
            data_a = data[0, :, :]
            data_b = data[1, :, :]
            data_c = data[2, :, :]
            # Extracting single channels from 3 channel image
            # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)

            # normalizing per channel data:
            data_a = (data_a - np.min(data_a)) / (np.max(data_a) - np.min(data_a))
            data_b = (data_b - np.min(data_b)) / (np.max(data_b) - np.min(data_b))
            data_c = (data_c - np.min(data_c)) / (np.max(data_c) - np.min(data_c))

            # putting the 3 channels back together:
            data_norm[i, 0, :, :] = data_a
            data_norm[i, 1, :, :] = data_b
            data_norm[i, 2, :, :] = data_c
        return data_norm

    @staticmethod
    def sample_data(data, angle, window_size, random_pad=False):

        N, C, T, V, M = data.shape
        _, C_angle, _, _, _ = angle.shape
        batch_T = FsdDataset.get_frame_num(data)
        data_pad = np.zeros((N, C, window_size, V, M))
        angle_pad = np.zeros((N, C_angle, window_size, V, M))
        for n, T in enumerate(batch_T):
            if T == window_size:
                data_pad[n][:, :, :, :] = data[n][:, :window_size, :, :]
                angle_pad[n][:, :, :, :] = angle[n][:, :window_size, :, :]

            elif T < window_size:
                begin = random.randint(0, window_size - T) if random_pad else 0
                # data_pad = np.zeros((C, window_size, V, M))
                data_pad[n][:, begin:begin + T, :, :] = data[n][:, :T, :, :]
                angle_pad[n][:, begin:begin + T, :, :] = angle[n][:, :T, :, :]

            else:
                if random_pad:
                    index = np.random.choice(T, window_size, replace=False).astype('int64')
                    index = sorted(index)
                else:
                    index = np.linspace(0, T - 1, window_size).astype("int64")
                data_pad[n][:, :, :, :] = data[n][:, index, :, :]
                angle_pad[n][:, :, :, :] = angle[n][:, index, :, :]

        return data_pad, angle_pad
    @staticmethod
    def sample_data_for_data(data, window_size, random_pad=False):

        N, C, T, V, M = data.shape
        batch_T = FsdDataset.get_frame_num(data)
        data_pad = np.zeros((N, C, window_size, V, M))
        for n, T in enumerate(batch_T):
            if T == window_size:
                data_pad[n][:, :, :, :] = data[n][:, :window_size, :, :]

            elif T < window_size:
                begin = random.randint(0, window_size - T) if random_pad else 0
                # data_pad = np.zeros((C, window_size, V, M))
                data_pad[n][:, begin:begin + T, :, :] = data[n][:, :T, :, :]
            else:
                if random_pad:
                    index = np.random.choice(T, window_size, replace=False).astype('int64')
                    index = sorted(index)
                else:
                    index = np.linspace(0, T - 1, window_size).astype("int64")
                data_pad[n][:, :, :, :] = data[n][:, index, :, :]

        return data_pad
    @staticmethod
    def get_frame_num(data):
        N, C, T, V, M = data.shape
        batch_t = []
        for n in range(N):
            for i in range(T - 1, -1, -1):
                tmp = np.sum(data[n, :, i, :, :])
                if tmp > 0:
                    T_temp = i + 1
                    batch_t.append(T_temp)
                    break
        return batch_t


class Cutmix(object):
    """ Cutmix operator
    Args:
        alpha(float): alpha value.
    """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
            'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        """ rand_bbox """
        w = size[3]
        h = size[4]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)
        labels = np.array(labels)

        bs = len(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.shape, lam)
        # 　I N C T V M
        imgs[:, :, :, bbx1:bbx2, bby1:bby2, :] = imgs[idx, :, :, bbx1:bbx2, bby1:bby2, :]

        # lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
        #            (imgs.shape[-2] * imgs.shape[-1]))
        # lams = np.array([lam] * bs, dtype=np.float32)

        return imgs, labels  # list(zip(imgs, labels, labels[idx], lams))


class Mixup(object):
    """
    Mixup operator.
    Args:
        alpha(float): alpha value.
    """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
            'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)
        labels = np.array(labels)
        bs = len(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self.alpha, self.alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return imgs, labels  # list(zip(imgs, labels, labels[idx], lams))


# 更快版本的K折划分
def GetKDataLoader(args, file=None, window_size=350, K=10, debug=False, local_debug2=False, parallel=True):
    # path = '/media/chensiyuan_data/liuguangjun/paddle/MS-G3D/data/new_data'
    path = 'data/new_data'
    # /media/chensiyuan_data/liuguangjun/paddle/MS-G3D/data/new_data/remove_zeros_shutil_train_data.npy
    # orginal_train_data_file_name = 'train_data.npy'
    # orginal_train_label_file_name = 'train_label.npy'
    # orginal_test_file_name = 'test_A_data.npy'
    # data_path = os.path.join(path, 'shutil_train_data.npy')
    # label_path = os.path.join(path, 'shutil_train_label.npy')
    # test_data_path = os.path.join(path, 'test_A_data.npy')

    data_path = os.path.join(path, 'remove_zeros_shutil_train_data.npy')
    label_path = os.path.join(path, 'shutil_train_label.npy')
    test_data_path = os.path.join(path, 'remove_zeros_test_A_data.npy')
    angle_train_path = os.path.join(path, 'x.npy')
    angle_test_path = os.path.join(path, 'x.npy')

    # if not os.path.exists(data_path) or not os.path.exists(label_path) or not os.path.exists(test_data_path):
    #     print(f"create {data_path}...")
    #     print(f"create {label_path}...")
    #     print(f"create {test_data_path}...")
    #     create_shutil_without_Null_dataset(path,
    #                                        orginal_train_data_file_name,
    #                                        orginal_train_label_file_name,
    #                                        orginal_test_file_name)
    if local_debug2:
        data_path = 'G:\花样滑冰\\new_data\\shutil_train_data.npy'
        label_path = 'G:\花样滑冰\\new_data\\shutil_train_label.npy'
        test_data_path = 'G:\花样滑冰\\new_data\\test_A_data.npy'
        angle_train_path = 'G:\花样滑冰\\new_data\\remove_zeros_shutil_train_angle.npy'
        angle_test_path = 'G:\花样滑冰\\new_data\\remove_zeros_test_A_angle.npy'

    log_info(f'K = {K}', file)
    log_info(f'train data path:{data_path}', file)
    log_info(f'train angle path:{angle_train_path}', file)
    log_info(f'label data path:{label_path}', file)
    log_info(f'test data path:{test_data_path}', file)
    log_info(f'test angle path:{angle_test_path}', file)

    dataset = FsdDataset(data_path, label_path, angle_train_path, window_size=window_size,
                         inputs=args.eff_input, debug=debug, PCA=args.pca, center_point=args.center_point)
    dataset_size = dataset.__len__()
    print("train dataset size", dataset_size)

    # 没有旋转矩阵的数据集
    dataset_for_val = dataset
    # FsdDataset(data_path, label_path, window_size=window_size, inputs=args.eff_input, debug=debug)

    val_dataset_size = dataset_for_val.__len__()
    print("val dataset size", val_dataset_size)

    test_dataset = FsdDataset(test_data_path, None, angle_test_path,
                              window_size, inputs=args.eff_input,
                              debug=debug, PCA=args.pca,
                              center_point=args.center_point)

    test_dataset_size = test_dataset.__len__()
    print("test dataset size", test_dataset_size)
    # print(len(dataset))
    # 随机划分成K=10份
    # dataset_id_ls = np.random.permutation([i for i in range(dataset_size)]).tolist()  # 打乱编号
    dataset_id_ls = [i for i in range(dataset_size)]
    # print(dataset_id_ls)
    train_sample_windows = dataset_size // K
    train_sample_nums = []  # 训练集ID编号
    val_sample_nums = []  # 测试集合ID编号

    # 划分
    for k in range(K):
        if k == 0:
            val_sample_nums.append(dataset_id_ls[:(k + 1) * train_sample_windows])
            train_sample_nums.append(dataset_id_ls[(k + 1) * train_sample_windows:])

        elif k == K - 1:
            val_sample_nums.append(dataset_id_ls[k * train_sample_windows:])
            train_sample_nums.append(dataset_id_ls[:k * train_sample_windows])

        else:
            _ = []
            val_sample_nums.append(dataset_id_ls[k * train_sample_windows:(k + 1) * train_sample_windows])
            _.extend(dataset_id_ls[:k * train_sample_windows])
            _.extend(dataset_id_ls[(k + 1) * train_sample_windows:])
            train_sample_nums.append(_)

    # K折交叉
    train_dataloader_ls = []  # 存放K个训练集
    val_dataloader_ls = []  # 存放K个验证集
    for train_set, val_set in zip(train_sample_nums, val_sample_nums):
        # print(val_set, train_set)
        train_dataloader = paddle.io.DataLoader(dataset=Subset(dataset, train_set),
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                # collate_fn=Cutmix(), # Cutmix 未使用
                                                drop_last=True,
                                                num_workers=args.workers)

        # ))
        val_dataloader = paddle.io.DataLoader(dataset=Subset(dataset_for_val, val_set),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=args.workers)

        train_dataloader_ls.append(train_dataloader)
        val_dataloader_ls.append(val_dataloader)
        # ))
    # if args.phase == "test":

    test_dataloader = paddle.io.DataLoader(dataset=test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=args.workers
                                           )
    return train_dataloader_ls, val_dataloader_ls, test_dataloader
    # else:
    #     return train_dataloader_ls, val_dataloader_ls, None


if __name__ == "__main__":
    train_data_path = 'G:\花样滑冰\\train_data.npy'
    train_label_path = 'G:\花样滑冰\\train_label.npy'

    import argparse
    from tqdm import tqdm
    import pandas as pd
    from utils.tools import import_class
    from net.ctrgcn import Model

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_format', default="fsd_all_npy")
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--workers', default=8)
    parser.add_argument('--rota', default=0)
    parser.add_argument('--bone', default=0)
    parser.add_argument('--eff_input', default='JVB')

    args = parser.parse_args()
    print(type(args))
    # train_loader, val_loader, test_loader = GetDataLoader(args, debug=True)
    # _, _, test_dataloader = GetKDataLoader(args,debug=True, local_debug2=True)
    # args.bone = True
    train_dataloader_ls, val_dataloader_ls, test_dataloader = GetKDataLoader(args, debug=True, local_debug2=True)
    # I, N, C, T, V, M = 5, 30, 2, 20, 25, 2
    labels = {k: [0 for i in range(30)] for k in range(10)}
    for k, (train_dataloader, test_dataloader) in enumerate(
            tqdm(zip(train_dataloader_ls, val_dataloader_ls), desc=f"{0} fold test ")):
        for data, y in train_dataloader:
            print(data.shape)
    #         y = int(y)
    #         labels[k][y] += 1
    #
    # frame = pd.DataFrame(data=labels)
    # frame.to_csv('k_data.csv')
    # for k, v in labels.items():
    #     print(k, '  ', v)
