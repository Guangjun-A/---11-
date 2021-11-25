#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/1 19:49
# @Author  : Jun
import numpy as np
import os
import copy
from tqdm import tqdm


def remove_zero_frame(root_path, file_name):
    data = np.load((os.path.join(root_path, file_name)))
    # np.save("data.npy", data)
    N, C, T, V, M = data.shape
    # print(data.shape)
    new_data = np.zeros((N, C, T, V, M))

    for n in tqdm(range(N)):
        if n % 100 == 0:
            miss_frame_file_name = f'{n}_miss_frame_id.txt'
        if not os.path.exists(os.path.join(root_path, "miss_frame2")):
            os.makedirs(os.path.join(root_path, "miss_frame2"))
        f = open(os.path.join(root_path, "miss_frame2", miss_frame_file_name), 'a')
        for m in range(M):
            valid_frame_id_ls = []
            for t in range(T):
                score = data[n, 2, t, :, m]  # (25, ) (V, )
                n_m_t_data = data[n, :2, t, :, m]  # (2, 25) (C, V)
                currect_frame_val = np.sum(abs(n_m_t_data), axis=0)

                # 遇到无效帧，跳过
                if currect_frame_val.any() == 0:
                    miss_batch_frame_id = "file_name: {} sample {:<5d} frame: {:<5d}\n" \
                        .format(file_name, n, t)
                    f.write(miss_batch_frame_id)
                    continue
                # 遇到有效帧，加入列表
                valid_frame_id_ls.append(t)

            # print(valid_frame_id_ls)
            # print(new_data[n, :, :len(valid_frame_id_ls), :, m].shape)
            # print(data[n, :, valid_frame_id_ls, :, m].shape)
            # 有疑惑
            # C, T, V, M
            # data = np.transpose(data, [0, 2, 1, 3, 4])
            # new_data[n][:, :len(valid_frame_id_ls), :] = data[n][:, valid_frame_id_ls, :]
            new_data[n, :, np.arange(len(valid_frame_id_ls)), :, :] = data[n, :, valid_frame_id_ls, :, :]

    np.save(os.path.join(root_path, f"remove_zeros_{file_name}"), new_data)


def shutil_data(path, train_data_file_name, train_label_file_name):
    train_data_path = os.path.join(path, train_data_file_name)  # 'train_data.npy')
    train_label_path = os.path.join(path, train_label_file_name)  # 'train_label.npy')
    # train_label_path = 'G:\花样滑冰\\new_data\\train_label.npy'
    data = np.load(train_data_path)
    label = np.load(train_label_path)
    dataset_size = len(data)
    dataset_id_ls = np.random.permutation([i for i in range(dataset_size)]).tolist()  # 打乱编号
    shutil_train_data = data[dataset_id_ls]
    shutil_label_data = label[dataset_id_ls]
    # np.save(f"G:\花样滑冰\\shutil_train_data.npy", shutil_train_data)
    # np.save(f"G:\花样滑冰\\shutil_train_label.npy", shutil_label_data)
    np.save(os.path.join(path, 'shutil_train_data.npy'), shutil_train_data)
    np.save(os.path.join(path, 'shutil_train_label.npy'), shutil_label_data)


def create_shutil_without_Null_dataset(root_path, train_data_file_name, train_label_file_name, test_file_name):
    shutil_data('data/data104924', train_data_file_name, train_label_file_name)
    remove_zero_frame('data/data104924', 'shutil_train_data.npy')
    # remove_zero_frame(root_path, test_file_name)  # 'test_A_data.npy')
    remove_zero_frame('data/data104925', 'test_A_data.npy')  # 'test_A_data.npy')


if __name__ == "__main__":
    import paddle

    root_path = 'data'
    # C, T, V = 2, 20, 25

    orginal_train_data_file_name = 'train_data.npy'
    orginal_train_label_file_name = 'train_label.npy'
    orginal_test_file_name = 'test_A_data.npy'
    create_shutil_without_Null_dataset(root_path, orginal_train_data_file_name, orginal_train_label_file_name, orginal_test_file_name)



