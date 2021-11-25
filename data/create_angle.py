#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/28 19:44
# @Author  : Jun
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select the GPU
from tqdm import tqdm
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from numpy import linalg as LA
import paddle.nn as nn
import paddle


def angle(v1, v2):
    v1_n = v1 / LA.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / LA.norm(v2, axis=1, keepdims=True)
    dot_v1_v2 = v1_n * v2_n
    dot_v1_v2 = 1.0 - np.sum(dot_v1_v2, axis=1)
    dot_v1_v2 = np.nan_to_num(dot_v1_v2)
    return dot_v1_v2


fsd_skeleton_bone_pairs = ((1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                           (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                           (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                           (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                           (21, 14), (19, 14), (20, 19))

fsd_skeleton_orig_bone_pairs = {1: 8, 0: 1, 15: 0, 17: 15,
                                16: 0, 18: 16, 5: 1, 6: 5, 7: 6, 2: 1, 3: 2, 4: 3, 9: 8,
                                10: 9, 11: 10, 24: 11, 22: 11, 23: 22, 12: 8, 13: 12,
                                14: 13, 21: 14, 19: 14, 20: 19}

#
# ntu_skeleton_bone_pairs = (
#     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
#     (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
#     (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
#     (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
# )
#
# ntu_skeleton_orig_bone_pairs = {
#     25: 12, 24: 25, 23: 8, 21: 21, 22: 23, 20: 19, 19: 18, 18: 17,
#     17: 1, 16: 15, 15: 14, 14: 13, 13: 1, 12: 11, 11: 10, 10: 9,
#     9: 21, 8: 7, 7: 6, 6: 5, 5: 21, 4: 3, 3: 21, 2: 21, 1: 2
# }

# ntu_bone_adj = {
#     25: 12,
#     24: 12,
#     12: 11,
#     11: 10,
#     10: 9,
#     9: 21,
#     21: 21,
#     5: 21,
#     6: 5,
#     7: 6,
#     8: 7,
#     22: 8,
#     23: 8,
#     3: 21,
#     4: 3,
#     2: 21,
#     1: 2,
#     17: 1,
#     18: 17,
#     19: 18,
#     20: 19,
#     13: 1,
#     14: 13,
#     15: 14,
#     16: 15
# }
# ntu_bone_angle_pairs = {
#     25: (24, 12),
#     24: (25, 12),
#     12: (24, 25),
#     11: (12, 10),
#
#     10: (11, 9),
#     9: (10, 21),
#     21: (9, 5),
#     5: (21, 6),
#     6: (5, 7),
#     7: (6, 8),
#     8: (23, 22),
#     22: (8, 23),
#     23: (8, 22),
#     3: (4, 21),
#     4: (4, 4),
#     2: (21, 1),
#     1: (17, 13),
#     17: (18, 1),
#     18: (19, 17),
#     19: (20, 18),
#     20: (20, 20),
#     13: (1, 14),
#     14: (13, 15),
#     15: (14, 16),
#     16: (16, 16)
# }

fsd_bone_angle_pairs = {
    4: (4, 4),
    3: (4, 2),
    2: (3, 1),
    1: (2, 5),

    5: (1, 6),
    6: (5, 7),
    7: (7, 7),
    17: (17, 17),
    15: (17, 0),
    0: (15, 16),
    16: (0, 18),
    18: (18, 18),
    23: (23, 23),  # 左脚

    22: (23, 11),
    11: (22, 24),
    24: (24, 24),
    10: (11, 9),
    9: (10, 8),
    8: (9, 12),
    12: (8, 13),
    13: (12, 14),
    14: (21, 19),
    21: (21, 21),
    19: (14, 20),
    20: (20, 20)  # 右脚
}


# bone_pairs = {
#     'ntu/xview': ntu_skeleton_bone_pairs,
#     'ntu/xsub': ntu_skeleton_bone_pairs,
#
#     # NTU 120 uses the same skeleton structure as NTU 60
#     'ntu120/xsub': ntu_skeleton_bone_pairs,
#     'ntu120/xset': ntu_skeleton_bone_pairs,
#
#     'kinetics': (
#         (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
#         (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
#     )
# }

# benchmarks = {
#     'ntu': ('ntu/xview', 'ntu/xsub'),
#     'ntu120': ('ntu120/xset', 'ntu120/xsub'),
#     'kinetics': ('kinetics',)
# }
#
# parts = {'train', 'val'}

if __name__ == "__main__":
    # Hyperparameters
    bch_sz = 300
    # print_freq = 3
    path = ['G:\花样滑冰\\new_data2\\remove_zeros_shutil_train_data.npy',
            'G:\花样滑冰\\new_data2\\remove_zeros_test_B_data.npy']

    new_data_path = ['G:\花样滑冰\\new_data2\\remove_zeros_shutil_train_angle3.npy',
                     'G:\花样滑冰\\new_data2\\remove_zeros_test_B_angle3.npy']

    for i, new_data_file_name in enumerate(new_data_path):
        # if os.path.exists(new_data_file_name):
        #     continue

        data = np.load(path[i])
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            filename=new_data_file_name,
            # save_name.format(benchmark, part),
            dtype='float32',
            mode='w+',
            shape=(N, 9, T, V, M))

        # fp_sp = np.zeros((N, 12, T, V, M), dtype=float)
        bch_len = N // bch_sz

        for i in tqdm(range(bch_len + 1)):
            # if i % print_freq == 0:
            #     print(f'{i} out of {bch_len}')
            a_bch = paddle.to_tensor(data[i * bch_sz:(i + 1) * bch_sz])
            # generating bones
            # fp_sp_joint_list_bone = []
            fp_sp_joint_list_bone_angle = []
            fp_sp_joint_list_body_center_angle_1 = []
            fp_sp_joint_list_body_center_angle_2 = []
            fp_sp_left_hand_angle = []
            fp_sp_right_hand_angle = []
            fp_sp_two_hand_angle = []
            fp_sp_two_elbow_angle = []
            fp_sp_two_knee_angle = []
            fp_sp_two_feet_angle = []

            all_list = [
                fp_sp_joint_list_bone_angle, fp_sp_joint_list_body_center_angle_1,  # fp_sp_joint_list_bone,
                fp_sp_joint_list_body_center_angle_2, fp_sp_left_hand_angle, fp_sp_right_hand_angle,
                fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
                fp_sp_two_feet_angle
            ]

            # cosine
            cos = nn.CosineSimilarity(axis=1, eps=0)

            for a_key in fsd_bone_angle_pairs:
                a_angle_value = fsd_bone_angle_pairs[a_key]
                # a_bone_value = ntu_bone_adj[a_key]
                the_joint = a_key
                # a_adj = a_bone_value - 1
                # a_bch = a_bch  # .to('cuda')
                # bone_diff = (a_bch[:, :3, :, the_joint, :] -
                #              a_bch[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
                # fp_sp_joint_list_bone.append(bone_diff)

                # bone angles
                v1 = a_angle_value[0] - 1
                v2 = a_angle_value[1] - 1
                vec1 = a_bch[:, :2, :, v1, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, v2, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # body angles 1  111
                vec1 = a_bch[:, :2, :, 8, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 1, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_joint_list_body_center_angle_1.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # body angles 2  111
                vec1 = a_bch[:, :2, :, the_joint, :] - a_bch[:, :2, :, 1, :]
                vec2 = a_bch[:, :2, :, 8, :] - a_bch[:, :2, :, 1, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_joint_list_body_center_angle_2.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # left feet angle
                vec1 = a_bch[:, :2, :, 23, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 22, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_left_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # right feet angle 111
                vec1 = a_bch[:, :2, :, 20, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 19, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_right_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # two hand angle  111
                vec1 = a_bch[:, :2, :, 4, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 7, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # two elbow angle 111
                vec1 = a_bch[:, :2, :, 3, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 6, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # two knee angle 111
                vec1 = a_bch[:, :2, :, 10, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 13, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

                # two feet angle  111
                vec1 = a_bch[:, :2, :, 23, :] - a_bch[:, :2, :, the_joint, :]
                vec2 = a_bch[:, :2, :, 20, :] - a_bch[:, :2, :, the_joint, :]
                angular_feature = (1.0 - cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            for a_list_id in range(len(all_list)):
                all_list[a_list_id] = paddle.concat(all_list[a_list_id], axis=3)

            all_list = paddle.concat(all_list, axis=1)

            # # Joint features. 去掉joint特征
            # fp_sp[i * bch_sz:(i + 1) * bch_sz, :3, :, :, :] = a_bch.cpu().numpy()
            # Bone and angle features.
            # fp_sp[i * bch_sz:(i + 1) * bch_sz, 3:, :, :, :] = all_list.numpy()
            fp_sp[i * bch_sz:(i + 1) * bch_sz, :, :, :, :] = all_list.numpy()

        print('fp sp: ', fp_sp.shape)
    # with open('G:\花样滑冰\\new_data\\remove_zeros_shutil_train_angle.npy', 'wb') as f:
    # np.save(fp_sp)
    # np.save('G:\花样滑冰\\new_data\\remove_zeros_shutil_train_angle.npy',fp_sp)
