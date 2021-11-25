#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 15:54
# @Author  : Jun
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 15:28
# @Author  : Jun
"""
 CUDA_VISIBLE_DEVICES=7 python boosting.py --eval_model_path checkpoint/fsd_all_npy/seg_350/2021-10-02_02-29-17 -k 10 --config configs/ctrgcn-fsd-train.yaml --seg 350 -b 1
"""
import os
import paddle
import pandas as pd
import paddle.nn as nn
import paddle.nn.initializer as init
from utils.tools import *
from utils.tools import load_model, import_class, AverageMeter
from feeder.efficent_fsd_feed_with_angle2_test_B import GetKDataLoader_test as GetKDataLoader
from paddle.optimizer import SGD, Momentum, Adam
from paddle.io import Subset

from paddle.regularizer import L2Decay
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, ReduceOnPlateau
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import warnings
import argparse
import yaml
import shutil
from configs import fit
import random
from utils.confusion_matrix import draw_confusion

warnings.filterwarnings("ignore", category=Warning)
parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
# parser.add_argument('--test', action='store_true', default=False)
# parser.add_argument('--epoch', type=int, default=400)
args = parser.parse_args()

random.seed(1117)  # Python random module.

def weight_init_(layer,
                 func,
                 weight_name=None,
                 bias_name=None,
                 bias_value=0.0,
                 **kwargs):
    """
    In-place params init function.
    Usage:
    .. code-block:: python

        import paddle
        import numpy as np

        data = np.ones([3, 4], dtype='float32')
        linear = paddle.nn.Linear(4, 4)
        input = paddle.to_tensor(data)
        print(linear.weight)
        linear(input)

        weight_init_(linear, 'Normal', 'fc_w0', 'fc_b0', std=0.01, mean=0.1)
        print(linear.weight)
    """

    if hasattr(layer, 'weight') and layer.weight is not None:
        getattr(init, func)(**kwargs)(layer.weight)
        if weight_name is not None:
            # override weight name
            layer.weight.name = weight_name

    if hasattr(layer, 'bias') and layer.bias is not None:
        init.Constant(bias_value)(layer.bias)
        if bias_name is not None:
            # override bias name
            layer.bias.name = bias_name


class MyDataset(paddle.io.Dataset):
    def __init__(self, data_npy_path, label_npy_path, debug=False):
        super(MyDataset, self).__init__()
        self.data = np.load(data_npy_path)
        self.label = np.load(label_npy_path) if label_npy_path is not None else None

        if debug:
            self.data = self.data[0:100]
            self.label = self.label[0:100] if label_npy_path is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.label is not None:
            return self.data[item], self.label[item]
        else:
            return self.data[item], 0.0


# 搭建分类器模型
class Classifer(nn.Layer):
    def __init__(self, fold_num, hidden_dim=128, out_dim=30):
        super(Classifer, self).__init__()
        # self.bn = nn.BatchNorm(30)
        self.fc = nn.Linear(30, 30)

    def forward(self, x):

        return self.fc(x)


def accuracy(y_pre, label):
    _, pre_label = paddle.topk(y_pre, k=1)
    pre_label = pre_label.numpy()

    if label.dim() == 1:
        label = label.numpy()
    else:
        _, label = paddle.topk(label, k=1)
        label = label.numpy()
    acc = f1_score(label, pre_label, average='weighted')  # weighted macro

    return acc

K_fold = 10
assert args.eval_model_path != None, 'MISS eval model path !'

checkpoint_path = f'./{args.eval_model_path}'
print("checkpoint_path: ", checkpoint_path)
if not os.path.exists(os.path.join(checkpoint_path, 'linear_checkpoint')):
    os.makedirs(os.path.join(os.path.join(checkpoint_path, 'linear_checkpoint')))

boost_data_npy_path = os.path.join(checkpoint_path, 'linear_checkpoint', 'boost_data.npy')
boost_label_npy_path = os.path.join(checkpoint_path, 'linear_checkpoint', 'boost_label_data.npy')
boost_test_data_npy_path = os.path.join(checkpoint_path, 'linear_checkpoint', 'boost_test_data.npy')

train_label_npy_path = 'data/new_data/shutil_train_label.npy'
linear_checkpoint_path = os.path.join(checkpoint_path, 'linear_checkpoint')
shutil.copy(__file__, linear_checkpoint_path)
args.batch_size = 2
fc_train_epoch = args.epoch

log_txt_file = open(os.path.join(checkpoint_path, 'linear_log.txt'), 'a')
log_info(args, log_txt_file)
model_dict = {
    'configs/ensemble/0.yaml': f'{args.eval_model_path}/2021-11-16_10-50-34',  # 67.12acc
    'configs/ensemble/1.yaml': f'{args.eval_model_path}/2021-11-16_15-33-28',  # 65.41
    'configs/ensemble/2.yaml': f'{args.eval_model_path}/2021-11-08_18-54-10',  # 在 70.54
    'configs/ensemble/3.yaml': f'{args.eval_model_path}/2021-11-19_00-26-01',  # JB  # 64.726
    'configs/ensemble/4.yaml': f'{args.eval_model_path}/2021-11-16_15-33-28',  # 68.49
    'configs/ensemble/5.yaml': f'{args.eval_model_path}/2021-11-06_16-52-53',  # 68.15
    'configs/ensemble/6.yaml': f"{args.eval_model_path}/2021-11-13_19-37-10",  # 63.014
    'configs/ensemble/7.yaml': f'{args.eval_model_path}/2021-11-15_12-52-15',  # 70.89
    'configs/ensemble/8.yaml': f'{args.eval_model_path}/2021-11-15_12-52-15',  # 70.54
    'configs/ensemble/9.yaml': f'{args.eval_model_path}/2021-11-13_23-09-26'  # 67.34
    # 'configs/ensemble/file_pre/10.yaml': f'{args.eval_model_path}/2021-11-05_21-34-34' # 67.34
}
# train_fc = True  # True #
# _, val_dataloader_ls, test_dataloader = GetKDataLoader(args, K=args.kfold, window_size=args.seg)  # , feed_args)
_, _, test_dataloader = GetKDataLoader(args, K=args.kfold, window_size=args.seg)  # , feed_args)

# 制作FC的数据
print('yes')
if not os.path.exists(boost_data_npy_path):
    # 得到K折交叉验证，K个模型的预测进行cat
    boost_data = paddle.to_tensor([], dtype=paddle.float32)
    # boost_test_data = paddle.to_tensor([], dtype=paddle.float32)
    # boost_data = paddle.to_tensor([], dtype=paddle.float32)
    boost_test_data = paddle.to_tensor([], dtype=paddle.float32)
    boost_label_data = paddle.to_tensor([], dtype=paddle.float32)
    for k in range(K_fold):
        config = list(model_dict.keys())[k]
        with open(config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(args).keys()
        for kk in default_arg.keys():
            if kk not in key:
                print('WRONG ARG:', kk)
                assert (kk in key)
        parser.set_defaults(**default_arg)
        args = parser.parse_args()
        args.config = list(model_dict.keys())[k]
        print(args.config)
        print(args.eff_input)
        if k==9:
            _, _, test_dataloader = GetKDataLoader(args, K=args.kfold, window_size=args.seg)  # , feed_args)

        model_checkpoint_path = list(model_dict.values())[k]
        model = import_class(args.model)(args.config, args.eff_input)

        model = load_model(model, root_path=model_checkpoint_path, k=k, load_best=True)

        test_K_pre_value = paddle.to_tensor([], dtype=paddle.float32)  # 存放第K个模型测试集的预测值
        model.eval()
        for i, (test_data, _) in enumerate(tqdm(test_dataloader, desc=f"{k} fold test ")):
            with paddle.no_grad():
                test_data = paddle.to_tensor(test_data, dtype='float32')
                test_y = model(test_data)  # (N, C)

            test_K_pre_value = paddle.concat([test_K_pre_value, paddle.unsqueeze(test_y, 0)],
                                             axis=1) if i != 0 else paddle.unsqueeze(test_y, 0)  # (1, N, C)

        boost_test_data = paddle.concat([boost_test_data, test_K_pre_value],
                                        axis=0) if k != 0 else test_K_pre_value  # (K, N, C)

    boost_test_data = paddle.mean(boost_test_data, axis=0)  # (N, C)
    print(boost_test_data.shape)
    np.save(boost_test_data_npy_path, np.array(boost_test_data))
# 将预测值进行linear训练

# if args.test:
# else:
print('-' * 50 + 'testing...' + '-' * 50)

test_dataset = MyDataset(data_npy_path=boost_test_data_npy_path, label_npy_path=None)
test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

FC_model = Classifer(fold_num=K_fold)

checkpoint = paddle.load(os.path.join(linear_checkpoint_path, f"checkpoint_best.pdparams"))
if args.eval_epoch is not None:
    checkpoint = paddle.load(os.path.join(linear_checkpoint_path, f"checkpoint_{args.eval_epoch}.pdparams"))
    print('model path:{}'.format(os.path.join(linear_checkpoint_path, f"checkpoint_{args.eval_epoch}.pdparams")))
FC_model.set_state_dict(checkpoint)

losses = AverageMeter()
FC_model.eval()
results = {'sample_index': [], 'predict_category': []}
for i, (boost_x, _) in enumerate(tqdm(test_dataloader)):
    boost_x = paddle.to_tensor(boost_x, dtype=paddle.float32)
    with paddle.no_grad():
        y = FC_model(boost_x)
        y_test_score = np.concatenate([y_test_score, np.array(y)], axis=0) if i != 0 else np.array(y)

        _, class_pre = paddle.topk(y, axis=1, k=1)
        results['sample_index'].append(i)
        results['predict_category'].append(class_pre.item())

np.save(os.path.join(checkpoint_path, 'y_test_score.npy'), y_test_score)
data_frame = pd.DataFrame(data=results)
data_frame.to_csv(os.path.join(checkpoint_path, 'submission.csv'), index=False)
print("save path: ", checkpoint_path)
