#!/usr/bin/env python
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
from utils.focal_loss import FocalLoss, FocalLoss2
from utils.tools import load_model, import_class, AverageMeter
from feeder.efficent_fsd_feed_with_angle2 import GetKDataLoader
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
args = parser.parse_args()

random.seed(1117)  # Python random module.

def weight_init_(layer,
                 func,
                 weight_name=None,
                 bias_name=None,
                 bias_value=0.0,
                 **kwargs):

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

alpha3 = [0.25 for _ in range(30)]

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
    'configs/ensemble/0.yaml':f'{args.eval_model_path}/2021-11-16_10-50-34',#67.12acc
    'configs/ensemble/1.yaml':f'{args.eval_model_path}/2021-11-16_15-33-28', # 65.41
    'configs/ensemble/2.yaml': f'{args.eval_model_path}/2021-11-08_18-54-10', # 70.54
    'configs/ensemble/3.yaml':f'{args.eval_model_path}/2021-11-19_00-26-01',# JB  # 64.726
    'configs/ensemble/4.yaml': f'{args.eval_model_path}/2021-11-16_15-33-28', # 68.49
    'configs/ensemble/5.yaml': f'{args.eval_model_path}/2021-11-06_16-52-53', # 68.15
    'configs/ensemble/6.yaml':f"{args.eval_model_path}/2021-11-13_19-37-10", # 63.014
    'configs/ensemble/7.yaml':f'{args.eval_model_path}/2021-11-15_12-52-15', #  70.89
    'configs/ensemble/8.yaml': f'{args.eval_model_path}/2021-11-15_12-52-15', # 70.54
    'configs/ensemble/9.yaml': f'{args.eval_model_path}/2021-11-13_23-09-26' # 67.34
    # 'configs/ensemble/file_pre/10.yaml': f'{args.eval_model_path}/2021-11-05_21-34-34' # 67.34
}
# 制作FC的数据
print('yes')
if not os.path.exists(boost_data_npy_path):
    # 得到K折交叉验证，K个模型的预测进行cat
    boost_data = paddle.to_tensor([], dtype=paddle.float32)
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
        _, val_dataloader_ls, test_dataloader = GetKDataLoader(args, K=args.kfold, window_size=args.seg)  # , feed_args)

        model_checkpoint_path = list(model_dict.values())[k]
        model = import_class(args.model)(args.config, args.eff_input)

        model = load_model(model, root_path=model_checkpoint_path, k=k, load_best=True)
        K_pre_value = paddle.to_tensor([], dtype=paddle.float32)  # 存放第K个模型验证集的预测值
        K_true_label = paddle.to_tensor([], dtype=paddle.int64)  # 存放第K个模型验证集的预测值

        test_K_pre_value = paddle.to_tensor([], dtype=paddle.float32)  # 存放第K个模型测试集的预测值
        train_dataloader = val_dataloader_ls[k]
        model.eval()
        # 制作验证集（10折合一）
        for i, (fc_train_data, fc_train_label) in enumerate(tqdm(train_dataloader, desc=f"{k} fold train")):
            with paddle.no_grad():
                fc_train_data = paddle.to_tensor(fc_train_data, dtype='float32')
                y = model(fc_train_data)  # (N, C)
            K_true_label = paddle.concat([K_true_label, fc_train_label], axis=0) if i != 0 else fc_train_label
            K_pre_value = paddle.concat([K_pre_value, y], axis=0) if i != 0 else y

        boost_data = paddle.concat([boost_data, K_pre_value], axis=0) if k != 0 else K_pre_value
        boost_label_data = paddle.concat([boost_label_data, K_true_label], axis=0) if k != 0 else K_true_label

        for i, (test_data, _) in enumerate(tqdm(test_dataloader, desc=f"{k} fold test ")):
            with paddle.no_grad():
                test_data = paddle.to_tensor(test_data, dtype='float32')
                test_y = model(test_data)  # (N, C)

            test_K_pre_value = paddle.concat([test_K_pre_value, paddle.unsqueeze(test_y, 0)],
                                             axis=1) if i != 0 else paddle.unsqueeze(test_y, 0)  # (1, N, C)

        boost_test_data = paddle.concat([boost_test_data, test_K_pre_value],
                                        axis=0) if k != 0 else test_K_pre_value  # (K, N, C)

    boost_test_data = paddle.mean(boost_test_data, axis=0)  # (N, C)
    print(boost_data.shape, boost_label_data.shape, boost_test_data.shape)

    np.save(boost_data_npy_path, np.array(boost_data))  # (N, C) 最后是N, C
    np.save(boost_label_npy_path, np.array(boost_label_data))
    np.save(boost_test_data_npy_path, np.array(boost_test_data))
# 将预测值进行linear训练

if not args.test:  # 训练分类器
    print("train")
    # 制作FC数据集
    fc_train_dataset = MyDataset(data_npy_path=boost_data_npy_path, label_npy_path=train_label_npy_path)
    data_id_ls = [i for i in range(len(fc_train_dataset))]
    random.shuffle(data_id_ls)
    test_data_id = data_id_ls[int(0.1 * len(data_id_ls)):]
    train_data_id = data_id_ls[:len(data_id_ls) - int(0.1 * len(data_id_ls))]

    fc_train_dataloader = paddle.io.DataLoader(Subset(fc_train_dataset, train_data_id), batch_size=64, shuffle=True,
                                               drop_last=False,
                                               num_workers=8)
    fc_test_dataloader = paddle.io.DataLoader(Subset(fc_train_dataset, test_data_id), batch_size=64, shuffle=True,
                                              drop_last=False,
                                              num_workers=8)
    # scheduler = CosineAnnealingDecay(learning_rate=0.1, T_max=80, eta_min=1e-5)
    # scheduler = MultiStepDecay(learning_rate=0.1, milestones=[200, 400, 600, 800])
    scheduler = ReduceOnPlateau(learning_rate=0.1, patience=20, min_lr=1e-6)
    # 优化器

    FC_model = Classifer(fold_num=K_fold)
    optimizer = Adam(parameters=FC_model.parameters(),
                     learning_rate=scheduler,
                     weight_decay=L2Decay(1e-6),
                     )
    criterion = paddle.nn.CrossEntropyLoss()  # soft_label=True)
    if args.focal_loss:
        criterion = FocalLoss2(alpha3, gamma=2)
    best_acc, best_f1_acc, best_epoch, best_train_acc, best_train_f1_acc = 0, 0, 0, 0, 0
    for epoch in range(fc_train_epoch):
        lr = optimizer.get_lr()

        losses = AverageMeter()
        acces_f1_score = AverageMeter()
        acces = paddle.metric.Accuracy()

        acces_test_f1_score = AverageMeter()
        acces_test = paddle.metric.Accuracy()

        FC_model.train()
        y_pre_class = np.array([])
        y_true_class = np.array([])
        y_score = np.array([])
        for i, (boost_x, label) in enumerate(tqdm(fc_train_dataloader)):
            label2 = label
            boost_x = paddle.to_tensor(boost_x, dtype=paddle.float32)
            label = paddle.to_tensor(label, dtype=paddle.int64)

            y = FC_model(boost_x)
            y_pre_class = np.concatenate([y_pre_class, np.array(paddle.topk(y, 1, 1)[1])],
                                         axis=0) if i != 0 else np.array(paddle.topk(y, 1, 1)[1])
            y_true_class = np.concatenate([y_true_class, np.array(label2)], axis=0) if i != 0 else np.array(label2)
            y_score = np.concatenate([y_score, np.array(y)], axis=0) if i != 0 else np.array(y)

            loss = criterion(y, label)
            correct = acces.compute(y, label)

            f1_correct = accuracy(y, label)
            acces_f1_score.update(f1_correct)

            losses.update(loss.item(), boost_x.shape[0])
            acces.update(correct)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        if isinstance(scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            scheduler.step(losses.avg)
        else:
            scheduler.step()

        FC_model.eval()
        y_pre_class = np.array([])
        y_true_class = np.array([])
        y_score = np.array([])
        for i, (boost_x, label) in enumerate(tqdm(fc_test_dataloader)):
            label2 = label
            boost_x = paddle.to_tensor(boost_x, dtype=paddle.float32)
            label = paddle.to_tensor(label, dtype=paddle.int64)

            y = FC_model(boost_x)
            y_pre_class = np.concatenate([y_pre_class, np.array(paddle.topk(y, 1, 1)[1])],
                                         axis=0) if i != 0 else np.array(paddle.topk(y, 1, 1)[1])
            y_true_class = np.concatenate([y_true_class, np.array(label2)], axis=0) if i != 0 else np.array(label2)
            y_score = np.concatenate([y_score, np.array(y)], axis=0) if i != 0 else np.array(y)

            correct = acces.compute(y, label)

            f1_correct = accuracy(y, label)
            acces_test_f1_score.update(f1_correct)

            acces_test.update(correct)
        if acces_test_f1_score.avg * 100 > best_f1_acc:
            best_train_acc = acces.accumulate() * 100
            best_train_f1_acc = acces_f1_score.avg * 100
            best_acc = acces_test.accumulate() * 100
            best_f1_acc = acces_test_f1_score.avg * 100
            best_epoch = epoch
        paddle.save(FC_model.state_dict(), os.path.join(linear_checkpoint_path, f"checkpoint_best.pdparams"))
        np.save(os.path.join(checkpoint_path, 'y_val_score.npy'), y_score)
        log_info('epoch: {:<3d} lr:{:.10f} loss: {:.5f} acc: {:.2f}({:.2f}), f1 acc: {:.2f}({:.2f}) '

                 ' test best acc {:.2f}({:.2f}) test best f1 acc {:.2f}({:.2f}) best epoch {:<3d}'.format(epoch, lr,
                                                                                                          losses.avg,
                                                                                                          acces_test.accumulate() * 100,
                                                                                                          acces.accumulate() * 100,
                                                                                                          acces_test_f1_score.avg * 100,
                                                                                                          acces_f1_score.avg * 100,
                                                                                                          best_acc,
                                                                                                          best_train_acc,
                                                                                                          best_f1_acc,
                                                                                                          best_train_f1_acc,
                                                                                                          best_epoch),
                 log_txt_file)

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

    # 保留得分
    np.save(os.path.join(checkpoint_path, 'y_test_score.npy'), y_test_score)

    data_frame = pd.DataFrame(data=results)
    data_frame.to_csv(os.path.join(checkpoint_path, 'submission.csv'), index=False)

    print("save path: ", checkpoint_path)

if args.test:
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
