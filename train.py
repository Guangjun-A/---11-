#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/9 15:08
# @Author  : Jun
from configs import fit
import argparse
import os
import paddle.distributed as dist
from stack_boosting import stack_boosting
# from sklearn.metrics import f1_score
from paddle.regularizer import L2Decay
from utils.focal_loss import FocalLoss
from utils.tools import *
# from feeder.data_npy import load_data
# from feeder.data import NTUDataLoaders
import paddle.optimizer as optim
from paddle.optimizer.lr import MultiStepDecay, ReduceOnPlateau, CosineAnnealingDecay
from utils.my_lr import WarmupCosineAnnealingDecay, WarmupMultiStepDecay
import paddle.nn.functional as F
import yaml
import warnings
import paddle
import pandas as pd
# from feeder.efficent_fsd_feed1 import GetKDataLoader
from feeder.efficent_fsd_feed_with_angle2 import GetKDataLoader
import numpy as np
import shutil
from utils.preciseBN import do_preciseBN

paddle.device.set_device("gpu")

paddle.seed(1117)
# paddle.manual_seed(117)
warnings.filterwarnings("ignore", category=Warning)


def train(data_loader, model, criterion, optim, precise=False, epoch=0):
    losses = AverageMeter()
    # acces = AverageMeter()

    acces = paddle.metric.Accuracy()
    model.train()
    for i, (data, label) in enumerate(data_loader):
        if paddle.device.is_compiled_with_cuda():
            data = paddle.to_tensor(data, dtype='float32')  # .cuda()
            label = paddle.nn.functional.one_hot(label, num_classes=30)
            label = F.label_smooth(label)  # .cuda()

        y_pre = model(data)
        loss = criterion(y_pre, label)
        # acc = accuracy(y_pre, label)
        # acc = paddle.metric.accuracy(y_pre, label)
        correct = acces.compute(y_pre, label)

        losses.update(loss.item(), data.shape[0])
        acces.update(correct)

        optim.clear_grad()
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print('        item:{:<5d} loss:{:.5f}'.format(i, loss.item()))
    return losses.avg, acces.accumulate() * 100


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


def evaluate(data_loader, model, criterion):
    losses = AverageMeter()
    acces = paddle.metric.Accuracy()
    acces_f1_score = AverageMeter()
    results = {'predict_category': [], 'true_category': []}

    model.eval()
    for i, (data, label) in enumerate(data_loader):

        with paddle.no_grad():
            if paddle.device.is_compiled_with_cuda():
                data = paddle.to_tensor(data, dtype='float32')  # .cuda()
                # data = data#.cuda()
                label = paddle.nn.functional.one_hot(label, num_classes=30)
                label = F.label_smooth(label)  # .cuda() # .cuda() # 加入smooth label
            y_pre = model(data)
            loss = criterion(y_pre, label)
            correct = acces.compute(y_pre, label)
            f1_correct = accuracy(y_pre, label)

            losses.update(loss.item(), data.shape[0])
            acces.update(correct)
            acces_f1_score.update(f1_correct)

    return losses.avg, acces.accumulate() * 100, acces_f1_score.avg * 100


def test(data_loader, model):
    model.eval()
    results = {'sample_index': [], 'predict_category': []}
    for i, (data) in enumerate(data_loader):

        with paddle.no_grad():
            if paddle.device.is_compiled_with_cuda():
                data = data
                # label = label#.cuda()
            y_pre = model(data)
            results['sample_index'].append(i)
            results['predict_category'].append(y_pre)
            # loss = criterion(y_pre, label)
            # correct = acces.compute(y_pre, label)

            # losses.update(loss.item(), data.shape[0])
            # acces.update(correct)

    return results


def model_init(args):
    # 加载模型
    # model 0 6
    if args.model == 'net.4BranchNetEff.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input)

    # model 1 3 4(JB)
    elif args.model == 'net.4BranchNet_with_tem2.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input, args.seg)

    # model 2
    elif args.model == 'net.efficent_gcn_with_joint_v3.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input)

    # model 5
    elif args.model == 'net.efficent_gcn_with_joint_v2.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input)

    # model 7 8 9
    elif args.model == 'net.efficent_gcn_with_joint_v2_with_tem2.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input)

    #　model 10
    elif args.model == 'net.efficent_gcn_with_angle_4branch_gate.Model':  # EFFICENT模型
        model = import_class(args.model)(args.config, args.eff_input)
    else:
        raise ValueError('args.model error!')

    if args.optim == 'Momentum':
        # scheduler = CosineAnnealingDecay(learning_rate=args.lr, T_max=70)  # 30
        scheduler = WarmupCosineAnnealingDecay(learning_rate=args.lr, T_max=70,
                                               warm_epoch=5, warm_lr=0.0005)

        optimizer = optim.Momentum(parameters=model.parameters(),
                                   learning_rate=scheduler,
                                   weight_decay=L2Decay(args.weight_decay),
                                   momentum=0.9)

    elif args.optim == 'SGD':
        # scheduler = MultiStepDecay(learning_rate=args.lr, milestones=args.step)
        scheduler = WarmupMultiStepDecay(learning_rate=args.lr, milestones=args.step,
                                         warm_epoch=5, warm_lr=0.0001)

        # 优化器
        optimizer = optim.SGD(parameters=model.parameters(),
                              learning_rate=scheduler,
                              weight_decay=L2Decay(args.weight_decay))
    else:
        raise ValueError('miss optimizer and scheduler')
    # 损失函数
    criterion = nn.CrossEntropyLoss(soft_label=True)
    # alpha2 = [0.25, 0.25, 0.25, 0.35, 0.25, 0.25, 0.33, 0.31, 0.25, 0.25, 0.3, 0.25, 0.35, 0.25, 0.35, 0.33, 0.29,
    # 0.33, 0.25, 0.29, 0.33, 0.27, 0.25, 0.25, 0.33, 0.35, 0.25, 0.25, 0.31, 0.25]

    # criterion = FocalLoss(alpha_t=alpha * 10, gamma=2)  # 原本（0.25 for _ in range(30)）
    if args.focal_loss:
        criterion = FocalLoss(alpha_t=[0.25 for _ in range(30)], gamma=2)  # 原本（0.25 for _ in range(30)）
        # criterion = FocalLoss(alpha_t=alpha2, gamma=2)  # 原本（0.25 for _ in range(30)）
        print("focal loss mode open")

    return model, optimizer, scheduler, criterion


def average_k_result(k_res, K_num, model_save_path):  # 传入的为第几折
    k_result = {'acc': [], 'loss': []}
    for k in range(K_num):
        k_result['acc'].append(k_res[k]['test_acc'])
        k_result['loss'].append(k_res[k]['test_loss'])
    k_result['acc'] = np.mean(np.array(k_result['acc']), axis=0)  # 求完平均后的值
    k_result['loss'] = np.mean(np.array(k_result['loss']), axis=0)

    Fold_mean_res = pd.DataFrame(data=k_result)
    Fold_mean_res.to_csv(os.path.join(model_save_path, f'{K_num}Fold_mean_res.csv'))


def K_fold(train_dataloader_ls, val_dataloader_ls, args, model_save_path, file):
    log_info(f"model is {args.model}", file)
    log_info(args, file)

    print(model_save_path)
    shutil.copy(__file__, model_save_path)

    K = len(train_dataloader_ls)
    k_res = {k: {"train_acc": [], "train_loss": [], "test_acc": [], "test_f1_acc": [], "best_acc": [], "test_loss": []}
             for k in range(K)}
    k_best_acc = {k: 0 for k in range(K)}
    k_best_f1_acc = {k: 0 for k in range(K)}
    k_best_epoch = {k: 0 for k in range(K)}
    for k, (train_loader, val_loader) in enumerate(zip(train_dataloader_ls, val_dataloader_ls)):
        start_epoch = 0
        model, optimizer, scheduler, criterion = model_init(args)
        if args.parallel:
            model = paddle.DataParallel(model)
        # 评估前k轮
        if args.start_k is not None and k < args.start_k:
            model = load_model(model, root_path=model_save_path, load_best=True, k=k)
            _, k_best_acc[k], k_best_f1_acc[k] = evaluate(val_loader, model, criterion)
            print(f'{k} fold acc: {k_best_acc[k]:.2f} f1 acc: {k_best_f1_acc[k]:.2f}')
            continue
        # 断点继训
        if args.start_k is not None and k == args.start_k and args.kfold_start_epoch is not None and args.kfold_start_epoch != 0:
            model = load_model(model, root_path=model_save_path, load_best=True, k=args.start_k)
            start_epoch = args.kfold_start_epoch

        # 记录最好轮数和准确率
        best_acc, best_f1_acc, best_epoch = 0.0, 0, 0
        for epoch in range(start_epoch, args.max_epoches):
            # 获取学习率
            lr = optimizer.get_lr()
            print('K={:<2d} epoch:{:<3d} lr: {:.5f}'.format(k, epoch, lr))

            train_loss, train_acc = train(train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch=epoch)

            scheduler.step()
            k_res[k]['train_acc'].append(train_acc)
            k_res[k]['train_loss'].append(train_loss)
            log_info('K={:<2d} epoch:{:<3d} done. lr:{:.5f} loss:{:.5f} acc:{:.3f}'
                     .format(k, epoch, lr, train_loss, train_acc), file)


            print('save path', model_save_path)

            # precise_BN， 实际没有用到
            if args.precise_BN:
                do_preciseBN(model, train_loader, True, len(train_loader))
            test_loss, test_acc, test_f1_acc = evaluate(val_loader, model, criterion)
            k_res[k]['test_acc'].append(test_acc)
            k_res[k]['test_f1_acc'].append(test_f1_acc)
            k_res[k]['test_loss'].append(test_loss)
            k_res[k]['best_acc'].append(best_acc)

            log_info(
                '\033[34mK={:<2d} test epoch:{:<3d} lr:{:.5f} test loss:{:.5f} test acc: {:.3f} test f1 acc: {:.3f}\033[0m'
                    .format(k, epoch, lr, test_loss, test_acc, test_f1_acc), file)

            # save best result
            if test_f1_acc > best_f1_acc:
                best_acc = test_acc
                best_f1_acc, best_epoch = test_f1_acc, epoch
                k_best_f1_acc[k] = best_f1_acc
                k_best_acc[k] = best_acc
                k_best_epoch[k] = best_epoch
                save_model(epoch, model, model_save_path, save_best=True, k=k)
            log_info('K={:<2d} best epoch: {:<3d} test acc: {:.3f} best test f1 acc: {:.3f}\n'
                     .format(k, best_epoch, best_acc, best_f1_acc),
                     file)
        k_result = pd.DataFrame(data=k_res[k])
        k_result.to_csv(os.path.join(model_save_path, f'{k}Fold_res.csv'))
        log_info('eff input is {}'.format(args.eff_input), file)
        for i, (K_acc, K_f1_acc, K_epoch) in enumerate(
                zip(k_best_acc.values(), k_best_f1_acc.values(), k_best_epoch.values())):
            log_info("{:<2d} fold best acc epoch: {:<3d}  acc: {:.3f}  best f1 acc: {:.3f}".format(i, K_epoch, K_acc,
                                                                                                   K_f1_acc), file)
        log_info('{:<2d} fold  average acc: {:.5f} average f1 acc: {:.5f}'.format(K, sum(k_best_acc.values()) / (k + 1),
                                                                                  sum(k_best_f1_acc.values()) / (
                                                                                          k + 1)), file)
    log_info('{:<2d} fold  average acc: {:.5f} average f1 acc: {:.5f}'.format(K, sum(k_best_acc.values()) / K,
                                                                              sum(k_best_f1_acc.values()) / K), file)
    average_k_result(k_res, K, model_save_path)
    print('end time is ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(__file__)
    print('model save path: ', model_save_path)


def main(parser):
    args = parser.parse_args()
    if args.parallel:
        dist.init_parallel_env()

    model_save_path = save_model_args(args, args.model)
    file = open(os.path.join(model_save_path, 'log.txt'), 'a')
    # 加载数据
    log_info(f'start time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', file)
    train_loader_ls, val_loader_ls, _ = GetKDataLoader(args, file, window_size=args.seg, K=args.kfold, debug=args.debug)
    K_fold(train_loader_ls, val_loader_ls, args, model_save_path, file)
    stack_boosting(args)

if __name__ == "__main__":

    #  加载参数
    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')

    fit.add_fit_args(parser)
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    main(parser)
