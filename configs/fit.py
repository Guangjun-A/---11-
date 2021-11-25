#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 14:55
# @Author  : Jun
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    args = parser.add_argument_group('Training', 'net training')
    args.add_argument('-c', '--config', type=str, required=True)
    args.add_argument('--model', type=str, default=None)

    args.add_argument('--data_format', type=str, default='h5')
    args.add_argument('--rota', type=float, default=0, help='rota matrix')
    args.add_argument('--bone', action='store_true', default=False, help='bone mode')
    args.add_argument('--debug', action='store_true', default=False)
    args.add_argument('--server', type=str, default='siyuan')

    args.add_argument('--phase', type=str, default='train')
    args.add_argument('--save_interval', type=int, default=1)
    args.add_argument('--seg', type=int, required=True, help='number of segmentation')
    args.add_argument('--eff_input', type=str, default='JVB',
                      choices=['J', 'V', 'B', 'A',
                               'JV', 'JB', 'JA', 'VB', 'VA', 'BA',
                               'JVB', 'JVA', 'VBA',
                               'JVBA'], help='number of segmentation')
    args.add_argument('--kfold_start_epoch', type=int, default=None)
    args.add_argument('--start_k', type=int, default=None)
    args.add_argument('--drop_out', type=float, default=0)
    args.add_argument('--pca', action='store_true', default=False)
    args.add_argument('--fast_seg', type=int, default=None, help='number of segmentation')
    args.add_argument('--model_type', type=str, default=None)
    args.add_argument('--sample', type=int, default=10,
                      help='number of segmentation')
    args.add_argument('--optim', choices=['SGD', 'Momentum'], default='Momentum')
    args.add_argument('--eval_epoch', type=int, default=None, help='the net to be evaluated')
    args.add_argument('--focal_loss', action='store_true', default=False, help='the net to be evaluated')
    args.add_argument('--pre_train_model_path', type=str, default=None, help='pre_train_model_path')
    args.add_argument('--model_distillation_path', type=str, default=None, help='model_distillation_path')
    args.add_argument('--precise_BN', action='store_true', default=False, help='precise BN')
    args.add_argument('--f1', action='store_true', default=False, help='precise BN')
    args.add_argument('--parallel', action='store_true', default=False, help='precise BN')
    args.add_argument('--seed', type=int, default=1117, help='the net to be evaluated')
    args.add_argument('--center_point', type=int, default=1, choices=[1, 8], help='the net to be evaluated')

    args.add_argument('--model_path', type=str, default='checkpoint', help='path of checkpoint file')
    args.add_argument('--fining_model_path', type=str, default=None, help='path of checkpoint file')
    args.add_argument('-k', '--kfold', type=int, default=10, choices=[1, 3, 4, 5, 10])
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    args.add_argument('--eval_model_path', type=str, default=None)
    args.add_argument('--load_best', action='store_true', default=False)
    args.add_argument('--device', type=str, help='cuda device')
    args.add_argument('--dataset', type=str, default='NTU',
                      help='select dataset to evlulate')
    args.add_argument('--num_joint', default=25, type=int, help='Number of joint ')
    args.add_argument('--start-epoch', default=0, type=int,
                      help='manual epoch number (useful on restarts)')
    args.add_argument('--max-epoches', type=int, default=70,
                      help='max number of epochs to run')
    args.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
    args.add_argument('--lr-factor', type=float, default=0.1,
                      help='the ratio to reduce lr on each step')
    args.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                      help='weight decay (default: 1e-4)')

    args.add_argument('-b', '--batch-size', type=int, default=256,
                      help='mini-batch size (default: 256)')
    args.add_argument('--num-classes', type=int, default=60,
                      help='the number of classes')

    args.add_argument('--case', type=str, default='CS',
                      help='select which case')

    args.add_argument('--workers', type=int, default=2,
                      help='number of data loading workers (default: 2)')

    return args
