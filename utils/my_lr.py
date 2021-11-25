#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 20:03
# @Author  : Jun
import math
from paddle.optimizer.lr import *


class WarmupCosineAnnealingDecay(LRScheduler):
    r"""

    Set the learning rate using a cosine annealing schedule, where :math:`\eta_{max}` is set to
    the initial learning_rate. :math:`T_{cur}` is the number of epochs since the last restart in
    SGDR.

    The algorithm can be described as following.

    .. math::

        \\begin{aligned}
            \eta_t & = \eta_{min} + \\frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\\frac{T_{cur}}{T_{max}}\pi\\right)\\right),
            & T_{cur} \\neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \\frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\\frac{1}{T_{max}}\pi\\right)\\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_.
    Note that this only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        learning_rate (float): The initial learning rate, that is :math:`\eta_{max}` . It can be set to python float or int number.
        T_max (int): Maximum number of iterations. It is half of the decay cycle of learning rate.
        eta_min (float|int, optional): Minimum learning rate, that is :math:`\eta_{min}` . Default: 0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 T_max,
                 warm_epoch,
                 warm_lr,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        self.T_max = T_max
        self.warm_epoch = warm_epoch
        self.warm_lr = warm_lr
        self.eta_min = float(eta_min)
        super(WarmupCosineAnnealingDecay, self).__init__(learning_rate, last_epoch,
                                                         verbose)

    def get_lr(self):
        if self.last_epoch <= self.warm_epoch:
            interval = (self.base_lr - self.warm_lr) / self.warm_epoch
            return self.warm_lr + interval * self.last_epoch

        if self.last_epoch == self.warm_epoch + 1:
            return self.base_lr

        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                       self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max)) / 2


class WarmupMultiStepDecay(LRScheduler):
    """
    Update the learning rate by ``gamma`` once ``epoch`` reaches one of the milestones.

    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.5
        milestones = [30, 50]
        gamma = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005

    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .


    Returns:
        ``MultiStepDecay`` instance to schedule learning rate.

    Examples:

        .. code-block:: python

            import paddle
            import numpy as np

            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch

            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)

            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 warm_epoch,
                 warm_lr,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))

        if not all([
            milestones[i] < milestones[i + 1]
            for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')

        self.milestones = milestones
        self.gamma = gamma
        self.warm_epoch = warm_epoch
        self.warm_lr = warm_lr

        super(WarmupMultiStepDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_epoch:
            interval = (self.base_lr - self.warm_lr) / self.warm_epoch
            return self.warm_lr + interval*self.last_epoch

        for i in range(len(self.milestones)):
            if self.last_epoch < self.milestones[i]:
                return self.base_lr * (self.gamma ** i)
        return self.base_lr * (self.gamma ** len(self.milestones))


class CustomWarmupCosineDecay(LRScheduler):
    r"""
    We combine warmup and stepwise-cosine which is used in slowfast model.

    Args:
        warmup_start_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        cosine_base_lr (float|int, optional): base learning rate in cosine schedule.
        max_epoch (int): total training epochs.
        num_iters(int): number iterations of each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.
    """

    def __init__(self,
                 warmup_start_lr,
                 warmup_epochs,
                 cosine_base_lr,
                 max_epoch,
                 num_iters=None,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_base_lr = cosine_base_lr
        self.max_epoch = max_epoch
        self.num_iters = num_iters
        # call step() in base class, last_lr/last_epoch/base_lr will be update
        super(CustomWarmupCosineDecay, self).__init__(last_epoch=last_epoch,
                                                      verbose=verbose)

    def step(self, epoch=None):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if self.last_epoch == -1:
                self.last_epoch += 1
            else:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch
        self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def _lr_func_cosine(self, cur_epoch, cosine_base_lr, max_epoch):
        return cosine_base_lr * (math.cos(math.pi * cur_epoch / max_epoch) +
                                 1.0) * 0.5

    def get_lr(self):
        """Define lr policy"""
        lr = self._lr_func_cosine(self.last_epoch, self.cosine_base_lr,
                                  self.max_epoch)
        lr_end = self._lr_func_cosine(self.warmup_epochs, self.cosine_base_lr,
                                      self.max_epoch)

        # Perform warm up.
        if self.last_epoch < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (lr_end - lr_start) / self.warmup_epochs
            lr = self.last_epoch * alpha + lr_start
        return lr


class CustomWarmupPiecewiseDecay(LRScheduler):
    r"""
    This op combine warmup and stepwise-cosine which is used in slowfast model.

    Args:
        warmup_start_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        step_base_lr (float|int, optional): base learning rate in step schedule.
        max_epoch (int): total training epochs.
        num_iters(int): number iterations of each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CustomWarmupPiecewiseDecay`` instance to schedule learning rate.
    """

    def __init__(self,
                 warmup_start_lr,
                 warmup_epochs,
                 step_base_lr,
                 lrs,
                 gamma,
                 steps,
                 max_epoch,
                 num_iters=None,
                 last_epoch=0,
                 verbose=False):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.step_base_lr = step_base_lr
        self.lrs = lrs
        self.gamma = gamma
        self.steps = steps
        self.max_epoch = max_epoch
        self.num_iters = num_iters
        self.last_epoch = last_epoch
        self.last_lr = self.warmup_start_lr  # used in first iter
        self.verbose = verbose
        self._var_name = None

    def step(self, epoch=None, rebuild=False):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if not rebuild:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch
        self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def _lr_func_steps_with_relative_lrs(self, cur_epoch, lrs, base_lr, steps,
                                         max_epoch):
        # get step index
        steps = steps + [max_epoch]
        for ind, step in enumerate(steps):
            if cur_epoch < step:
                break

        return lrs[ind - 1] * base_lr

    def get_lr(self):
        """Define lr policy"""
        lr = self._lr_func_steps_with_relative_lrs(
            self.last_epoch,
            self.lrs,
            self.step_base_lr,
            self.steps,
            self.max_epoch,
        )
        lr_end = self._lr_func_steps_with_relative_lrs(
            self.warmup_epochs,
            self.lrs,
            self.step_base_lr,
            self.steps,
            self.max_epoch,
        )

        # Perform warm up.
        if self.last_epoch < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (lr_end - lr_start) / self.warmup_epochs
            lr = self.last_epoch * alpha + lr_start
        return lr


class CustomPiecewiseDecay(PiecewiseDecay):
    def __init__(self, **kargs):
        kargs.pop('num_iters')
        super().__init__(**kargs)
