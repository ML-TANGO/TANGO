import torch
from torch.optim import lr_scheduler
import numpy as np


class _LRMomentumScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_momentum', group['momentum'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_momentum' not in group:
                    raise KeyError("param 'initial_momentum' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_momentums = list(map(lambda group: group['initial_momentum'], optimizer.param_groups))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def get_momentum(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr, momentum in zip(self.optimizer.param_groups, self.get_lr(), self.get_momentum()):
            param_group['lr'] = lr
            param_group['momentum'] = momentum


def apply_lambda(last_epoch, bases, lambdas):
    return [base * lmbda(last_epoch) for lmbda, base in zip(lambdas, bases)]


class LambdaScheduler(_LRMomentumScheduler):
    """Sets the learning rate and momentum of each parameter group to the initial lr and momentum
    times a given function. When last_epoch=-1, sets initial lr and momentum to the optimizer
    values.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
            Default: lambda x:x.
        momentum_lambda (function or list): As for lr_lambda but applied to momentum.
            Default: lambda x:x.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        # >>> # Assuming optimizer has two groups.
        # >>> lr_lambda = [
        # ...     lambda epoch: epoch // 30,
        # ...     lambda epoch: 0.95 ** epoch
        # ... ]
        # >>> mom_lambda = [
        # ...     lambda epoch: max(0, (50 - epoch) // 50),
        # ...     lambda epoch: 0.99 ** epoch
        # ... ]
        # >>> scheduler = LambdaScheduler(optimizer, lr_lambda, mom_lambda)
        # >>> for epoch in range(100):
        # >>>     train(...)
        # >>>     validate(...)
        # >>>     scheduler.step()
"""

    def __init__(self, optimizer, lr_lambda=lambda x: x, momentum_lambda=lambda x: x, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)

        if not isinstance(momentum_lambda, (list, tuple)):
            self.momentum_lambdas = [momentum_lambda] * len(optimizer.param_groups)
        else:
            if len(momentum_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} momentum_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(momentum_lambda)))
            self.momentum_lambdas = list(momentum_lambda)

        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate and momentum lambda functions will only be saved if they are
        callable objects and not if they are functions or lambdas.
        """
        state_dict = {key: value for key, value in self.__dict__.items()
                      if key not in ('optimizer', 'lr_lambdas', 'momentum_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)
        state_dict['momentum_lambdas'] = [None] * len(self.momentum_lambdas)

        for idx, (lr_fn, mom_fn) in enumerate(zip(self.lr_lambdas, self.momentum_lambdas)):
            if not isinstance(lr_fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = lr_fn.__dict__.copy()
            if not isinstance(mom_fn, types.FunctionType):
                state_dict['momentum_lambdas'][idx] = mom_fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop('lr_lambdas')
        momentum_lambdas = state_dict.pop('momentum_lambdas')
        self.__dict__.update(state_dict)

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

        for idx, fn in enumerate(momentum_lambdas):
            if fn is not None:
                self.momentum_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        return apply_lambda(self.last_epoch, self.base_lrs, self.lr_lambdas)

    def get_momentum(self):
        return apply_lambda(self.last_epoch, self.base_momentums, self.momentum_lambdas)