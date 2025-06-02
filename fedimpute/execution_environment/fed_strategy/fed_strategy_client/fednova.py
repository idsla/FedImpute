
import gc
import torch
from typing import Tuple, Dict
from .utils import trainable_params
from ...imputation.base import BaseNNImputer
from .strategy_base import StrategyBaseClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
import numpy as np


########################################################################################################################
# FedNova optimizer
########################################################################################################################
class FedNova(Optimizer):
    """Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(
            self,
            params,
            ratio: float,
            gmf,
            mu=0,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            variance=0
    ):

        self.etamu = 0
        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(self.mu, p.data - param_state['old_init'])

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(local_lr, d_p)

                p.data.add_(-local_lr, d_p)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss

    def get_gradient_scaling(self) -> Dict[str, float]:
        """Compute the scaling factor for local client gradients.

        Returns: A dictionary containing weight, tau, and local_norm.
        """
        if self.mu != 0:
            local_tau = torch.tensor(self.local_steps * self.ratio)
        else:
            local_tau = torch.tensor(self.local_normalizing_vec * self.ratio)

        local_stats = {
            "weight": self.ratio,
            "tau": local_tau.item(),
            "local_norm": self.local_normalizing_vec,
        }

        return local_stats

    def set_model_params(self, init_params):
        """Set the model parameters to the given values."""
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_tensor = torch.tensor(init_params[i])
                p.data.copy_(param_tensor)
                param_state["old_init"] = param_tensor
                i += 1

    def set_lr(self, lr: float):
        """Set the learning rate to the given value."""
        for param_group in self.param_groups:
            param_group["lr"] = lr


class FedNovaStrategyClient(StrategyBaseClient):

    def __init__(self, mu: float = 1.0):

        super(FedNovaStrategyClient, self).__init__(name='fednova')
        self.name = 'fednova'
        self.local_model = None
        self.global_model = None
        self.mu = mu

    def set_parameters(self, updated_model_params: torch.nn.Module, local_model: torch.nn.Module, params: dict):
        for new_param, old_param in zip(updated_model_params.parameters(), local_model.parameters()):
            old_param.data = new_param.data.clone()

    def pre_training_setup(self, params: dict):
        local_model, global_model = params['local_model'], params['global_model']
        for new_param, old_param in zip(global_model.parameters(), local_model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = global_model
        self.local_model = local_model

    def post_training_setup(self, params: dict):
        return {}

    def train_local_nn_model(
            self, imputer: BaseNNImputer, training_params: dict, X_train_imp: np.ndarray,
            y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, dict]:

        ################################################################################################################
        # training params
        try:
            local_epochs = training_params['local_epoch']
            global_model = training_params['global_model']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        ################################################################################################################
        # model and dataloader
        local_model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

        # optimizer and scheduler
        training_params['optimizer_name'] = 'fednova'
        optimizers, lr_schedulers = imputer.configure_optimizer(training_params, local_model) # need to use fednova optimizer
        local_model.to(DEVICE)

        ################################################################################################################
        # pre-training setup - set global_c, global_model, local_model
        pre_training_params = {
            'local_model': local_model, 'global_model': global_model,
        }
        self.pre_training_setup(pre_training_params)

        ################################################################################################################
        # training loop
        self.local_model.train()
        total_loss, total_iters = 0, 0
        # for ep in trange(local_epochs, desc='Local Epoch', colour='blue'):
        for ep in range(local_epochs):

            ############################################################################################################
            # training one epoch
            losses_epoch, ep_iters = [0 for _ in range(len(optimizers))], 0
            for batch_idx, batch in enumerate(train_dataloader):
                # for optimizer_idx, optimizer in enumerate(optimizers):
                ########################################################################################################
                # training step
                loss, res = self.local_model.train_step(batch, batch_idx, optimizers, optimizer_idx=0)

                ########################################################################################################
                # proximal updates
                for w, w_t in zip(trainable_params(self.local_model), trainable_params(self.global_model)):
                    w.grad.data += self.mu * (w.data - w_t.data)

                ########################################################################################################
                # update loss
                for optimizer_idx, optimizer in enumerate(optimizers):
                    losses_epoch[optimizer_idx] += loss

                ep_iters += 1

            ############################################################################################################
            # epoch end - update loss, early stopping, evaluation, garbage collection etc.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            losses_epoch = np.array(losses_epoch) / len(train_dataloader)
            epoch_loss = losses_epoch.mean()

            # update lr scheduler
            # for scheduler in lr_schedulers:
            #     scheduler.step()

            total_loss += epoch_loss  # average loss
            total_iters += 1

        final_loss = total_loss / total_iters
        gc.collect()
        self.local_model.to('cpu')

        #########################################################################################
        # post-training setup
        post_training_ret = self.post_training_setup({})

        return local_model, {
            'loss': final_loss, 'sample_size': len(train_dataloader.dataset),
        }
    
    def __str__(self):
        return f"FedNova Strategy Client"

    def __repr__(self):
        return f"FedNova Strategy Client"



"""
plan: 
1. scaffold, fednova, fedadam, fedadagrad, fedyogi, fedavgm, feddisco
2. per-fedavg, pfedme, pfedgraph

Think about the consistent of API:
-----------------------------------------------------------------------
client => server
1. fedavg, fedprox -> return model parameters
2. fedopt, fednova, scaffold -> return model gradients
------------------------------------------------------------------------
server => client -> model parameters

"""
