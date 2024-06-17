import gc

import numpy as np

from ...fed_strategy.fed_strategy_client.base_strategy import StrategyClient
import torch
from typing import Tuple
from .utils import trainable_params
from ...imputation.base import BaseNNImputer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedProxStrategyClient(StrategyClient):

    def __init__(self, mu: float = 1.0):
        super().__init__('fedprox')

        self.mu = mu
        self.global_model_params = None

    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        self.global_model_params = trainable_params(model, detach=True)

    def fed_updates(self, model: torch.nn.Module):
        #print(self.global_model_params)
        for w, w_t in zip(trainable_params(model), self.global_model_params):
            w.grad.data += self.mu * (w.data - w_t.data)

    def post_training_setup(self, model: torch.nn.Module):
        self.global_model_params = None
        gc.collect()

    def train_local_nn_model(
            self, imputer: BaseNNImputer, training_params: dict, X_train_imp: np.ndarray,
            y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, dict]:

        ################################################################################################################
        # training params
        try:
            local_epochs = training_params['local_epoch']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        ################################################################################################################
        # model and dataloader
        model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

        # optimizer and scheduler
        optimizers, lr_schedulers = imputer.configure_optimizer(training_params, model)
        model.to(DEVICE)

        ################################################################################################################
        # pre-training setup
        self.global_model_params = trainable_params(model)

        ################################################################################################################
        # training loop
        total_loss, total_iters = 0, 0

        # for ep in trange(local_epochs, desc='Local Epoch', colour='blue'):
        for ep in range(local_epochs):

            #################################################################################
            # training one epoch
            losses_epoch, ep_iters = [0 for _ in range(len(optimizers))], 0
            for batch_idx, batch in enumerate(train_dataloader):
                # for optimizer_idx, optimizer in enumerate(optimizers):
                #########################################################################
                # training step
                model.train()
                loss, res = model.train_step(batch, batch_idx, optimizers, optimizer_idx=0)

                #########################################################################
                # fedprox updates
                for w, w_t in zip(trainable_params(model), self.global_model_params):
                    w.grad.data += self.mu * (w.data - w_t.data)

                #########################################################################
                # update loss
                for optimizer_idx, optimizer in enumerate(optimizers):
                    losses_epoch[optimizer_idx] += loss

                ep_iters += 1

            #################################################################################
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

        #########################################################################################
        # post-training setup
        self.global_model_params = None
        gc.collect()

        model.to('cpu')
        final_loss = total_loss / total_iters

        return model, {'loss': final_loss, 'sample_size': len(train_dataloader.dataset)}
