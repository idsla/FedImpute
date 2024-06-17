from abc import ABC, abstractmethod

import numpy as np
import torch
from typing import Tuple
from ...imputation.base.base_imputer import BaseNNImputer


class StrategyClient(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        pass

    @abstractmethod
    def fed_updates(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def post_training_setup(self, model: torch.nn.Module):
        pass


def fit_local_model_base(
        imputer: BaseNNImputer, training_params: dict, fed_strategy: StrategyClient,
        X_train_imp: np.ndarray, y_train: np.ndarray, X_train_mask: np.ndarray
) -> Tuple[torch.nn.Module, dict]:

    ######################################################################################
    # training params
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        local_epochs = training_params['local_epoch']
        model_converge_tol = training_params['model_converge_tol']
        model_converge_patience = training_params['model_converge_patience']
    except KeyError as e:
        raise ValueError(f"Parameter {str(e)} not found in params")

    ######################################################################################
    # model and dataloader
    model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

    # optimizer and scheduler
    optimizers, lr_schedulers = imputer.configure_optimizer(training_params)
    model.to(DEVICE)

    ######################################################################################
    # pre-training setup
    fed_strategy.pre_training_setup(model)

    ######################################################################################
    # training loop
    total_loss, total_iters = 0, 0
    for ep in range(local_epochs):

        #################################################################################
        # training one epoch
        losses_epoch = [0 for _ in range(len(optimizers))]
        for batch_idx, batch in enumerate(train_dataloader):
            for optimizer_idx, optimizer in enumerate(optimizers):
                #########################################################################
                # training step
                model.train()
                loss, res = model.train_step(batch, batch_idx, optimizers, optimizer_idx)
                #########################################################################
                # fed updates
                fed_strategy.fed_updates(model)

                #########################################################################
                # update loss
                losses_epoch[optimizer_idx] += loss.item()

        #################################################################################
        # epoch end - update loss, early stopping, evaluation, garbage collection etc.
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        losses_epoch = [loss_epoch / len(train_dataloader) for loss_epoch in losses_epoch]

        # update lr scheduler
        for scheduler in lr_schedulers:
            scheduler.step()

        total_loss += sum(losses_epoch)/len(losses_epoch)  # average loss
        total_iters += 1

    #########################################################################################
    # post-training setup
    fed_strategy.post_training_setup(model)

    model.to('cpu')
    final_loss = total_loss / total_iters

    return model, {'loss': final_loss, 'sample_size': len(train_dataloader.dataset)}
