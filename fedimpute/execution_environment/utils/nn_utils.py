from typing import Iterable, Union, Any

import loguru
import torch
from torch import nn


def load_optimizer(
        optimizer_name: str, parameters: Iterable[torch.Tensor], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    if optimizer_name == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'asgd':
        return torch.optim.ASGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'lbfgs':
        return torch.optim.LBFGS(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def load_lr_scheduler(
        scheduler_name: str, optimizer: torch.optim.Optimizer, scheduler_params: dict
) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if scheduler_name == 'step':
        try:
            step_size = scheduler_params['step_size']
            gamma = scheduler_params['gamma']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'exp':
        try:
            gamma = scheduler_params['gamma']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'cos':
        try:
            step_size = scheduler_params['step_size']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    elif scheduler_name == None:
        return None
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")


def weights_init(layer: Any, initializer: str) -> None:
    if type(layer) == nn.Linear:
        if initializer == 'xavier':
            torch.nn.init.xavier_normal_(layer.weight, gain=1)
        elif initializer == 'orthogonal':
            torch.nn.init.orthogonal_(layer.weight)
        elif initializer == 'kaiming':
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        else:
            raise ValueError(f"Unknown initializer: {initializer}")


class EarlyStopping:
    def __init__(
            self,
            tolerance=1e-4,
            tolerance_patience=3,
            increase_patience=20,
            window_size=5,
            check_steps=1,
            backward_window_size=20,
            verbose=False
    ):
        """
        初始化EarlyStopping对象。

        参数:
        - tolerance: float, 指标变化的容差。
        - patience: int, 允许的最大连续不显著变化次数。
        - window_size: int, 滑动窗口大小。
        - check_steps: int, 每多少步检查一次。
        - max_tolerance_steps: int, 允许的最大连续容差超出次数。
        """
        self.tolerance = tolerance
        self.patience = tolerance_patience
        self.window_size = window_size
        self.increase_patience = increase_patience
        self.backward_window_size = backward_window_size
        self.check_steps = check_steps
        self.metrics = []
        self.patience_counter = 0
        self.increase_patience_counter = 0
        self.best_metric = float('inf')
        self.early_stop = False
        self.verbose = verbose

    def update(self, current_metric):
        """
        更新当前的指标。

        参数:
        - current_metric: float, 当前轮次的指标。
        """
        self.metrics.append(current_metric)

    def check_convergence(self):
        """
        检查是否应该执行早停。

        返回:
        - bool, 是否应该执行早停。
        """

        # warmup stage
        if len(self.metrics) < max(self.check_steps, self.backward_window_size, self.window_size):
            return False

        if len(self.metrics) < self.window_size + self.backward_window_size:
            return False

        # 每check_steps步检查一次指标变化
        if len(self.metrics) % self.check_steps == 0:

            # 计算滑动窗口内的平均指标
            window_metrics = self.metrics[-self.window_size:]
            window_avg = sum(window_metrics) / self.window_size

            backward_window_metrics = self.metrics[
                                      -(self.backward_window_size + self.window_size): -self.backward_window_size
                                      ]

            backward_window_avg = sum(backward_window_metrics) / self.window_size

            # 计算指标变化
            metric_change = abs(window_avg - backward_window_avg)

            if metric_change < self.tolerance:
                self.patience_counter += 1
                # self.tolerance_counter = 0
            else:
                self.patience_counter = 0
                # self.tolerance_counter += 1

            # update the current best metric if current window_avg is smaller
            if window_avg <= self.best_metric:
                self.best_metric = window_avg
                self.increase_patience_counter = 0
            else:
                self.increase_patience_counter += 1

            if self.patience_counter >= self.patience or self.increase_patience_counter >= self.increase_patience:
                self.early_stop = True

            if self.verbose:
                loguru.logger.debug(
                    f"Window Average: {window_avg:.4f}, Backpoint Window Average: {backward_window_avg:.4f}, "
                    f"Best Metric Change: {self.best_metric:.4f}, "
                    f"Metric Change: {metric_change:.4f}, Patience Counter: {self.patience_counter}, "
                    f"Early Stop: {self.early_stop}"
                )

        return self.early_stop

# # 示例用法
# import numpy as np
#
# # 创建EarlyStopping对象
# early_stopping = EarlyStopping(tolerance=1e-4, patience=3, window_size=5, check_steps=5, max_tolerance_steps=3)
#
# # 假设我们有一个训练过程，每轮都会得到一个新的metric
# np.random.seed(0)  # 为了示例的重复性
# metrics = np.random.rand(100)  # 假设这些是每轮的指标（随机生成）
#
# for epoch, metric in enumerate(metrics):
#     early_stopping.update(metric)
#     should_stop = early_stopping.check_convergence()
#     print(f"Epoch {epoch + 1}, Metric: {metric:.4f}, Should Stop: {should_stop}")
#     if should_stop:
#         print(f"Early stopping at epoch {epoch + 1}")
#         break
