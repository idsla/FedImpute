import numpy as np
import pytest
import torch
from torch import nn

from fedimpute.utils.nn_utils import EarlyStopping, load_lr_scheduler, load_optimizer, weights_init

pytestmark = pytest.mark.unit


def _linear_parameters():
    return nn.Linear(2, 1).parameters()


@pytest.mark.parametrize(
    ("optimizer_name", "optimizer_class"),
    [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("sgd", torch.optim.SGD),
        ("asgd", torch.optim.ASGD),
        ("lbfgs", torch.optim.LBFGS),
    ],
)
def test_load_optimizer_returns_requested_torch_optimizer(optimizer_name, optimizer_class):
    optimizer = load_optimizer(
        optimizer_name,
        _linear_parameters(),
        learning_rate=0.01,
        weight_decay=0.1,
    )

    assert isinstance(optimizer, optimizer_class)
    assert optimizer.defaults["lr"] == 0.01
    if optimizer_name != "lbfgs":
        assert optimizer.defaults["weight_decay"] == 0.1


def test_load_optimizer_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="not-supported|not supported"):
        load_optimizer("not-supported", _linear_parameters(), learning_rate=0.01, weight_decay=0.0)


def test_load_lr_scheduler_returns_none_when_scheduler_name_is_none():
    optimizer = torch.optim.SGD(_linear_parameters(), lr=0.1)

    assert load_lr_scheduler(None, optimizer, {}) is None


@pytest.mark.parametrize(
    ("scheduler_name", "params", "scheduler_class", "expected_attr", "expected_value"),
    [
        ("step", {"step_size": 2, "gamma": 0.5}, torch.optim.lr_scheduler.StepLR, "step_size", 2),
        ("exp", {"gamma": 0.5}, torch.optim.lr_scheduler.ExponentialLR, "gamma", 0.5),
        ("cos", {"step_size": 3}, torch.optim.lr_scheduler.CosineAnnealingLR, "T_max", 3),
    ],
)
def test_load_lr_scheduler_returns_requested_scheduler(
    scheduler_name, params, scheduler_class, expected_attr, expected_value
):
    optimizer = torch.optim.SGD(_linear_parameters(), lr=0.1)

    scheduler = load_lr_scheduler(scheduler_name, optimizer, params)

    assert isinstance(scheduler, scheduler_class)
    assert getattr(scheduler, expected_attr) == expected_value


@pytest.mark.parametrize(
    ("scheduler_name", "params", "missing_param"),
    [
        ("step", {"gamma": 0.5}, "step_size"),
        ("exp", {}, "gamma"),
        ("cos", {}, "step_size"),
    ],
)
def test_load_lr_scheduler_reports_missing_required_params(scheduler_name, params, missing_param):
    optimizer = torch.optim.SGD(_linear_parameters(), lr=0.1)

    with pytest.raises(ValueError, match=missing_param):
        load_lr_scheduler(scheduler_name, optimizer, params)


def test_load_lr_scheduler_rejects_unknown_scheduler():
    optimizer = torch.optim.SGD(_linear_parameters(), lr=0.1)

    with pytest.raises(ValueError, match="not-supported|not supported"):
        load_lr_scheduler("not-supported", optimizer, {})


@pytest.mark.parametrize("initializer", ["xavier", "orthogonal", "kaiming"])
def test_weights_init_applies_supported_initializers_to_linear_layers(initializer):
    torch.manual_seed(123)
    layer = nn.Linear(4, 3)
    before = layer.weight.detach().clone()

    weights_init(layer, initializer)

    assert not torch.equal(layer.weight, before)
    assert torch.isfinite(layer.weight).all()


def test_weights_init_rejects_unknown_initializer_for_linear_layers():
    layer = nn.Linear(2, 1)

    with pytest.raises(ValueError, match="Unknown initializer"):
        weights_init(layer, "bad")


def test_weights_init_ignores_non_linear_layers_even_with_unknown_initializer():
    layer = nn.ReLU()

    weights_init(layer, "bad")


def test_early_stopping_waits_until_enough_metrics_are_available():
    early_stopping = EarlyStopping(
        tolerance=1e-4,
        tolerance_patience=1,
        increase_patience=10,
        window_size=2,
        check_steps=1,
        backward_window_size=2,
    )

    for metric in [1.0, 1.0, 1.0]:
        early_stopping.update(metric)
        assert early_stopping.check_convergence() is False


def test_early_stopping_stops_after_tolerance_patience_is_reached():
    early_stopping = EarlyStopping(
        tolerance=1e-4,
        tolerance_patience=2,
        increase_patience=10,
        window_size=2,
        check_steps=1,
        backward_window_size=2,
    )

    results = []
    for metric in [1.0, 1.0, 1.0, 1.0, 1.0]:
        early_stopping.update(metric)
        results.append(early_stopping.check_convergence())

    assert results == [False, False, False, False, True]
    assert early_stopping.patience_counter == 2


def test_early_stopping_stops_after_metric_increases_for_too_long():
    early_stopping = EarlyStopping(
        tolerance=0.0,
        tolerance_patience=10,
        increase_patience=2,
        window_size=2,
        check_steps=1,
        backward_window_size=2,
    )

    results = []
    for metric in [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]:
        early_stopping.update(metric)
        results.append(early_stopping.check_convergence())

    assert results == [False, False, False, False, False, True]
    assert early_stopping.increase_patience_counter == 2


def test_early_stopping_only_checks_on_configured_steps():
    early_stopping = EarlyStopping(
        tolerance=1e-4,
        tolerance_patience=2,
        increase_patience=10,
        window_size=2,
        check_steps=2,
        backward_window_size=2,
    )

    for metric in [1.0, 1.0, 1.0, 1.0, 1.0]:
        early_stopping.update(metric)
        assert early_stopping.check_convergence() is False

    early_stopping.update(1.0)

    assert early_stopping.check_convergence() is True
    assert early_stopping.patience_counter == 2
