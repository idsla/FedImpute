import os
import random

import numpy as np
import pytest
import torch

from fedimpute.utils.reproduce_utils import set_seed, setup_clients_seed, setup_seeds

pytestmark = pytest.mark.unit


def test_setup_seeds_generates_deterministic_client_offsets():
    assert setup_seeds(10, 4) == [10, 1257, 2504, 3751]


def test_setup_clients_seed_draws_requested_number_of_rng_seeds():
    rng = np.random.default_rng(2026)
    expected = list(np.random.default_rng(2026).integers(0, 10000, 3))

    client_seeds = setup_clients_seed(3, rng)

    assert client_seeds == expected
    assert len(client_seeds) == 3
    assert all(0 <= seed < 10000 for seed in client_seeds)


def test_set_seed_makes_numpy_python_and_torch_randomness_reproducible(monkeypatch):
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark

    try:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)

        set_seed("123")
        np_first = np.random.random(3)
        python_first = [random.random() for _ in range(3)]
        torch_first = torch.rand(3)

        set_seed(123)

        assert np.allclose(np.random.random(3), np_first)
        assert [random.random() for _ in range(3)] == python_first
        assert torch.allclose(torch.rand(3), torch_first)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert os.environ["PYTHONHASHSEED"] == "123"
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
