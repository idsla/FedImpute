import pytest
from src.fed_imp.sub_modules.missing_simulate.missing_scenario import load_scenario3


def test_load_scenario3_random():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 'random@mrl=0.3_mrr=0.7_mm=mnarlrq'

    ret = load_scenario3(n_clients, cols, strategy)
    print(ret)


def test_load_scenario3_random2():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 'random2@mrl=0.3_mrr=0.7_mm=mnarlrq'

    ret = load_scenario3(n_clients, cols, strategy)
    print(ret)

def test_load_scenario3_s3():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 's3'

    ret = load_scenario3(n_clients, cols, strategy, seed=33)
    print(ret)


def test_load_scenario3_s31():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 's31'

    ret = load_scenario3(n_clients, cols, strategy, seed=33)
    print(ret)


def test_load_scenario3_s32():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 's32'

    ret = load_scenario3(n_clients, cols, strategy, seed=33)
    print(ret)

def test_load_scenario3_s33():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 's33'

    ret = load_scenario3(n_clients, cols, strategy, seed=33)
    print(ret)


def test_load_scenario3_s4():
    n_clients = 10
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    strategy = 's4'

    ret = load_scenario3(n_clients, cols, strategy, seed=0)
    print(ret)
