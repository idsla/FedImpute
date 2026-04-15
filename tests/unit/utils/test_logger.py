import sys

import loguru
import pytest

from fedimpute.utils.logger import setup_logger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_loguru_logger():
    yield
    loguru.logger.remove()
    loguru.logger.add(sys.stderr)


def test_setup_logger_verbose_zero_silences_logger(capsys):
    setup_logger(verbose=0)

    loguru.logger.success("not emitted")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_setup_logger_verbose_one_emits_success_but_not_info(capsys):
    setup_logger(verbose=1)

    loguru.logger.success("success emitted")
    loguru.logger.info("info hidden")

    captured = capsys.readouterr()
    assert "success emitted" in captured.out
    assert "info hidden" not in captured.out


def test_setup_logger_verbose_two_emits_info(capsys):
    setup_logger(verbose=2)

    loguru.logger.info("info emitted")

    assert "info emitted" in capsys.readouterr().out


def test_setup_logger_verbose_three_emits_debug(capsys):
    setup_logger(verbose=3)

    loguru.logger.debug("debug emitted")

    assert "debug emitted" in capsys.readouterr().out
