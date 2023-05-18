"""
Configuration file for pytest.
"""

import logging
from typing import Callable

import pytest

import churn_library as cls


def pytest_configure(config):
    """Setup pytest namespace variables"""
    # TODO do something with config?
    pytest.df = df_plugin()
    pytest.X_train = df_plugin()
    pytest.y_train = df_plugin()
    pytest.X_test = df_plugin()
    pytest.y_test = df_plugin()


def pytest_runtest_setup(item):
    """Initialize logger per test item."""
    # Get the logger for the current test
    logger = logging.getLogger(item.nodeid)

    # Log a message for the start of the test
    logger.info("Starting test: %s", item.name)


def df_plugin():
    return None


@pytest.fixture
def valid_input_path():
    return "./data/bank_data.csv"


"""
# You can test more paths with the following syntax
@pytest.fixture(params=["./data/bank_data.csv", "./data/bank_data_copy.csv"])
def valid_input_path(request):
    return request.param
"""


@pytest.fixture
def invalid_input_path():
    return "./data/bullshit.csv"


@pytest.fixture(scope="module")
def import_data() -> Callable:
    return cls.import_data


@pytest.fixture(scope="module")
def perform_eda() -> Callable:
    return cls.perform_eda


@pytest.fixture(scope="module")
def encoder_helper() -> Callable:
    return cls.encoder_helper


@pytest.fixture(scope="module")
def perform_feature_engineering() -> Callable:
    return cls.perform_feature_engineering


@pytest.fixture(scope="module")
def train_models() -> Callable:
    return cls.train_models