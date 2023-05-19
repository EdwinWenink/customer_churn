"""
Configuration file for pytest.

Author: Edwin Wenink
Date: May 2023
"""

import logging
from typing import Callable

import pytest

import churn_library as cls


def pytest_configure(config):
    """Setup pytest namespace variables"""
    pytest.df = None
    pytest.X_train = None
    pytest.y_train = None
    pytest.X_test = None
    pytest.y_test = None


def pytest_runtest_setup(item):
    """Initialize logger per test item."""
    # Get the logger for the current test
    logger = logging.getLogger(item.nodeid)

    # Log a message for the start of the test
    logger.info("Starting test: %s", item.name)


@pytest.fixture
def valid_input_path():
    """Defines a valid data input path."""
    return "./data/bank_data.csv"


@pytest.fixture
def invalid_input_path():
    """Defines an invalid data input path."""
    return "./data/bullshit.csv"


@pytest.fixture(scope="module")
def import_data() -> Callable:
    """Returns data importing function."""
    return cls.import_data


@pytest.fixture(scope="module")
def perform_eda() -> Callable:
    """Returns function to perform EDA."""
    return cls.perform_eda


@pytest.fixture(scope="module")
def encoder_helper() -> Callable:
    """Returns encoding function."""
    return cls.encoder_helper


@pytest.fixture(scope="module")
def perform_feature_engineering() -> Callable:
    """Returns feature engineering function."""
    return cls.perform_feature_engineering


@pytest.fixture(scope="module")
def train_models() -> Callable:
    """Returns function to train models."""
    return cls.train_models
