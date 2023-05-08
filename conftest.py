"""
Configuration file for pytest.
"""

import logging

import pytest
import churn_library as cls


def pytest_configure(config):
    """
    # Set up the root logger
    logging.basicConfig(
        filename='./logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')
    """
    pytest.df = df_plugin()


def pytest_runtest_setup(item):
    # Get the logger for the current test
    logger = logging.getLogger(item.nodeid)

    # Log a message for the start of the test
    logger.info("Starting test: %s", item.name)

def df_plugin():
    return None




@pytest.fixture
def valid_input_path():
    return "./data/bank_data.csv"


@pytest.fixture
def invalid_input_path():
    return "./data/bullshit.csv"


# TODO is this an appropriate usage of fixtures?
# @pytest.fixture(scope="module", params=[VALID_PATH, INVALID_PATH])
@pytest.fixture(scope="module")
def import_data():
    return cls.import_data
    # return cls.import_data(request.param)