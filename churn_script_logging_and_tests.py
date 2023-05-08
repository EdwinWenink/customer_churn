"""
Module for testing and logging `churn_library.py`.
"""

import os
import logging

import pytest


logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler('./logs/test_churn_library.log'))


def test_import(import_data, valid_input_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        pytest.df = import_data(valid_input_path)
        logger.info("Data successfully imported from %s", valid_input_path)
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file %s wasn't found", valid_input_path)
        raise err

    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_import_invalid_input(import_data, invalid_input_path):
    with pytest.raises(FileNotFoundError):
        import_data(invalid_input_path)

"""
def test_eda(perform_eda):
    '''
    test perform eda function
    '''


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    pass
"""