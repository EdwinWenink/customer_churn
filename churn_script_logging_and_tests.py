"""
Module for testing and logging `churn_library.py`.
"""

import os
import logging
import tempfile
from pathlib import Path

import pytest
import shap
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier

import src.constants as constants
from src.models import ChurnClassifier


logger = logging.getLogger(__name__)
if constants.VERBOSE:
    logger.addHandler(logging.StreamHandler())


def test_import(import_data, valid_input_path):
    '''
    Test data import for a correct input.
    '''
    try:
        # Stores the dataframe in the pytest namespace for later use
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
        logger.info("Testing whether FileNotFoundError is raised for invalid input.")


def test_eda(perform_eda):
    '''
    Tests whether EDA plots are stored in the expected place.
    Stores EDA plots in a temporary folder in order to not pollute
    the actual pipeline outputs.
    '''
    # Verify input is a dataframe
    assert isinstance(pytest.df, pd.DataFrame)

    # Make a temp directory that is cleaned up when done.
    with tempfile.TemporaryDirectory() as temp_dir:

        # Perform EDA on df from pytest namespace
        logger.info("Performing EDA and storing results in temp folder.")
        perform_eda(pytest.df, temp_dir)

        # Assert outputs are written to the expected locations
        try:
            assert os.path.exists(f'{temp_dir}/heatmap.png')
            assert os.path.exists(f'{temp_dir}/churn_distribution.png')
            assert os.path.exists(f'{temp_dir}/marital_status_distribution.png')
            assert os.path.exists(f'{temp_dir}/total_trans_ct_distribution.png')
            logger.info("All EDA outputs were correctly stored.")
        except AssertionError as err:
            logger.error("EDA output is missing at the expected location.")


def test_encoder_helper(encoder_helper):
    '''
    Test whether categorical columns are encoded correctly.
    The output columns will be encoded as `{column name}_{response name}`.
    The encoded columns should also have a numeric dtype.
    '''

    # Verify input is a dataframe
    assert isinstance(pytest.df, pd.DataFrame)

    # Test encoding for all categorical features
    categorical_columns = constants.CAT_COLUMNS
    response = constants.RESPONSE
    df_encoded = encoder_helper(pytest.df, categorical_columns, response)
    for cat_col in categorical_columns:
        encoded_cat_col = f"{cat_col}_{response}"
        try:
            assert encoded_cat_col in df_encoded.columns
            assert is_numeric_dtype(df_encoded[encoded_cat_col].dtype)
            logger.info("%s correctly encoded as %s", cat_col, encoded_cat_col)
        except AssertionError as err:
            logger.error("%s not encoded correctly.", cat_col)


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test whether the feature engineering step produces data sets
    with expected features, shapes, and dtypes.
    '''

    # Verify input is a dataframe
    try:
        assert isinstance(pytest.df, pd.DataFrame)
        logger.info("Input to feature engineering is a dataframe as expected.")
    except AssertionError as err:
        logger.error("Feature engineering expects a dataframe as input.")
        raise err

    # Save results in pytest namespace
    pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test =\
        perform_feature_engineering(pytest.df, constants.RESPONSE)

    # Test if features are missing
    expected_features_set = set(constants.SELECT_FEATURES)
    try:
        assert set(pytest.X_train.columns) == expected_features_set
        assert set(pytest.X_test.columns) == expected_features_set
        logger.info("The expected features are present in X_train and X_test.")
    except AssertionError as err:
        logger.error("The expected features are not present in X_train and X_test.")
        raise err

    # Assert that the target label vectors have the correct shape
    try:
        assert pytest.y_train.shape == (len(pytest.X_train),)
        assert pytest.y_test.shape == (len(pytest.X_test),)
        logger.info("The target label vectors have the correct shape.")
    except AssertionError as err:
        logger.error("The target label vectors do not have the correct shape.")
        raise err

    # Check output schema of dataset features
    try:
        assert pytest.X_train.dtypes.apply(is_numeric_dtype).all()
        assert pytest.X_test.dtypes.apply(is_numeric_dtype).all()
        logger.info("All dataset dtypes are numeric as expected.")
    except AssertionError as err:
        logger.error("Some dataset dtypes are not numeric.")
        raise err

    # Check output schema of response vectors
    try:
        assert is_numeric_dtype(pytest.y_train)
        assert is_numeric_dtype(pytest.y_test)
        logger.info("All response vectors are numeric as expected.")
    except AssertionError as err:
        logger.error("A response vector is not numeric.")
        raise err


def test_train_models(train_models):
    '''
    Test the steps of `train_models`:

    - Model saved in right place
    - Classification report in the right place
    - Feature importance plot in the right place
    - RoC curve comparison in the right place
    '''

    # Define and train a simple LogisticRegression model, incl. Shap explainer.
    estimator = RandomForestClassifier(random_state=42, n_estimators=10)
    model = ChurnClassifier(
        estimator=estimator,
        shap_explainer=shap.TreeExplainer  # works without passing masker
    )

    model_name = type(estimator).__name__
    logger.info("Model name: %s", model_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        train_models([model], pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test,
                     model_dir=temp_dir, img_dir=temp_dir)
        logger.info("Trained %s", model_name)
        logger.info("Contents of temp_dir %s:", os.listdir(temp_dir))
        logger.info("Contents of temp_dir/results %s:", os.listdir(temp_dir / 'results'))

        # Test model is saved in the correct place with the specified filename format.
        try:
            model_path = temp_dir / f'{model.name}.pkl'
            assert os.path.exists(model_path)
            logger.info("Model is saved in the expected location %s", model_path)
        except AssertionError as err:
            logger.error("Model file was not found in the expected location %s", model_path)
            raise err

        # Test feature importance plots
        try:
            assert os.path.exists(temp_dir / f"results/{model_name}_feature_importances.png")
            assert os.path.exists(temp_dir / f"results/{model_name}_shap_feature_importances.png")
            logger.info("Feature importance plots found at expected location.")
        except AssertionError as err:
            logger.error("Feature importance plots not found at expected location.")
            raise err

        # Test classification report
        try:
            classification_report_path = temp_dir / f"results/{model_name}_results.png"
            assert os.path.exists(classification_report_path)
            logger.info("Classification report found at expected location %s.",
                        classification_report_path)
        except AssertionError as err:
            logger.error("Classification report not found at %s.",
                         classification_report_path)
            raise err

        # ROC curve comparison
        try:
            roc_path = temp_dir / f"results/ROC_RandomForestClassifier.png"
            assert os.path.exists(roc_path)
            logger.info("ROC curve found at expected location %s.", roc_path)
        except AssertionError as err:
            logger.error("ROC curve not found at %s.", roc_path)
            raise err


if __name__ == "__main__":
    # This testing suite does not follow pytest naming conventions, but I'm required
    # to hand this specific file in. So I make the module discoverable as such:
    pytest.main([__file__])
