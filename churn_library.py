"""
Module for predicting customer churn.
"""

import warnings
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from plotting import (plot_histogram, plot_hist_with_kde,
                      plot_normalized_barplot, plot_correlation_heatmap,
                      compare_roc_curves, plot_classification_reports,
                      feature_importance_plot)
import constants
from models import ChurnClassifier, save_model

# SHAP throws numba deprecation warnings; suppress until fix is available.
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Only needed by Udacity platform
# import os
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def import_data(path: str) -> pd.DataFrame:
    """
    Returns a dataframe for the csv found at `path`.

    Args:
        path: a path to an input csv

    Returns:
        df: pandas dataframe
    """
    logger.info("Reading data from path %s", path)
    df = pd.read_csv(path)
    df = preprocess_data(df)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe `df` in place.

    Args:
        df: input dataframe

    Returns:
        df: dataframe with lower case column names, unneeded columns dropped,
            and a `churn` column, the target variable in this project.
    """
    logger.info("Preprocessing dataframe.")

    # The unnamed column is a duplicate of the index
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Consistent lower case for column names
    df.rename(columns=lambda x: x.lower(), inplace=True)

    # Create a new column specifying the churn as a binary variable
    df['churn'] = df['attrition_flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(df: pd.DataFrame) -> None:
    '''
    Perform EDA on `df` and save figures to images folder.

    Args:
        df: pandas dataframe
    '''
    # General statistics
    logger.info("Data shape: %s", df.shape)
    logger.info("Null values per columns:\n%s", df.isnull().sum())
    logger.info(df.describe())

    # EDA plots
    plot_histogram(df['churn'], bins=np.arange(df['churn'].min()-.1, df['churn'].max()+.1, .1),
                   align='left', xlabel='Value', ylabel='Counts',
                   out_fn=f"eda/{df['churn'].name}_distribution.png")

    plot_histogram(df['customer_age'], xlabel='Value', ylabel='Counts',
                   out_fn=f"eda/{df['customer_age'].name}_distribution.png")

    plot_normalized_barplot(df['marital_status'],
                            out_fn=f"eda/{df['marital_status'].name}_distribution.png")

    plot_hist_with_kde(df['total_trans_ct'],
                       out_fn=f"eda/{df['total_trans_ct'].name}_distribution.png")

    plot_correlation_heatmap(df, out_fn=f"eda/heatmap.png")


def encoder_helper(df: pd.DataFrame, cat_columns: List[str],
                   response: str) -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    Args:
        df: pandas dataframe
        cat_columns: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
                  for naming variables or index y column]

    Returns:
            df: pandas dataframe with the new columns
    '''
    logger.info("Encoding categorical columns %s", cat_columns)
    for cat_col in cat_columns:
        mean_churn_per_group = df.groupby(cat_col).mean(numeric_only=True)[response]
        df[f"{cat_col}_{response}"] = df[cat_col].map(mean_churn_per_group)

    return df


def perform_feature_engineering(df: pd.DataFrame, response: str) ->\
      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    df = encoder_helper(df, constants.CAT_COLUMNS, response)

    # Determine training features `X` and target labels `y`
    X = feature_selection(df, constants.SELECT_FEATURES)
    y = df['churn']

    # Pick a 70/30 train-test split
    test_size = 0.3
    random_state = 42
    logger.info("Train-test split with test proportion %s (random state = %s)",
                test_size, random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test


def feature_selection(df: pd.DataFrame, keep_columns: List[str]) -> pd.DataFrame:
    """Utility function for feature selection"""
    logger.info("Selecting features: %s", keep_columns)
    return df[keep_columns]


def classification_report_image(model_name: str,
                                y_train: np.ndarray,
                                y_test: np.ndarray,
                                y_train_preds: np.ndarray,
                                y_test_preds: np.ndarray,
                                ) -> None:
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder.

    Args:
            model_name: model name that will be used in the report subtitles
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from random forest
            y_test_preds: test predictions from random forest

    '''

    train_report = classification_report(y_train, y_train_preds, output_dict=False)
    test_report = classification_report(y_test, y_test_preds, output_dict=False)

    logger.info("Generating classification report on train and test set.")
    out_fn = f"results/{model_name}_results.png"
    plot_classification_reports(train_report, test_report, model_name, out_fn)


def train_models(models: List[ChurnClassifier],
                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    '''
    Train models and compare their results. The models are persisted to disk.
    Model results are saved as images.

    Args:
        models: list of `ChurnModel` objects with a fit and predict function
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    logger.info("Start training loop for models: %s", [model.name for model in models])

    for model in models:

        # Train and serialize trained model
        model.train(X_train, y_train)
        save_model(model, constants.MODEL_DIR / f'{model.name}.pkl')

        # Store model results
        y_train_preds = model.predict(X_train)
        y_test_preds = model.predict(X_test)

        # Generate a classification report on train and test set
        classification_report_image(model.name, y_train, y_test,
                                    y_train_preds, y_test_preds)

        # Compute and store feature importances
        feature_importance_plot(model=model._estimator, X_data=X_test,
                                output_path=None, shap_explainer=model.shap_explainer)

    # Plot ROC curves of both models in the same plot
    compare_roc_curves([model._estimator for model in models], X_test, y_test)


def main() -> None:
    INPUT_PATH = r"./data/bank_data.csv"
    df = import_data(INPUT_PATH)
    perform_eda(df)

    # Perform feature engineering and split into train and test set
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, constants.RESPONSE)

    # Define a Logistic Regression Model
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = ChurnClassifier(
        estimator=LogisticRegression(solver='lbfgs', max_iter=3000),
        shap_explainer=shap.LinearExplainer,
        shap_masker=shap.maskers.Partition(X_train, max_samples=10000)
    )

    # Define a Random Forest classifier + Grid Search hyperparameter tuning
    rfc_param_grid = {
        # 'n_estimators': [200, 500],
        'n_estimators': [10, 20],
        'max_features': ['sqrt'],
        # 'max_depth': [4, 5, 100],
        'max_depth': [4, 5],
        'criterion': ['gini', 'entropy']
    }

    rfc = ChurnClassifier(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rfc_param_grid,
        cv=5,
        shap_explainer=shap.TreeExplainer  # works without passing masker
    )

    # Train and evaluate all defined models
    models = [lrc, rfc]
    train_models(models, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
