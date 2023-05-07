"""
Module for predicting customer churn.
"""

import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from plotting import (plot_histogram, plot_hist_with_kde,
                      plot_normalized_barplot, plot_correlation_heatmap,
                      plot_roc_curve, plot_classification_reports)
import constants
from utils import save_model, load_model, grid_search

# Needed by Udacity platform
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(path: str) -> pd.DataFrame:
    """
    Returns a dataframe for the csv found at `path`.

    Args:
        path: a path to an input csv

    Returns:
        df: pandas dataframe
    """
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
    print("Data shape:", df.shape)
    print("Null values per columns:\n", df.isnull().sum())
    print(df.describe())

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def feature_selection(df: pd.DataFrame, keep_columns: List[str]) -> pd.DataFrame:
    """Utility function for feature selection"""
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

    # TODO where to define?
    out_fn = f"results/{model_name}_results.png"
    plot_classification_reports(train_report, test_report, model_name, out_fn)


# TODO typing
# TODO move to plotting.py?
def feature_importance_plot(model, X_data, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # TODO refactor to work for both models?
    # TODO make a model class; ChurnClassifier? inherit from "ClassifierMixin";
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # NOTE for now only Logistic Regression
    # TODO how to read in params
    model = LogisticRegression(solver='lbfgs', max_iter=3000)

    # TODO hyper param search; define elsewhere
    GRID_SEARCH = False
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    if GRID_SEARCH:
        model = grid_search(model, X_train, y_train, param_grid, cv=5)
    else:
        # Train model without grid search
        model.fit(X_train, y_train)

    # Serialize trained model
    # TODO Make model class?; use model.name
    save_model(model, constants.MODEL_DIR / 'model.pkl')

    # Store model results
    '''
    y_train_preds_lr = None
    y_train_preds_rf = None
    y_test_preds_lr = None
    y_test_preds_rf = None
    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)
    '''

    # Compute and store feature importances
    feature_importance_plot(model = None, X_data=None, output_path=None)


def main() -> None:
    INPUT_PATH = r"./data/bank_data.csv"
    df = import_data(INPUT_PATH)
    perform_eda(df)

    # TODO what does response do? Does this make sense?
    response = constants.RESPONSE
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
    print(X_train.head())

    train_models(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
