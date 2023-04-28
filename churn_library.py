"""
TODO Module docstring should go here.
"""

import os
from typing import List, Tuple
# import shap
# import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

from constants import CAT_COLUMNS, QUANT_COLUMNS
from plotting import plot_histogram, plot_normalized_barplot, plot_hist_with_kde, plot_correlation_heatmap


# Used within Udacity learning platform
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
    print("Data shape:", df.shape)
    print("Null values per columns:\n", df.isnull().sum())
    print(df.describe())
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


def encoder_helper(df: pd.DataFrame, category_lst: List[str], response: str) -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    Args:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    Returns:
            df: pandas dataframe with new columns
    '''
    pass

    return df


def perform_feature_engineering(df: pd.DataFrame, response: str) -> Tuple[np.array, np.array,
                                                                          np.array, np.array]:
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    pass


# TODO typing
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

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
    pass


if __name__ == '__main__':
    INPUT_PATH = r"./data/bank_data.csv"
    df = import_data(INPUT_PATH)
    perform_eda(df)
