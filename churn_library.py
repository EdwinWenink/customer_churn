"""
TODO Module docstring should go here.
"""

import os
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from plotting import (plot_histogram, plot_hist_with_kde,
                      plot_normalized_barplot, plot_correlation_heatmap)
import constants

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


def encoder_helper(df: pd.DataFrame, category_list: List[str], response: str) -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    Args:
        df: pandas dataframe
        category_list: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
                  for naming variables or index y column]

    Returns:
            df: pandas dataframe with encoded categorical columns
    '''
    # TODO REFACTOR FROM HERE
    # 1. Make more efficient
    # 2. Make function (4x duplicated)

    # Gender encoded column
    gender_lst = []
    gender_groups = df.groupby('gender').mean(numeric_only=True)['churn']

    for val in df['gender']:
        gender_lst.append(gender_groups.loc[val])

    df['gender_churn'] = gender_lst

    # Education encoded column
    edu_lst = []
    edu_groups = df.groupby('education_level').mean()['churn']

    for val in df['education_level']:
        edu_lst.append(edu_groups.loc[val])

    df['education_level_churn'] = edu_lst

    # Marital encoded column
    marital_lst = []
    marital_groups = df.groupby('marital_status').mean()['churn']

    for val in df['marital_status']:
        marital_lst.append(marital_groups.loc[val])

    df['marital_status_churn'] = marital_lst

    # Income encoded column
    income_lst = []
    income_groups = df.groupby('income_category').mean()['churn']

    for val in df['income_category']:
        income_lst.append(income_groups.loc[val])

    df['income_category_churn'] = income_lst

    # card encoded column
    card_lst = []
    card_groups = df.groupby('card_category').mean()['churn']

    for val in df['card_category']:
        card_lst.append(card_groups.loc[val])

    df['card_category_churn'] = card_lst

    # TODO REFACTOR ABOVE

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

    # TODO what does response do?
    df = encoder_helper(df, constants.CAT_COLUMNS, response)

    # Determine training features `X` and target labels `y`
    X = feature_selection(df, constants.KEEP_COLUMNS)
    y = df['churn']

    # Pick a 70/30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def feature_selection(df: pd.DataFrame, keep_columns: List[str]) -> pd.DataFrame:
    """Utility function for feature selection"""
    return df[keep_columns]


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

    response = 'what does this do?'
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
    print(X_train.head())
