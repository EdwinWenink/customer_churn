"""
Utility module containing plotting functions.
"""

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import RocCurveDisplay

from constants import IMG_DIR, DEFAULT_FIG_SIZE

# Apply seaborn styling globally
sns.set()


def plot_histogram(data: pd.Series, figsize=DEFAULT_FIG_SIZE, out_fn: str | None = None,
                   *args, **kwargs) -> None:
    """Utility function to plot a histogram of a pandas series."""
    fig = plt.figure(figsize=figsize)
    plt.title(f"Distribution of {str(data.name).replace('_', ' ')}")

    if 'xlabel' in kwargs.keys():
        plt.xlabel(kwargs.pop('xlabel'))
    if 'ylabel' in kwargs.keys():
        plt.ylabel(kwargs.pop('ylabel'))

    plt.hist(data, **kwargs)
    save_or_show(out_fn)


def plot_normalized_barplot(data: pd.Series, figsize=DEFAULT_FIG_SIZE, out_fn: str | None = None,
                            *args, **kwargs) -> None:
    """Utility function to plot a bar plot using normalization."""
    fig = plt.figure(figsize=figsize)
    data.value_counts(normalize=True).plot(kind='bar')
    plt.title(f"Distribution of {str(data.name).replace('_', ' ')}")

    if 'xlabel' in kwargs.keys():
        plt.xlabel(kwargs.pop('xlabel'))
    if 'ylabel' in kwargs.keys():
        plt.ylabel(kwargs.pop('ylabel'))

    plt.hist(data, **kwargs)
    save_or_show(out_fn)


def plot_hist_with_kde(data: pd.Series, figsize=DEFAULT_FIG_SIZE, out_fn: str | None = None,
                       *args, **kwargs) -> None:
    """Plot a histogram with kernel density estimation."""
    fig = plt.figure(figsize=figsize)

    if 'xlabel' in kwargs.keys():
        plt.xlabel(kwargs.pop('xlabel'))
    if 'ylabel' in kwargs.keys():
        plt.ylabel(kwargs.pop('ylabel'))

    sns.histplot(data, stat='density', kde=True)
    save_or_show(out_fn)


def plot_correlation_heatmap(df: pd.DataFrame, figsize=DEFAULT_FIG_SIZE,
                             out_fn: str | None = None, *args, **kwargs) -> None:
    """Plot a heatmap showing pairwise correlation between all variables."""
    fig = plt.figure(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=False,
                cmap='Dark2_r', linewidths=2)
    save_or_show(out_fn)


def compare_roc_curves(estimators: List[BaseEstimator], X_test: np.ndarray,
                       y_test: np.ndarray, figsize=DEFAULT_FIG_SIZE,
                       out_fn: str | None = None) -> None:
    # TODO module docstring; only appropriate for binary classifier
    # TODO plot_roc_curve deprecated
    # Use: RocCurveDisplay.from_predictions or ..from_estimator()
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for estimator in estimators:
        # plot_roc_curve(estimator, X_test, y_test, ax=ax, alpha=0.8)
        display = RocCurveDisplay.from_estimator(estimator, X_test, y_test,
                                                 ax=ax, alpha=0.8)
        display.plot()
    save_or_show(out_fn)


def plot_classification_reports(train_report: str, test_report: str,
                                model_name: str, out_fn: str) -> None:
    """Save or show a classification report on train and test sets in text format. """
    font_dict = {'fontsize': 10}
    font_properties = 'monospace'  # approach improved by OP -> monospace!
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, f'{model_name} Train', font_dict, font_properties=font_properties)
    plt.text(0.01, 0.05, str(train_report), font_dict, font_properties=font_properties)
    plt.text(0.01, 0.6, f'{model_name} Test', font_dict, font_properties=font_properties)
    plt.text(0.01, 0.7, str(test_report), font_dict, font_properties=font_properties)
    plt.axis('off')
    save_or_show(out_fn)


# TODO typing
def feature_importance_plot(model: BaseEstimator, X_data: pd.DataFrame,
                            output_path: str | None) -> None:
    '''
    Creates and stores the feature importances in `output_path`
    Args:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure
    '''
    # TODO assert has feature_importances_?

    # NOTE in notebook *all* data is thrown in here.
    # Should we not pass train or test separately?

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    # TODO save
    save_or_show(out_path)


def save_or_show(out_fn: str | None) -> None:
    """Save a pyplot figure to an image folder, or show the plot otherwise."""
    if out_fn:
        plt.savefig(IMG_DIR / out_fn)
    else:
        plt.show()
    cleanup()


def cleanup() -> None:
    """Clean up pyplot figures."""
    plt.cla()
    plt.clf()
