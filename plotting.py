"""
Utility module containing plotting functions.

TODO FIX: many empty plots are generated.
TODO FIX: classification report title seems to fall off screen.
"""

from typing import List
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from sklearn.base import BaseEstimator
from sklearn.metrics import RocCurveDisplay
from shap import Explainer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

from constants import IMG_DIR, DEFAULT_FIG_SIZE

# Apply seaborn styling globally
sns.set()

# SHAP throws numba deprecation warnings; suppress until fix is available.
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logger = logging.getLogger(__name__)


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
    # TODO module docstring; only appropriate for probabilistic binary classifier with predict_proba
    # TODO plot_roc_curve deprecated
    # Use: RocCurveDisplay.from_predictions or ..from_estimator()
    logger.info("Plotting ROC curve comparison for binary classifiers.")
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for estimator in estimators:
        display = RocCurveDisplay.from_estimator(estimator, X_test, y_test,
                                                 ax=ax, alpha=0.8)
        # display.plot()
    save_or_show(out_fn)


def plot_classification_reports(train_report: str, test_report: str,
                                model_name: str, out_fn: str | None = None) -> None:
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


def feature_importance_plot(model: BaseEstimator, X_data: pd.DataFrame,
                            shap_explainer: Explainer | None = None,
                            output_path: str | None = None) -> None:
    '''
    Creates and stores the feature importances in `output_path`
    Args:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure
    '''
    model_name = type(model).__name__

    # Plot Shapely values if a Shap explainer is provided
    if shap_explainer:
        try:
            assert isinstance(shap_explainer, shap.Explainer), "Explainer needs to be initialized."
        except AssertionError:
            logger.error("Shap explainer is not initialized yet.")
            return

        logger.info("Generating feature importance plot for %s "
                    "with Shapley values", model_name)
        shap_values = shap_explainer.shap_values(X=X_data)
        shap.summary_plot(shap_values, X_data, plot_type="bar")

    # If the model has a native feature method for computing
    # feature importances, use that.
    elif hasattr(model, "feature_importances_"):
        logger.info("Generating feature importance plot for %s "
                    "using sklearn `feature_importances_", model_name)
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
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        save_or_show(output_path)
    else:
        logger.warning("Estimator %s does not have `feature_importances_` implemented "
                       "and no Shap Explainer was provided.", model_name)

    # Save or show feature importance plot
    save_or_show(output_path)


def save_or_show(out_fn: str | None) -> None:
    """Save a pyplot figure to an image folder, or show the plot otherwise."""
    if out_fn:
        out_path = IMG_DIR / out_fn
        logger.info("Saving figure at %s", out_path)
        plt.savefig(out_path)
    else:
        plt.show()
    cleanup()


def cleanup() -> None:
    """Clean up pyplot figures."""
    plt.cla()
    plt.clf()
