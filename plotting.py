"""
Utility module containing plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    fig = plt.figure(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=False,
                cmap='Dark2_r', linewidths = 2)
    save_or_show(out_fn)


def save_or_show(out_fn: str | None) -> None:
    """Utility function to save a plot to an image folder, or show the plot otherwise."""
    if out_fn:
        plt.savefig(IMG_DIR / out_fn)
    else:
        plt.show()
    cleanup()


def cleanup() -> None:
    """Clean up pyplot figures."""
    plt.cla()
    plt.clf()
