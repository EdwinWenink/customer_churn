"""
Generic utilities used in `churn_library.py`.
"""

import joblib
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


# TODO maybe move logger here? Or `get_logger`


# TODO what is type of model?
def save_model(model: Any, save_path: str):
    """
    Save the model object using joblib. If you provide a file extension
    the best compression method is determined automatically.
    E.g. the object is pickled is `save_path` is `model.pkl`.
    This function overwrites files.

    Args:
        model: python object of the model
        save_path: string indicating where to save the model, including file extension.
    """
    try:
        joblib.dump(model, save_path)
    except (FileNotFoundError, KeyError) as err:
        # TODO log error
        # logging.error("During model saving the following error occurred: %s", err)
        print(err)


def load_model(model_path: str):
    """
    Load a model previously written to disk using `save_model` using joblib.

    Args:
        model_path: string indicating where model is stored on disk.

    """
    try:
        return joblib.load(model_path)
    except (FileNotFoundError, KeyError, UnicodeDecodeError, ValueError) as err:
        # TODO log error
        # logging.error("During model loading the following error occurred: %s", err)
        print(err)


def grid_search(estimator: BaseEstimator, X_train, y_train, param_grid: dict, cv: int | None, **kwargs) -> BaseEstimator:
    """Perform grid search with cross validation and return the best estimator."""
    # Perform hyperparameter search
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, **kwargs)
    grid.fit(X_train, y_train)

    # Return the best estimator
    return grid.best_estimator_
