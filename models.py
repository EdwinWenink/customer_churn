import warnings
import logging
import joblib
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from shap import Explainer
from shap.maskers import Masker
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logger = logging.getLogger(__name__)


class ChurnClassifier():
    """
    Light wrapper around sklearn estimators that defines a single interface
    for all models used in `churn_library`. Includes support for grid search
    and Shap explainers.
    """

    def __init__(self, estimator: BaseEstimator, param_grid: dict | None = None,
                 cv: int = 5, shap_explainer: Explainer | None = None,
                 shap_masker: Masker | None = None) -> None:
        """
        Args:
            model: sklearn estimator with fit and predict functions
            param_grid: if provided, grid search will be performed using these parameters
            cv: amount of cross validation splits, if CV is performed.
            shap_explainer: optionally provide a suitable Shap explainer for this model
                NOTE that this should be passed as an uninitialized class, because the explainer
                can only be instantiated on a model *after* the model is fitted.
            shap_masker: some shap explainers may need an explicit definition of a masker, see docs.
        """
        self._estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self._name = type(self._estimator).__name__
        self._shap_explainer = shap_explainer
        self._shap_masker = shap_masker

    @property
    def name(self) -> str:
        """Return the name of the used model."""
        return self._name

    @property
    def estimator(self) -> BaseEstimator:
        """Getter for the used estimator."""
        return self._estimator

    @property
    def shap_explainer(self) -> Explainer | None:
        """Return the defined Shap explainer, if any, and if model is fitted."""
        if (hasattr(self.estimator, "coef_")
            or (hasattr(self.estimator, "estimators_")
                and len(self.estimator.estimators_) > 0)):
            return self._shap_explainer

        print(f"Model {self.name} needs to be fitted before calculating Shap values.")
        logger.warning("Model %s needs to be fitted before calculating Shap values.", self.name)
        check_is_fitted(self.estimator)
        return None

    def train(self, X_train, y_train) -> None:
        """
        Train the chosen estimator. If `param_grid` is defined,
        these parameters will be used for grid search.
        The model with the highest score will be used.
        """

        if self.param_grid:
            # Only perform grid search if search parameters where provided
            # The grid search returns the estimator with the highest CV score
            logger.info("Fitting model using grid search.")
            self._estimator = grid_search(self._estimator, X_train, y_train,
                                          self.param_grid, cv=self.cv)
        else:
            # Train model without grid search
            logger.info("Fitting model.")
            self._estimator = self._estimator.fit(X_train, y_train)

        # Once the model is fitted, we can instantiate the explainer
        if self._shap_explainer:
            try:
                # Explainer cannot be None here so mypy warning can be ignored
                self._shap_explainer = self.shap_explainer(  # type: ignore
                    self._estimator, masker=self._shap_masker)
                logger.info("Initializing Shap explainer after model fitting.")
            except NotImplementedError:
                logger.error("The provided masker was not valid for the chosen Shap explainer.")
                # As a fallback, see if there is a default masker for this model and explainer
                self._shap_explainer = self._shap_explainer(self._estimator, masker=None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.info("Predicting on data with shape %s", X.shape)
        preds = self._estimator.predict(X)
        preds_distribution = dict(zip(*np.unique(preds, return_counts=True)))  # type: ignore
        logger.info("%s: Distribution of predictions: %s", self.name, preds_distribution)
        return preds


def save_model(model: ChurnClassifier, save_path: str | Path):
    """
    Save the model object using joblib. If you provide a file extension
    the best compression method is determined automatically.
    E.g. the object is pickled is `save_path` is `model.pkl`.
    This function overwrites files.

    Args:
        model: ChurnModel
        save_path: string indicating where to save the model, including file extension.
    """
    try:
        joblib.dump(model, save_path)
        logging.info("Persisted model %s to disk at %s", model.name, save_path)
    except (FileNotFoundError, KeyError) as err:
        logging.error("During model saving the following error occurred: %s", err)


def load_model(model_path: str | Path) -> Any:
    """
    Load a model previously written to disk using `save_model` using joblib.

    Args:
        model_path: string indicating where model is stored on disk.

    Returns:
        Model python object

    """
    try:
        return joblib.load(model_path)
    except (FileNotFoundError, KeyError, UnicodeDecodeError, ValueError) as err:
        print(err)
        logging.error("During model saving the following error occurred: %s", err)


def grid_search(estimator: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray,
                param_grid: dict, cv: int | None, **kwargs) -> BaseEstimator:
    """
    Perform grid search with cross validation and return the best estimator.

    Args:
        estimator: scikit-learn BaseEstimator or subclass.
        X_train: input feature array.
        y_train: array of target labels.
        param_grid: parameters with value ranges to do grid search over.
        cv: the amount of splits in cross-validation.

    Returns:
        the BaseEstimator with the best cross validation score.

    """
    # Perform hyperparameter search
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, **kwargs)
    grid.fit(X_train, y_train)

    # Return the best estimator
    return grid.best_estimator_
