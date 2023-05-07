from typing import Tuple
import warnings
import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from shap import Explainer
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from utils import grid_search

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logging.getLogger(__name__)


class ChurnModel():
    """
    Light wrapper around sklearn estimators that defines a single interface
    for all models used in `churn_library`. Includes support for grid search
    and Shap explainers.
    """

    def __init__(self, estimator: BaseEstimator, param_grid: dict | None = None,
                 cv: int = 5, shap_explainer: Explainer | None = None) -> None:
        """
        Args:
            model: sklearn estimator with fit and predict functions
            param_grid: if provided, grid search will be performed using these parameters
            cv: amount of cross validation splits, if CV is performed.
            shap_explainer: optionally provide a suitable Shap explainer for this model
        """
        # TODO ChurnClassifier? inherit from "ClassifierMixin";
        self._estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self._name = type(self._estimator).__name__
        self._shap_explainer = shap_explainer

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
        logging.warning("Model %s needs to be fitted before calculating Shap values.", self.name)
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
            self._estimator = grid_search(self._estimator, X_train, y_train,
                                          self.param_grid, cv=self.cv)
        else:
            # Train model without grid search
            self._estimator = self._estimator.fit(X_train, y_train)

    def predict(self, X) -> np.ndarray:
        return self._estimator.predict(X)
