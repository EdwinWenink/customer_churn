from typing import Tuple
import warnings
import logging

import numpy as np
import shap
from shap import Explainer
from shap.maskers import Masker
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from utils import grid_search

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
            self._estimator = grid_search(self._estimator, X_train, y_train,
                                          self.param_grid, cv=self.cv)
        else:
            # Train model without grid search
            self._estimator = self._estimator.fit(X_train, y_train)

        # Once the model is fitted, we can instantiate the explainer
        if self._shap_explainer:
            try:
                self._shap_explainer = self._shap_explainer(self._estimator,
                                                            masker=self._shap_masker)
            except NotImplementedError:
                logger.error("The provided masker was not valid for the chosen Shap explainer.")
                # As a fallback, see if there is a default masker for this model and explainer
                self._shap_explainer = self._shap_explainer(self._estimator, masker=None)

    def predict(self, X) -> np.ndarray:
        preds = self._estimator.predict(X)
        print(f"{self.name}: Distribution of predictions:",
              dict(zip(*np.unique(preds, return_counts=True))))  # pragma: notype
        logger.info("%s: Distribution of predictions: %s", self.name, )
        return preds
