
import unittest
import numpy as np
import pandas as pd

from sklearn_from_scratch.base import BasePredictor
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn


def _check_if_arg_is_in_supported_format(passed_arg):
    # list of supported type
    supported_type = (list, np.ndarray, pd.Series, pd.DataFrame)
    if not isinstance(passed_arg, supported_type):
        raise ValueError(f"X must be a array-like")


def _ensure_that_array_is_in_2D(original_array, passed_arg):
    if len(passed_arg.shape) < 2:
        raise ValueError(f"Expected 2D array, got 1D array instead: {original_array} \n"
                         "Reshape your data either using array.reshape(-1, 1)")


def _convert_arg_to_numpy_arrays(passed_arg) -> np.ndarray:
    if isinstance(passed_arg, list):
        passed_arg = np.array(passed_arg)
    elif isinstance(passed_arg, (pd.Series, pd.DataFrame)):
        passed_arg = passed_arg.values

    return passed_arg


class LinearRegression(BasePredictor):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False):
        """
        Check the sklearn.linear_model.LinearRegression docs for specifications on the parameters.
        It uses the Normal equation (a closed-form solution) to computes the model parameters that
        best fit the model to the training set.

        :param fit_intercept: bool
        :param normalize: bool
        :param copy_X: bool
        :param n_jobs:
        :param positive: bool
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y):
        """
        Fit linear model

        Parameters
        ----------
        X :
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples,)
        :return:
        """
        original_X = X

        # perform validation on the X param
        _check_if_arg_is_in_supported_format(X)
        X = _convert_arg_to_numpy_arrays(X)
        _ensure_that_array_is_in_2D(original_X, X)

        # perform validation on the y param
        _check_if_arg_is_in_supported_format(y)
        y = _convert_arg_to_numpy_arrays(y)
        y = y.ravel()

        # save the number of features present in the dataset
        # for validation during prediction
        self.__n_features = X.shape[1]

        if len(y) != len(X):
            raise ValueError(f"Found input variables with inconsistent number of samples: [{len(X)}, {len(y)}]")

        # convert y to 2D of shape (n_samples, )
        y = y.reshape(-1, 1)

        if self.fit_intercept:
            X = np.c_[np.ones((len(X), 1)), X]

        # calculate the best params (theta) using normal equation
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot()

    def predict(self, X) -> np.ndarray:
        original_X = X

        # perform validation on the X param
        _check_if_arg_is_in_supported_format(X)
        X = _convert_arg_to_numpy_arrays(X)
        _ensure_that_array_is_in_2D(original_X, X)

        # check if X has the same number of features that the model was trained on
        if X.shape[1] != self.__n_features:
            raise ValueError("matmul: Input operand 1 has a mismatch in its core dimension 0")

        if self.fit_intercept:
            X = np.c_[np.ones((len(X), 1)), X]

        predict = X.dot(self._theta)

        return predict

    def get_params(self):
        return {'fit_intercept': self.fit_intercept, 'normalize': self.normalize,
                'copy_X': self.copy_X, 'n_jobs': self.n_jobs, 'positive': self.positive}
