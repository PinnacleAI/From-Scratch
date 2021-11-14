
import unittest
import numpy as np
import pandas as pd

from sklearn_from_scratch.base import AbstractClassPredictors
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn


################################################################
# default Errors subclass will be created here if any

class NotFittedError(ValueError):
    pass

################################################################


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

    # if the type of the values in the array is not numeric
    try:
        passed_arg.astype(int)
    except ValueError:
        raise ValueError("Unable to convert array of bytes/strings into decimal numbers with dtype"
                         "='numeric'")

    return passed_arg


class LinearRegression(AbstractClassPredictors):
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
        self.__is_fitted = False

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
        # for validation during prediction. i.e. the same no of feature is used during prediction
        self.__n_features = X.shape[1]

        if len(y) != len(X):
            raise ValueError(f"Found input variables with inconsistent number of samples: [{len(X)}, {len(y)}]")

        # convert y to 2D of shape (n_samples, )
        y = y.reshape(-1, 1)

        if self.fit_intercept:
            X = np.c_[np.ones((len(X), 1)), X]

        # calculate the best params (theta) using normal equation
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        if self.fit_intercept:
            # the first value in self._theta is the param learned on the intercept
            self.coef_ = self._theta[1:].T.ravel()
        else:
            self.coef_ = self._theta.T.ravel()

        self.__is_fitted = True

    def predict(self, X) -> np.ndarray:

        original_X = X

        if not self.__is_fitted:
            raise NotFittedError("This LinearRegression instance is not fitted yet, Call 'fit'"
                                 " with appropriate arguments before using this estimator")

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

        return predict.ravel()

    def get_params(self):
        return {'copy_X': self.copy_X, 'fit_intercept': self.fit_intercept,
                'n_jobs': self.n_jobs, 'normalize': self.normalize, 'positive': self.positive}

    def score(self, X, y, sample_weight=None):
        pred = self.predict(X).ravel()

        if not isinstance(y, (list, pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(f"Expected sequence or array-like, got {type(y)}")

        # check to ensure y is in the proper format
        # if it isn't convert to np.ndarray
        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        y = y.ravel()

        # The score values is calculated using coefficient of determination (1 - u / v)
        # check sklearn.linear_model LinearRegression docs for more details

        u = ((y - pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()

        coef_deter = 1 - (u / v)

        return coef_deter

    def __repr__(self):
        return "LinearRegression()"


######################################## TEST CASES #################################################
# In most case the name of the test are descriptive enough to know what they test for
# a short description will be provided in cases they are not

# the shape of dataset that will be generated throughout the tests
_size = (50000, 50)


class LinearRegressionTest(unittest.TestCase):

    def test_empty_function_call(self):
        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        self.assertEquals(str(reg), str(reg_source))

    def test_get_params_function(self):
        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        self.assertDictEqual(reg.get_params(), reg_source.get_params())

    def test_making_prediction_without_fitting_model(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression(fit_intercept=False)

        try:
            reg.predict(x)
        except NotFittedError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_fitting_with_array_of_str_instead_of_numericals(self):
        x = [['love', 'you', 'dad']]
        y = np.random.randint(1, 5, size=3)

        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        try:
            reg.fit(x, y)
        except ValueError as e:
            err_message = str(e)

        try:
            reg_source.fit(x, y)
        except ValueError as e:
            err_message_source = str(e)

        self.assertEqual(err_message, err_message_source)

    def test_if_linear_regression_fits_the_data_without_raising_error(self):
        x = np.random.randint(1, 10, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression()
        try:
            reg = reg.fit(x, y)
        except Exception:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_if_model_coef_is_equal_to_sklearn_coef_with_fit_intercept_True(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        reg.fit(x, y)
        reg_source.fit(x, y)

        np.testing.assert_almost_equal(reg.coef_, reg_source.coef_)

    def test_if_model_coef_is_equal_to_sklearn_coef_with_fit_intercept_False(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression(fit_intercept=False)
        reg_source = LinearRegressionSklearn(fit_intercept=False)

        reg.fit(x, y)
        reg_source.fit(x, y)


        np.testing.assert_almost_equal(reg.coef_, reg_source.coef_)

    def test_checks_model_predictions_with_fit_intercept_True(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        reg.fit(x, y)
        reg_source.fit(x, y)

        x_new = np.random.randint(1, 100, size=_size)

        pred_train = reg.predict(x)
        pred_test = reg.predict(x_new)

        pred_train_source = reg_source.predict(x)
        pred_test_source = reg_source.predict(x_new)

        np.testing.assert_almost_equal(pred_train, pred_train_source)
        np.testing.assert_almost_equal(pred_test, pred_test_source)

    def test_checks_model_predictions_with_fit_intercept_False(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression(fit_intercept=False)
        reg_source = LinearRegressionSklearn(fit_intercept=False)

        reg.fit(x, y)
        reg_source.fit(x, y)

        x_new = np.random.randint(1, 100, size=_size)

        pred_train = reg.predict(x)
        pred_test = reg.predict(x_new)

        pred_train_source = reg_source.predict(x)
        pred_test_source = reg_source.predict(x_new)

        np.testing.assert_almost_equal(pred_train, pred_train_source)
        np.testing.assert_almost_equal(pred_test, pred_test_source)

    def test_ensures_the_score_function_works__output_correct_results_with_fit_intercept_True(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression()
        reg_source = LinearRegressionSklearn()

        reg.fit(x, y)
        reg_source.fit(x, y)

        train_score = reg.score(x, y)
        train_score_source = reg_source.score(x, y)

        np.testing.assert_almost_equal(train_score, train_score_source)

    def test_ensures_the_score_function_works__output_correct_results_with_fit_intercept_False(self):
        x = np.random.randint(1, 100, size=_size)
        y = np.random.randint(1, 5, size=_size[0])

        reg = LinearRegression(fit_intercept=False)
        reg_source = LinearRegressionSklearn(fit_intercept=False)

        reg.fit(x, y)
        reg_source.fit(x, y)

        a = reg.predict(x)
        b = reg_source.predict(x)

        train_score = reg.score(x, y)
        train_score_source = reg_source.score(x, y)

        np.testing.assert_almost_equal(train_score, train_score_source)


if __name__ == '__main__':
    unittest.main(verbosity=3)
