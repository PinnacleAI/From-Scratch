import unittest
import numpy as np
import pandas as pd

from sklearn_from_scratch.base import AbstractClassTransformers
from sklearn.preprocessing import StandardScaler as StandardScalerSklearn


"""
AUTHOR: Samuel Nkopuruk E.
LinkedIn:  www.linkedin.com/samuel

This is an implementation of StandardScaler class from sklearn_from_scratch.preprocessing module

LIMITATION
1) partial_fit method hasn't yet been implemented, haven't gotten around to learning 
the maths needed to implement that.

2) Unable to scale a dataset using sample_weights

"""


################################################################
# default Errors subclass will be created here if any

class NotFittedError(ValueError):
    pass

################################################################


#################################################################
# Helper sub functions are declared here


# PLEASE DON'T START READING FROM HERE, GO TO THE MAIN FUNCTION, THESE ARE
# JUST HELPER FUNCTIONS TO SIMPLIFY AND INCREASE THE READABILITY OF THE MAIN FUNCTION
# `StandardScaler`

def _check_if_arg_is_in_supported_format(passed_arg) -> None:
    # list of supported type
    supported_type = (list, np.ndarray, pd.Series, pd.DataFrame)
    if not isinstance(passed_arg, supported_type):
        raise ValueError(f"Invalid argument")


def _convert_arg_to_numpy_arrays(passed_arg) -> np.ndarray:
    if isinstance(passed_arg, list):
        passed_arg = np.array(passed_arg)
    elif isinstance(passed_arg, (pd.Series, pd.DataFrame)):
        passed_arg = passed_arg.values

    return passed_arg


def _ensure_that_array_is_in_2D(original_array, passed_arg: np.ndarray):
    if len(passed_arg.shape) < 2:
        raise ValueError(f"Expected 2D array, got 1D array instead: {original_array} \n"
                         "Reshape your data either using array.reshape(-1, 1)")


def _check_if_instance_is_fitted(is_fitted):
    if not is_fitted:
        raise NotFittedError("This StandardScaler instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this estimator.")


#################################################################


class StandardScaler(AbstractClassTransformers):
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        super(StandardScaler, self).__init__()
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fitted = False

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling, even when mean or std is False

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : None
            Ignored. I don't use this feature
        """

        # keep a reference to the original array for error reporting
        original_X = X
        _check_if_arg_is_in_supported_format(X)

        X = _convert_arg_to_numpy_arrays(X)

        # if the array is not 2D
        _ensure_that_array_is_in_2D(original_X, X)

        features_mean = []
        features_std = []

        # TODO: optimize by finding a way of optimizing this for loop  {'Difficult_lvl': 'Intermediate'}
        for feature in range(X.shape[1]):
            features_mean.append(np.mean(X[:, feature]))
            features_std.append(np.std(X[:, feature]))

        self._features_mean = features_mean
        self._features_std = features_std

        self._is_fitted = True

        return self

    def transform(self, X, copy=None) -> np.ndarray:
        # save a reference to the original argument passed
        X_original = X
        if self.copy:
            X = X.astype(float)

        # 1ST VALIDATION
        # ensure that this StandardScaler instance has already been fitted
        _check_if_instance_is_fitted(self._is_fitted)
        _check_if_arg_is_in_supported_format(X)

        # 2ND VALIDATION
        X = _convert_arg_to_numpy_arrays(X)

        _ensure_that_array_is_in_2D(X_original, X)

        # check to see if the passed argument have the same number of features
        # as the original dataset it was fitted on

        if X.shape[1] != len(self._features_mean):
            raise ValueError(
                f"X has {X.shape[1]} features, but StandScaler is expecting {len(self._features_mean)} features as input")

        # TODO: optimize this for loop  {'Difficult_lvl': 'Intermediate'}
        # scale the features using the formula (X - mean) / std
        for index in range(len(self._features_mean)):
            X[:, index] = (X[:, index] - self._features_mean[index]) / self._features_std[index]

        return X

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X)

    def inverse_transform(self, X, copy=None) -> np.ndarray:
        X_original = X
        X = X.copy()

        _check_if_instance_is_fitted(self._is_fitted)
        _check_if_arg_is_in_supported_format(X)

        X = _convert_arg_to_numpy_arrays(X)
        _ensure_that_array_is_in_2D(X_original, X)

        if X.shape[1] != len(self._features_mean):
            raise ValueError(
                f"X has {X.shape[1]} features, but StandScaler is expecting {len(self._features_mean)} features as input")

        for index in range(len(self._features_mean)):
            X[:, index] = (X[:, index] * self._features_std[index]) + self._features_mean[index]

        return X

    def partial_fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError("This feature have not been implemented, pls check back later")

    def get_params(self) -> dict:
        return {'copy': self.copy, 'with_mean': self.with_mean, 'with_std': self.with_std}

    def __repr__(self):
        return "StandardScaler()"


######################################## TEST CASES #################################################
# In most case the name of the test are descriptive enough to know what they test for
# a short description will be provided in cases they are not

class StandardScalerTest(unittest.TestCase):

    def test_instantiating_class_without_any_argument(self):
        scaler = StandardScaler()
        scaler_source = StandardScalerSklearn()

        self.assertEqual(scaler.with_std, scaler_source.with_std)
        self.assertEqual(scaler.with_mean, scaler_source.with_mean)
        self.assertEqual(str(scaler), str(scaler_source))

    def test_transforming_scaler_without_fitting_first(self):
        scaler = StandardScaler()
        scaler_source = StandardScalerSklearn()

        data = np.random.randint(1, 1000, size=(100, 100))

        try:
            scaler.transform(data)
        except ValueError as e:
            err_message = str(e)

        try:
            scaler_source.transform(data)
        except ValueError as e:
            err_message_source = str(e)

        self.assertEqual(err_message, err_message_source)

    def test_ensures_input_and_computed_output_have_the_same_shape(self):
        scaler = StandardScaler()

        data = np.random.randint(1, 10000, size=(1000, 100))
        data_transformed = scaler.fit_transform(data)

        self.assertTupleEqual(data.shape, data_transformed.shape)

    def test_transformed_values_should_be_equal(self):
        scaler = StandardScaler()
        scaler_source = StandardScalerSklearn()

        data = np.random.randint(1, 10000, size=(1000, 100))

        scaler.fit(data)
        scaler_source.fit(data)

        data_transformed = scaler.transform(data)
        data_transformed_source = scaler_source.transform(data)

        np.testing.assert_almost_equal(data_transformed, data_transformed_source)

    def test_ensures_the_fit_transform_function_works_properly(self):
        scaler = StandardScaler()
        scaler_source = StandardScalerSklearn()

        data = np.random.randint(1, 1000, size=(1000, 100))

        data_transformed = scaler.fit_transform(data)
        data_transformed_source = scaler.fit_transform(data)

        np.testing.assert_almost_equal(data_transformed, data_transformed_source)

    def test_ensures_the_inverse_transform_function_works_properly(self):
        scaler = StandardScaler()
        scaler_source = StandardScalerSklearn()

        data = np.random.randint(1, 1000, size=(1000, 100))

        data_transformed = scaler.fit_transform(data)
        data_inverse = scaler.inverse_transform(data_transformed)

        np.testing.assert_almost_equal(data_inverse, data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
