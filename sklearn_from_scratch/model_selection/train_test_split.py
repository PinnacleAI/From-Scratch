import unittest
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split as train_test_split_sklearn


"""
AUTHOR: Samuel Nkopuruk E.
LinkedIn:  https://www.linkedin.com/in/samuel-effiong-778b23154/

This is an implementation of train_test_split function from sklearn_from_scratch.model_selection module

LIMITATION
1) partial_fit method hasn't yet been implemented, haven't gotten around to learning 
the maths needed to implement that.

"""


################################################################
# helpful Errors subclass will be declared here

class LogicalErrorDetectedHere(ValueError):
    pass


# End declaration
#################################################################


#################################################################
# Helper sub functions are declared here


# PLEASE DON'T START READING FROM HERE, GO TO THE MAIN FUNCTION, THESE ARE
# JUST HELPER FUNCTIONS TO SIMPLIFY AND INCREASE THE READABILITY OF THE MAIN FUNCTION
# `train_test_split`


def _set_unpassed_arg_to_implied_value(passed_arg, n_samples, subset='test'):
    """Compute the value of either train_size or test_size if either one is None
     using the value passed

     E.g
     if train_size = 0.8 and test_size = None, therefore test_size will be
     test_size = 1.0 - train_size (vice_versa)
     """
    unpassed_arg = None

    if isinstance(passed_arg, float):
        _fraction_should_not_be_greater_or_less_than_one_and_zero(passed_arg, n_samples, subset=subset)
        unpassed_arg = 1.0 - passed_arg
    elif isinstance(passed_arg, int):
        _test_train_samples_should_not_be_greater_or_less_than_total_sample_and_zero(passed_arg, n_samples,
                                                                                     subset=subset)
        unpassed_arg = n_samples - passed_arg
    else:
        if unpassed_arg is None:
            raise LogicalErrorDetectedHere("Please review your code, unpassed_arg should not be None")

    return unpassed_arg


def _fraction_should_not_be_greater_or_less_than_one_and_zero(passed_arg: float, n_samples, subset='test'):
    if not isinstance(passed_arg, float):
        raise LogicalErrorDetectedHere("We are evaluating float here, so arg should be of type float")

    if not (0.0 < passed_arg < 1.0):
        raise ValueError(f"{subset}_size={passed_arg} should be either positive and smaller than the number of "
                         f"samples {n_samples} or a float in the (0, 1) range")


def _test_train_samples_should_not_be_greater_or_less_than_total_sample_and_zero(passed_arg, n_samples, subset='test'):
    if not isinstance(passed_arg, int):
        raise LogicalErrorDetectedHere("We are evaluating integers here, so arg should be of type int")

    if not (0 < passed_arg < n_samples):
        raise ValueError(f"{subset}_size={passed_arg} should be either positive and smaller than the number of "
                         f"samples {n_samples} or a float in the (0, 1) range")


def _validate_subset_args(passed_arg, n_samples, subset='test'):
    if not isinstance(passed_arg, (int, float)):
        raise ValueError(f"Invalid value for {subset}_size: {passed_arg}")

    # A generic range validation, it ensures that value passed to test_size or train_size
    # do not exceed (0, n_samples) regardless of required type [int, float],
    # it goes into specificity down the line of codes

    elif passed_arg <= 0.0 or passed_arg >= n_samples:
        raise ValueError(
            f"{subset}_size={passed_arg} should either be positive and smaller than the number of samples {n_samples} "
            "or a float in the (0, 1) range")

    elif isinstance(passed_arg, float):
        _fraction_should_not_be_greater_or_less_than_one_and_zero(passed_arg, n_samples, subset=subset)

    elif isinstance(passed_arg, int):
        _test_train_samples_should_not_be_greater_or_less_than_total_sample_and_zero(passed_arg, n_samples,
                                                                                     subset=subset)


def _compute_final_subset_value(passed_arg, n_samples):
    subset = None
    if isinstance(passed_arg, float):
        subset = round(passed_arg * n_samples)
    elif isinstance(passed_arg, int):
        subset = passed_arg
    else:
        raise LogicalErrorDetectedHere("final value for a subset can't be None")

    return subset


# End declaration
################################################################


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None,
                     shuffle=True):
    ###### PERFORM A SERIES OF VALIDATION ON THE PASSED ARGUMENTS #######

    # TODO: Reduce the function to half its size  {'Difficult_lvl': Expert}

    # 0TH VALIDATION
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # 1ST VALIDATION
    # ensure that all the given positional arguments are arrays or sequences
    for arr in arrays:
        if not isinstance(arr, (np.ndarray, pd.DataFrame, pd.Series, tuple, list, str)):
            raise ValueError("Scalar values cannot be considered a valid collection")

    # 2ND VALIDATION
    # check the number of arrays entered as positional arguments
    # and ensure that they are of the same number of samples
    total = [len(arr) for arr in arrays]

    if len(Counter(total)) > 1:
        raise ValueError(f"Input variables have inconsistent number of samples: {total}")

    # if the arrays of the same length, then save the length of the array
    n_samples = total[0]

    # 3RD VALIDATION
    # set the value of test_size and train_size parameters to their default of 0.25 and 0.75 respectively
    # if they are both None
    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 1 - test_size

    # these default values will be further checked to ensure, that they do not violate
    # their constraint, even tho I am sure they won't

    # 4TH VALIDATION
    # this check instances where only one parameter of the two (test_size or train_size)
    # is given value while the other, is None, it then set the variable to the implied value
    if test_size is not None and train_size is None:
        train_size = _set_unpassed_arg_to_implied_value(test_size, n_samples)

    elif test_size is None and train_size is not None:
        test_size = _set_unpassed_arg_to_implied_value(train_size, n_samples, subset='train')

    # 5TH VALIDATION
    # check the value of test_size and train_size parameters separately and ensure they contain
    # the appropriate value types, valid types [int, float]

    if test_size is not None:
        _validate_subset_args(test_size, n_samples)

    if train_size is not None:
        _validate_subset_args(train_size, n_samples, subset='train')

    # 6TH VALIDATION
    # checks the test_size and train_size interactions, i.e when value is passed to both params
    # if both test_size and train_size are passed values, it ensures that added
    # together they do not exceed their range: (0, 1) if float or (0, n_samples) if int
    # TODO: optimize by finding a way of removing duplicate codes  {'Difficult_lvl': 'Intermediate'}

    if test_size is not None and train_size is not None:

        # if train_size and test_size are both float 
        # ensure that their summed proportion do not exceed 1.0

        def _check_total_range(total_sample, n_sample):
            if total_sample > n_sample:
                raise ValueError(f"The sum of test_size and train_size = {total_sample} should be smaller "
                                 "than the number of samples {n_samples}. Reduce test_size and/or train_size")

        if isinstance(test_size, float) and isinstance(train_size, float):
            # total fractions must equal 1.0
            total_subset_fraction = test_size + train_size
            _check_total_range(total_subset_fraction, 1.0)

        # if test_size is an int and train_size is float
        elif isinstance(test_size, int) and isinstance(train_size, float):
            # check if the absolute number of train_size added to the test_size proportion of the n_sample
            # is greater than the n_sample

            # convert from proportion to n_subset_samples out of the n_samples
            train_size_converted = round(train_size * n_samples)
            n_subset_samples = test_size + train_size_converted

            _check_total_range(n_subset_samples, n_samples)

        # if test_size is a float and train_size is an int
        elif isinstance(test_size, float) and isinstance(train_size, int):
            test_size_converted = round(test_size * n_samples)

            n_subset_samples = train_size + test_size_converted
            _check_total_range(n_subset_samples, n_samples)

        # if both test_size and train_size are both int
        elif isinstance(test_size, int) and isinstance(train_size, int):
            n_subset_samples = test_size + train_size
            _check_total_range(n_subset_samples, n_samples)

    ######## END OF VALIDATION PERFORMED ON THE PASSED ARGUMENTS #########

    # Now that all the necessary preliminaries checks are done, compute the final values of the
    # test_set and training_set

    # compute the test value
    # no need for it though, as we will be using the train_set value for the slicing operation
    # will comment to improve performance

    # test_set = _compute_final_subset_value(test_size, n_samples)

    # compute the train value
    train_set = _compute_final_subset_value(train_size, n_samples)

    # shuffle the dataset if the shuffle argument is True
    indexes = list(range(n_samples))
    if shuffle:
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            indexes = rng.permutation(n_samples)
        else:
            indexes = np.random.permutation(n_samples)

    train_dataset = []
    test_dataset = []

    for arr in arrays:
        if isinstance(arr, (pd.Series, pd.DataFrame)):
            train_dataset.append(arr.iloc[indexes[:train_set]])
            test_dataset.append(arr.iloc[indexes[train_set:]])
        elif isinstance(arr, np.ndarray):
            train_dataset.append(arr[indexes[:train_set]])
            test_dataset.append(arr[indexes[train_set:]])
        else:
            # convert to numpy for faster slicing operation
            arr_np = np.array(arr)

            train_dataset.append(arr_np[indexes[:train_set]].tolist())
            test_dataset.append(arr_np[indexes[train_set:]].tolist())

    return *train_dataset, *test_dataset


######################################## TEST CASES #################################################
# In most case the name of the test are descriptive enough to know what they test for
# a short description will be provided in cases they are not

class TrainTestSplitErrorTest(unittest.TestCase):
    def test_empty_function_call_should_give_the_same_error(self):
        try:
            train_test_split_sklearn()
        except ValueError as e:
            err_message_source = str(e)

        try:
            train_test_split()
        except ValueError as e:
            err_message_custom = str(e)

        self.assertEquals(err_message_custom, err_message_source)

    def test_passed_array_type_should_also_be_return(self):

        array = np.random.randint(1, 1000, size=(1000, 100))
        array_df = pd.DataFrame(array)
        array_np = np.array(array)

        output_arr_custom = train_test_split(array)
        output_arr_df_custom = train_test_split(array_df)
        output_arr_np_custom = train_test_split(array_np)

        self.assertEqual(type(output_arr_custom[0]), type(array))
        self.assertEqual(type(output_arr_df_custom[0]), type(array_df))
        self.assertEqual(type(output_arr_np_custom[0]), type(array_np))

    def test_n_passed_arrays_should_be_split_2n_times(self):
        array = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ]

        for n_arg in range(1, 11):
            arr = [array] * n_arg
            output_arr_custom = train_test_split(*arr)
            self.assertEqual(len(output_arr_custom), 2 * n_arg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
