def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, 
                     shuffle=True):
    
    ###### PERFORM A SERIES OF VALIDATION ON THE PASSED ARGUMENTS #######
    
    # TODO: Reduce the function to half it size  {'Difficult_lvl': Expert}
    
    
    # 1ST VALIDATION
    # first of all ensure that all the given positional arguments are arrays or sequences
    for arr in arrays:
        if not isinstance(arr, (np.ndarray, pd.DataFrame, pd.Series, tuple, list, str)):
            raise ValueError("Scalar values cannot be considered a valid collection")
    
    
    # 2ND VALIDATION
    # check the number of arrays entered as positional arguments
    # and ensure that they are of the same number of samples
    # TODO: optimize this block of code into a list comprehension  {'difficult_lvl': 'beginner'}
    total = []
    for arr in arrays:
        total.append(len(arr))
    n_samples = total[0]
    if len(Counter(total)) > 1:
        raise ValueError(f"Input variables have inconsistent number of samples: {total}")
        
    
    # 3RD VALIDATION
    # check if both the test_size and train_size parameters both have the values of 
    # None. If so, set them to their default values
    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 1 - test_size
        
    # these default values will be further checked to ensure, that they do not violate
    # their constraint, even tho I am sure they won't
    
    
    # 4TH VALIDATION
    # this check instances where only one parameter of the two (test_size, train_size) 
    # is given value while the other, is None, then set the variable to the implied value
    if test_size is not None and train_size is None:
        if isinstance(test_size, float):
            train_size = 1.0 - test_size
        elif isinstance(test_size, int):
            train_size = n_samples - test_size

    elif test_size is None and train_size is not None:
        if isinstance(train_size, float):
            test_size = 1.0 - train_size
        elif isinstance(train_size, int):
            test_size = n_samples - train_size
            
        
    # 5TH VALIDATION
    # check the value of test_size and train_size parameters separately and ensure they contain
    # the appropriate value types, valid types [int, float]
    # TODO: optimize by finding a way of removing duplicate codes  {'Difficult_lvl': 'Intermediate'}
    
    if test_size is not None:
        if not isinstance(test_size, (int, float)):
            raise ValueError(f"Invalid value for test_size: {test_size}")
        # A generic range validation, it ensures that value passed to test_size or train_size
        # do not exceed (0, n_samples) regardless of required type [int, float],
        # it goes into specificity down the line of codes
        elif test_size <= 0.0 or test_size >= n_samples:
            raise ValueError(f"test_size={test_size} should either be positive and smaller than the number of samples {total[0]} "
            "or a float in the (0, 1) range")
        elif isinstance(test_size, float):
            if not (test_size > 0.0 and test_size < 1.0):
                raise ValueError(f"test_size={test_size} should be either positive and smaller than the number of "
                "samples {n_samples} or a float in the (0, 1) range")
        elif isinstance(test_size, int):
            if not (test_size > 0 and test_size < n_samples):
                raise ValueError(f"test_size={test_size} should be either positive and smaller than the number of "
                "samples {n_samples} or a float in the (0, 1) range")
            
    if train_size is not None:
        if not isinstance(train_size, (int, float)):
            raise ValueError(f"Invalid value for train_size: {train_size}")
        elif train_size <= 0 or train_size >= n_samples:
            raise ValueError(f"train_size={train_size} should either be positive and smaller than the number of samples {total[0]} "
            "or a float in the (0, 1) range")
        elif isinstance(train_size, float):
            if not (train_size > 0.0 and train_size < 1.0):
                raise ValueError(f"train_size={train_size} should be either positive and smaller than the number of "
                "samples {n_samples} or a float in the (0, 1) range")
        elif isinstance(train_size, int):
            if not (train_size > 0 and train_size < n_samples):
                raise ValueError(f"train_size={train_size} should be either positive and smaller than the number of "
                "samples {n_samples} or a float in the (0, 1) range")
            
    
    # 6TH VAIDATION
    # checks the test_size and train_size interactions, i.e when value is passed to both params
    # if both test_size and train_size are passed values, it ensures that added
    # together it do not exceed their limit: (0, 1) if float or (0, n_samples) if int
    # TODO: optimize by finding a way of removing duplicate codes  {'Difficult_lvl': 'Intermediate'}
            
    if test_size is not None and train_size is not None:
        
        # if train_size and test_size are both float 
        # ensure that their summed proportion do not exceed 1.0
        if isinstance(test_size, float) and isinstance(train_size, float):
            # total fractions must equal 1.0
            total_fraction = test_size + train_size
            if total_fraction > 1.0:
                raise ValueError(f"The sum of test_size and train_size = {total_fraction}, should be in the "
                "(0, 1) range. Reduce test_size and/or train_size")
                
        # if test_size is an int and train_size is float
        elif isinstance(test_size, int) and isinstance(train_size, float):
            # check if the absolute number of train_size added to the test_size proportion of the n_sample
            # is greater than the n_sample
            
            # convert from proportion to samples out of the n_samples
            train_size_converted = round(train_size * n_samples)
            print(train_size_converted)
            total_sample = test_size + train_size_converted
            
            # if greater than n_samples reduce the absolute value of the test_size
            if total_sample > n_samples:
                raise ValueError(f"The sum of test_size and train_size = {total_sample} should be smaller "
                    "than the number of samples {n_samples}. Reduce test_size and/or train_size")
                
        # if test_size is a float and train_size is an int
        elif isinstance(test_size, float) and isinstance(train_size, int):
            test_size_converted = round(test_size * n_samples)
            print(test_size_converted)
            total_sample = train_size + test_size_converted
            if total_sample > n_samples:
                raise ValueError(f"The sum of test_size and train_size = {total_sample} should be smaller "
                    "than the number of samples {n_samples}. Reduce test_size and/or train_size")
        
        # if both test_size and train_size are both int
        elif isinstance(test_size, int) and isinstance(train_size, int):
                total_sample = test_size + train_size
                if total_sample > n_samples:
                    raise ValueError(f"The sum of test_size and train_size = {total_sample} should be smaller "
                    "than the number of samples {n_samples}. Reduce test_size and/or train_size")
                    
    ######## END OF VALIDATION PERFORMED ON THE PASSED ARGUMENTS #########
    
            
    # Now that all the necessary preliminaries checks are done, get the final values of the 
    # test_set and training_set
    if isinstance(test_size, float):
        test_set = round(test_size * n_samples)
    elif isinstance(test_size, int):
        test_set = test_size
        
    if isinstance(train_size, float): 
        train_set = round(train_size * n_samples)
    elif isinstance(train_size, int):
        train_set = train_size
        
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
            
            train_dataset.append(arr_np[indexes[:train_set]])
            test_dataset.append(arr_np[indexes[train_set:]])

    return *train_dataset, *test_dataset
