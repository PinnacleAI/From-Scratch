
################################################################
# default Errors subclass will be created here if any

class NotFittedError(ValueError):
    pass
################################################################




class Standard_Scaler:
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._is_fitted = False
    
    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.

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
        orgnal_X = X
        
        # if X is a list convert to numpy arrary for better crunching
        if isinstance(X, list):
            X = np.array(X)
            
        if isinstance(X, pd.Series):
            X = X.values
            
        # if the array is not 2D 
        if len(X.shape) < 2:
            raise ValueError(f"Expected 2D array, got 1D array instead: {orgnal_X} \n"
                            "Reshape your data either using array.reshape(-1, 1)")        
            
        # calculate the mean and std for each of the feature
        features_mean = []
        features_std = []
        
        for feature in range(X.shape[2]):
            features_mean.append(np.mean(X[:, feature]))
            features_std.append(np.std(X[:, feature]))
            
        self._feature_mean = features_mean
        self._features_std = features_std
        
        self._is_fitted = True
        
        return self
    
    def transform(self, X, copy=None):
        # save a reference to the original argument passed
        X_orgnal = X
        
        # 1ST VALIDATION
        # ensure that this StandardScaler instance has already been fitted
        if not self._is_fitted:
            raise NotFittedError("This StandardScaler instance is not fitted yet. " 
                                 "Call 'fit' with appropriate arguments before using this estimator")
        
        
        # 2ND VALIDATION
        # again check to see if the passed argument is of 2-Dimensional
        # TODO: optimize by finding a way of removing duplicate codes  {'Difficult_lvl': 'Intermediate'}
        
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
            
        if len(X.shape) < 2:
            raise ValueError(f"Expected 2D array, got 1D array instead: {orgnal_X} \n"
                            "Reshape your data either using array.reshape(-1, 1)")        
            
        # check to see if the passed argument have the same number of features
        # as the original dataset it was fitted on
        
        if X.shape[1] != len(self.features_mean):
            raise ValueError(f"X has {X.shape[1]} features, but StandScaler is expecting {len(self.features_mean)} features as input")
            
        
        
    
    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y, fit_params).transform(X)
    
    def inverse_transform(self, X, copy=None):
        raise NotImplementedError("This feature have not been implemented, pls check back later")
    
    def partial_fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError("This feature have not been implemented, pls check back later")
