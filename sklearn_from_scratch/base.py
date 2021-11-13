
from abc import ABCMeta, abstractmethod, ABC


class BasePredictor(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("This method should not be called directly")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("This method should not be called directly")

    @abstractmethod
    def get_params(self):
        raise NotImplementedError("This method should not be called directly")


class BaseTransformer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError("This method should not be called directly")

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError("This method should not be called directly")

    @abstractmethod
    def fit_transform(self, X, y=None):
        raise NotImplementedError("This method should not be called directly")

    @abstractmethod
    def get_params(self):
        raise NotImplementedError("This method should not be called directly")
