
from abc import abstractmethod
from sklearn_from_scratch.base import AbstractClassPredictors


class AbstractClassLinearModel(AbstractClassPredictors):
    @abstractmethod
    def decision_function(self, X):
        raise NotImplementedError("This should not be implemented directly")
