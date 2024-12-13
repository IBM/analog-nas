"""Abstract class for base evaluator."""
from analogainas.utils import kendal_correlation


"""Base class for Accuracy Evaluation Methods."""
class Evaluator:
    def __init__(self, model_type=None):
        self.model_type = model_type

    def query(self, x_test):
        """
        Get the accuracy/rank prediction for x_test.

        Args:
            x_test: list of architectures

        Returns:
            Predictions for the architectures
        """
        pass


    def set_hyperparams(self, hyperparams):
        """
        Modifies/sets hyperparameters of the evaluator.

        Args:
            hyperparams: dictionary of hyperparameters.
        """
        self.hyperparams = hyperparams

    def get_hyperparams(self):
        """
        Get the hyperparameters of the evaluator.

        Returns:
            A dictionary of hyperparameters.
            If not manually set, a dictionary of the default hyperparameters.
        """
        if hasattr(self, "hyperparams"):
            return self.hyperparams
        else:
            return None

    def get_correlation(self, x_test, y_test):
        y_pred = self.query(x_test)
        return kendal_correlation(y_test, y_pred)

