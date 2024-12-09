"""Hyperparameter Configuration Space."""
from analogainas.search_spaces.config_space.interface.config_space import ConfigSpace

from analogainas.search_spaces.config_space.base_config_space import BaseConfigSpace


class AutoEncoderConfigSpace(BaseConfigSpace):
    """
    This class defines the search space.
    """
    def __init__(self, dataset="CIFAR-10"):
        # initialize super
        super().__init__("autoencoder-like", dataset)
        self.dataset = dataset  # VWW, KWS
        self.search_space = "autoencoder-like"
        self.hyperparameters = []  # list of Hyperparameters to search for
        self.set_hyperparameters()

    def sample_arch_uniformly(self, n):
        raise Exception("Not implemented")

    def set_hyperparameters(self):
        raise Exception("Not implemented")
