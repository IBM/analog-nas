"""Hyperparameter Configuration Space."""

from analogainas.search_spaces.config_spaces.base_config_space import BaseConfigSpace


class RPUConfigSpace(BaseConfigSpace):
    def __init__(self, dataset="CIFAR-10"):
        # initialize super
        super().__init__("rpu-like", dataset)
        self.dataset = dataset  # VWW, KWS
        self.search_space = "rpu-like"
        self.hyperparameters = []  # list of Hyperparameters to search for
        self.set_hyperparameters()

    def sample_arch_uniformly(self, n):
        archs = []
        for i in range(n):
            tmp = self.sample_arch()
            archs.append(tmp)
        return archs

    def set_hyperparameters(self):
        self.add_hyperparameter_range("g_max", "discrete", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("tile_size", "discrete", [64, 128, 256, 512, 1024, 2048])
        self.add_hyperparameter_range("dac_resolution", "discrete", [32, 64, 128, 256, 512, 1024, 2048])
        self.add_hyperparameter_range("adc_resolution", "discrete", [32, 64, 128, 256, 512, 1024, 2048])