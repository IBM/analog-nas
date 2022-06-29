import numpy as np
import random


class Hyperparameter:
    """
    Class defines a hyperparameter and its range.
    """
    def __init__(self, name, type, range=None, min_value=0, max_value=0):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.range = range
        if self.range is not None:
            self.sampling = "range"
        else:
            self.type = type  # Discrete, continuous, categorical
            self.sampling = "uniform"

    def sample_hyp(self):
        if self.sampling == "range":
            return random.choice(self.range)
        if self.type == "discrete":
            return np.random.randint(self.min_value, high=self.max_value)
        if self.type == "continuous":
            return np.random.uniform(self.min_value, high=self.max_value)


class ConfigSpace:
    """
    This class defines the search space.
    """
    def __init__(self, dataset):
        self.dataset = dataset  # VWW, KWS
        self.search_space = "resnet-like"  # for now only resnet-like
        self.hyperparameters = []  # list of Hyperparameters to search for
        self.set_hyperparameters()

    def add_hyperparameter(self, name, type, min_value, max_value):
        hyp = Hyperparameter(name,
                             type,
                             min_value=min_value,
                             max_value=max_value)
        self.hyperparameters.append(hyp)

    def add_hyperparameter_range(self, name, type, range):
        hyp = Hyperparameter(name, type, range=range)
        self.hyperparameters.append(hyp)

    def sample_arch(self):
        arch = {}
        for hyp in self.hyperparameters:
            arch[hyp.name] = hyp.sample_hyp()
        return arch

    def sample_arch_uniformly(self, n):
        archs = []
        for i in range(n):
            tmp = self.sample_arch()
            for j in range(5, tmp["M"], -1):
                tmp["convblock%d" % j] = 0
                tmp["widenfact%d" % j] = 0
                tmp["B%d" % j] = 0
                tmp["R%d" % j] = 0
            archs.append(tmp)

        return archs

    def set_hyperparameters(self):
        if self.search_space == "resnet-like":
            self.add_hyperparameter_range("out_channel0",
                                          "discrete",
                                          [8, 12, 16, 32, 48, 64])
            self.add_hyperparameter("M", "discrete", 1, 5)
            self.add_hyperparameter("R1", "discrete", 1, 16)
            self.add_hyperparameter("R2", "discrete", 0, 16)
            self.add_hyperparameter("R3", "discrete", 0, 16)
            self.add_hyperparameter("R4", "discrete", 0, 16)
            self.add_hyperparameter("R5", "discrete", 0, 16)

            for i in range(1, 6):
                self.add_hyperparameter_range("convblock%d" % i,
                                              "discrete",
                                              [1, 2])
                self.add_hyperparameter("widenfact%d" % i,
                                        "continuous",
                                        0.5,
                                        0.8)
                self.add_hyperparameter("B%d" % i, "discrete", 1, 5)


def main():
    CS = ConfigSpace("Cifar-10")
    configs = CS.sample_arch_uniformly(20)
    print(configs)


if __name__ == "__main__":
    main()
