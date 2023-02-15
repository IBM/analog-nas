import numpy as np
import random


class Hyperparameter:
    """
    Class defines a hyperparameter and its range.
    """
    def __init__(self, name, type, range=None, min_value=0, max_value=0, step=1):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step = step 
        self.range = range
        if self.range is not None:
            self.sampling = "range"
        else:
            self.type = type  # Discrete, continuous
            self.sampling = "uniform"

    def sample_hyp(self):
        if self.sampling == "range":
            return random.choice(self.range)
        if self.type == "discrete":
            return np.random.randint(self.min_value, high=self.max_value)
        if self.type == "continuous":
            return np.random.uniform(self.min_value, high=self.max_value)

    def size(self):
        if self.sampling == "range":
            return len(self.range)
        if self.type == "continuous":
            return 1
        return len(np.arange(self.min_value, self.max_value, self.step))

    def __repr__(self) -> str:
        return "Name: {}\nMin_Value:{}\nMax_value:{}\nStep:{}".format(
            str(self.name), str(self.min_value), str(self.max_value), str(self.step)
        )



class ConfigSpace:
    """
    This class defines the search space.
    """
    def __init__(self, dataset="CIFAR-10"):
        self.dataset = dataset  # VWW, KWS
        self.search_space = "resnet-like"  # for now only resnet-like
        self.hyperparameters = []  # list of Hyperparameters to search for
        self.set_hyperparameters()

    def add_hyperparameter(self, name, type, min_value, max_value, step=1):
        for h in self.hyperparameters:
            if h.name == name:
                raise Exception("Name should be unique!")

        hyp = Hyperparameter(name,
                             type,
                             min_value=min_value,
                             max_value=max_value, 
                             step=step)
        self.hyperparameters.append(hyp)

    def add_hyperparameter_range(self, name, type, range):
        for h in self.hyperparameters:
            if h.name == name:
                raise Exception("Name should be unique!")

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
                                          range=[8, 12, 16, 32, 48, 64])
            self.add_hyperparameter("M", "discrete", min_value=1, max_value=5)
            self.add_hyperparameter("R1", "discrete", min_value=1, max_value=16)
            self.add_hyperparameter("R2", "discrete", min_value=0, max_value=16)
            self.add_hyperparameter("R3", "discrete", min_value=0, max_value=16)
            self.add_hyperparameter("R4", "discrete", min_value=0, max_value=16)
            self.add_hyperparameter("R5", "discrete", min_value=0, max_value=16)

            for i in range(1, 6):
                self.add_hyperparameter_range("convblock%d" % i,
                                              "discrete",
                                              range=[1, 2])
                self.add_hyperparameter("widenfact%d" % i,
                                        "continuous",
                                        min_value=0.5,
                                        max_value=0.8)
                self.add_hyperparameter("B%d" % i, "discrete", min_value=1, max_value=5)

    def remove_hyperparameter(self, name):
        for i, h in enumerate(self.hyperparameters):
            if h.name == name:
                self.hyperparameters.remove(h)
                break

    def compute_cs_size(self):
        size = 1
        for h in self.hyperparameters:
            size *= h.size()
        return size

    def get_hyperparameters(self):
        l = []
        for h in self.hyperparameters:
            l.append(h.name)
        print(l)

    def __repr__(self) -> str:
        str_ = ""
        str_ += "Architecture Type: {}\n".format(self.search_space)
        str_ += "Search Space Size: {}\n".format(self.compute_cs_size())
        str_ += "------------------------------------------------\n"
        for i, h in enumerate(self.hyperparameters):
            str_ += "{})\n".format(i) + str(h) + "\n\n"
        str_ += "------------------------------------------------\n"
        return str_ 

def main():
    CS = ConfigSpace("Cifar-10")
    configs = CS.sample_arch_uniformly(20)
    print(configs)


if __name__ == "__main__":
    main()
