from analogainas.search_spaces.resnet_macro_architecture import Network
from analogainas.search_spaces.config_space import ConfigSpace
from analogainas.search_algorithms.worker import Worker
from analogainas.search_spaces.train import train
from analogainas.utils import *
import csv

EPOCHS = 40
LEARNING_RATE = 0.05


def random_sample(dataset, n):
    """Randomly samples n architectures from ConfigSpace."""
    cs = ConfigSpace(dataset)
    sampled_architectures = cs.sample_arch_uniformly(n)

    keys = sampled_architectures[0].keys()

    for config in sampled_architectures:
        model = Network(config)
        model_name = "resnet_{}_{}".format(config["M"], get_nb_convs(config))

        with open("./configs/"+model_name+".config",
                  'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(config)

        train(model, model_name, LEARNING_RATE, EPOCHS)


def ea_sample(dataset, n, n_iter):
    """Samples n architectures from ConfigSpace
    using an evolutionary algorithm."""
    cs = ConfigSpace(dataset)
    worker = Worker(dataset, cs, 3, n_iter)
    worker.search(population_size=n)
