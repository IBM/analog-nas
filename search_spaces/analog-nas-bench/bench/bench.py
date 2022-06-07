import pandas as pd
import torch 
import ..config_space as CS
from macro_architecture import Network

class AnalogNASBench():
    """ User facing interface for accessing the search space. """

    def __init__(self, dataset_file, surrogate_file, dataset="CIFAR-10"):
        self.cs = CS(dataset)
        self.dataset = pd.DataFrame(dataset_file)
        self.surrogate = torch.load(surrogate_file)

    def check_id(self, id):
        if id in self.dataset['id']:
            return True
        return False

    def query(self, id, only_dataset=False):
        if not self.check_id(id) and not only_dataset:
            print('ID is not in Dataset')
        elif not self.check_id(id):
            print('Generating random architecture')
            config = self.cs.sample_arch()
            arch = Network(config)
            statistics = self.surrogate(config)
            self.print_config(config, statistics)
            return config, arch, statistics
        else: 
            config = self.dataset.iloc(id)
            arch = Network(config)
            self.print_config(config)
            return config, arch


    def print_config(self, config, statistics=None):
        if statistics == None: 
            print('Architecture id {}\n'.format(config["id"]))
            print('\tNumber of parameters = {}'.format(config['nb_param']))
            print('\tNumber of convs = {}'.format(config['nb_conv']))
            print('\tNumber of blocks = {}'.format(config['M']))
            print('\tDepth = {}'.format(config['depth']))
            print('\tFP Accuracy = {}'.format(config['fp_accuracy']))
            print('\tFP HW-trained Accuracy = {}'.format(config['hw_trained_accuracy']))
            print('\tAccuracy 1sec = {}'.format(config['acc_1_sec']))
            print('\tAccuracy 1hour = {}'.format(config['acc_1_hour']))
            print('\tAccuracy 1day = {}'.format(config['acc_1_day']))
            print('\tAccuracy 1month = {}'.format(config['acc_1_month']))

        else: 
            print('Architecture random id {}'.format(statistics[id]))
            print('\tNumber of parameters = {}'.format(config["nb_param"]))
            print('\tNumber of blocks = {}'.format(config['M']))
            print('\tDepth = {}'.format(config["depth"]))
            print('\tPredicted Accuracy = {}'.format(statistics['acc']))
            print('\tPredicted Drop = {}'.format(statistics['drop']))





        


