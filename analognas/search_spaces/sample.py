import torchvision.models as models

from resnet_macro_architecture import Network
from config_space import ConfigSpace
from train_ import train
from utils import * 

from search_algorithm.worker import worker

import os
import argparse
import random
import csv 

EPOCHS = 40 
LEARNING_RATE = 0.05 

'''
Randomly samples n architectures from ConfigSpace.
'''
def random_sample(dataset, n): 
    cs = ConfigSpace(dataset)
    sampled_architectures = cs.sample_arch_uniformly(n) 
    
    keys = sampled_architectures[0].keys()
    
    for config in sampled_architectures:
        model = Network(config) 
        model_name = "resnet_{}_{}".format(config["M"], get_nb_convs(config)) 
        
        with open("./configs/"+model_name+".config", 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(config)
            
        train(model, model_name, LEARNING_RATE, EPOCHS)
        
'''
Samples n architectures from ConfigSpace using an evolutionary algorithm.
'''
def ea_sample(dataset, n, n_iter):
    cs = ConfigSpace(dataset) 
    worker = Worker(dataset, cs, 3, n_iter) 
    
    worker.ea_search(population_size=n)

def main():
    random_sample("VWW", 30) 
        
if __name__ == "__main__":
    main()
        