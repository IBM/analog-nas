import os
import csv
import numpy as np

from analogainas.search_spaces.config_space import ConfigSpace
from analogainas.search_spaces.resnet_macro_architecture import Network

"""Wrapper class to launch NAS search."""
class Worker():
    def __init__(self, 
                 cs: ConfigSpace=None,
                 eval = None,
                 optimizer=None,
                 runs=5,
                 max_budget=1,
                 n_iter=100):
        self.max_budget = max_budget
        self.n_iter = n_iter
        self.config_space = cs
        self.evaluation = eval
        self.optimizer=optimizer
        self.runs = runs
        self.best_config = None
        self.best_acc  = 0
        self.std_err = 0

    @property 
    def best_arch(self):
        return Network(self.best_config)

    def search(self):
        os.mkdir("results")
        print("Result directory created.\n")

        results = []
        for i in range(self.runs):
            print("Search {} started".format(i))
            best_config, best_acc = self.optimizer.run(self.config_space)
            
            with open('results/best_results_{}.csv'.format(i), 'w') as f:
                for key in best_config.keys():
                    f.write("%s,%s\n"%(key,best_config[key]))

            results.append(best_acc)
            if best_acc > self.best_acc:
                self.best_config = best_config
                self.best_acc = best_acc

            print("Best Acc = {}".format(best_acc))
        self.std_err = np.std(results, ddof=1) / np.sqrt(np.size(results))

        print("SEARCH ENDED")

    def result_summary(self):
        print("Best architecture accuracy: ", self.best_acc)
        print(f"Standard deviation of accuracy over {self.runs} runs: {self.best_acc}")
