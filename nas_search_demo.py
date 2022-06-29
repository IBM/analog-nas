from analognas.search_spaces import config_space 
from analognas.evaluators import XGBoostEvaluator
from analognas.search_algorithms import EAOptimizer, Worker
from analognas.search_spaces.resnet_macro_architecture import Network

CS = config_space('CIFAR-10')
surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True)
optimizer = EAOptimizer(surrogate, population_size=20, nb_iter=50) # The default population size is 100.

nb_runs = 2
worker = Worker(CS, optimizer=optimizer, runs=nb_runs)

worker.search()
worker.summary()

best_config = worker.best_config
best_model = worker.best_arch


