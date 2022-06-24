from analognas.search_spaces import config_space 
from analognas.evaluators import XGBoostEvaluator
from analognas.search_algorithms import EAOptimizer, Worker

CS = config_space('CIFAR-10')
surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True)
optimizer = EAOptimizer(population_size=20, nb_iter=50) # The default population size is 100.

worker = Worker(CS, eval=surrogate, optimizer=optimizer, runs=2)

result = worker.search()
print("Best architecture predicted accuracy: ", result["best_acc"])
print("Standard deviation of accuracy over 5 runs: ", result['std_err'])
