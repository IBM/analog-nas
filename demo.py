from analognas.search_spaces import config_space 
from analognas.evaluators import XGBoostEvaluator
from analognas.search_algorithms import EAOptimizer, Worker

CS = config_space('CIFAR-10')
surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True)
optimizer = EAOptimizer()

worker = Worker(CS, eval=surrogate, optimizer=optimizer, runs=5)

result = worker.search()
print("Best architecture accuracy: ", result["best_acc"])
print("Standard deviation of accuracy over 5 runs: ", result['std_err'])