from analogainas.search_spaces import config_space 
from analogainas.evaluators import XGBoostEvaluator
from analogainas.search_algorithms import EAOptimizer, Worker

CS = config_space('CIFAR-10')  # Search Space Definition
surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True) # 
optimizer = EAOptimizer(surrogate, population_size=20, nb_iter=50) # The default population size is 100.

nb_runs = 2
worker = Worker(CS, optimizer=optimizer, runs=nb_runs)

worker.search()
worker.summary()

best_config = worker.best_config
best_model = worker.best_arch

