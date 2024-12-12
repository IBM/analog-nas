from analogainas.search_spaces.autoencoder.autoencoder_config_space import AutoEncoderConfigSpace
from analogainas.evaluators.xgboost import XGBoostEvaluator
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker

CS = AutoEncoderConfigSpace()

print(CS.hyperparameters)

surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True)  #
optimizer = EAOptimizer(surrogate, population_size=100, nb_iter=50) # The default population size is 100.

nb_runs = 2
worker = Worker(CS, optimizer=optimizer, runs=nb_runs)

worker.search()
worker.result_summary()

best_config = worker.best_config
best_model = worker.best_arch
