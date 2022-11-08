from tabnanny import verbose
import xgboost as xgb 
from analognas.evaluators import Evaluator

"""
XGboost Evalutor Wrapper class. 
"""
class XGBoostEvaluator(Evaluator):
    def __init__(
        self,
        model_type = "XGBRanker",
        load_weight = False,
        hpo_wrapper=False,
        hparams_from_file=False
    ):
        self.model_type = model_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {
            'objective': 'rank:ndcg',
            'eta': 0.1,
            'gamma': 1.0,
            'min_child_weight': 0.1,
            'max_depth': 6}
        self.hyperparams = None
        self.hparams_from_file = hparams_from_file
        self.load_weight = load_weight

    def get_model(self, **kwargs):
        evaluator = xgb.XGBRanker(**kwargs)
        if self.load_weight == True: 
            evaluator.load_model("./weights/surrogate_xgboost.json")
        return evaluator

    def fit(self, x_train, y_train, train_info_file="xgboost.txt", hyperparameters=None, epochs=500, verbose=True):
        if hyperparameters == None:
            self.evaluator = self.get_model(self.default_hyperparams)
        else: 
            self.hyperparams = hyperparameters
            self.evaluator = self.get_model(self.hyperparams)

        progress = dict()
        d_train = xgb.DMatrix(x_train, y_train)
        watchlist  = [(d_train,'rank:ndcg')] 
        self.evaluator = self.evaluator.train(self.hyperparams, d_train, epochs, watchlist, evals_result=progress)

        #SAVE MODEL 
        self.evaluator.save_model("xgboost.json")

        return progress['rank:ndcg']

    def query(self, x_test):
        return self.evaluator.predict(x_test)
