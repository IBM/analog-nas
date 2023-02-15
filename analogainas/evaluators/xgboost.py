from tabnanny import verbose
import xgboost as xgb 
#from base_evaluator import Evaluator

"""
XGboost Evalutor Wrapper class. 
"""
class XGBoostEvaluator():
    def __init__(
        self,
        model_type = "XGBRanker",
        load_weight = True,
        hpo_wrapper=False,
        hparams_from_file=False
    ):
        self.model_type = model_type
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {
            'tree_method':'gpu_hist',
            'booster':'gbtree',
            'objective':'rank:pairwise',
            'random_state':42, 
            'learning_rate':0.1,
            'colsample_bytree':0.9, 
            'eta':0.05, 
            'max_depth':6, 
            'n_estimators':110, 
            'subsample':0.75,
            'enable_categorical':True}
        self.hyperparams = None
        self.hparams_from_file = hparams_from_file
        self.load_weight = load_weight
        self.ranker = self.get_ranker()
        self.avm_predictor = self.get_avm_predictor()
        self.std_predictor = self.get_std_predictor()

    def get_ranker(self):
        ranker = xgb.XGBRanker(**self.default_hyperparams)
        if self.load_weight == True: 
            ranker.load_model(r"C:\Users\hadjer\analog-nas\analogainas\evaluators\weights\surrogate_xgboost_ranker.json")
        return ranker

    def get_avm_predictor(self):
        avm_predictor = xgb.XGBRegressor()
        if self.load_weight == True: 
            avm_predictor.load_model(r"C:\Users\hadjer\analog-nas\analogainas\evaluators\weights\surrogate_xgboost_avm.json")
        return avm_predictor

    def get_std_predictor(self):
        std_predictor = xgb.XGBRegressor()
        if self.load_weight == True: 
            std_predictor.load_model(r"C:\Users\hadjer\analog-nas\analogainas\evaluators\weights\surrogate_xgboost_std.json")
        return std_predictor

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
        self.evaluator.save_model(train_info_file)

        return progress['rank:ndcg']

    def query_pop(self, P):
        x_test = []
        for a in P:
            arch = list(a[0].values())
            x_test.append(arch)
        return self.ranker.predict(x_test), self.avm_predictor.predict(x_test)

    def query(self, P):
        x_test = []
        arch = list(P[0].values())
        x_test.append(arch)
        return self.ranker.predict(x_test), self.avm_predictor.predict(x_test)
