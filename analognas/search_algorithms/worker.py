from analognas.search_spaces.config_space import ConfigSpace
import numpy as np 

'''
Worker is a NAS search laucher. 
max_budget (hrs): maximum search time if n_iter is not exceeded.
n_iter: number of iteration 
'''
class Worker():
    def __init__(self, dataset = "VWW", cs: ConfigSpace=None, eval = None, max_budget=1, n_iter=100, search_algorithm="rs"): 
        self.dataset = dataset 
        self.max_budget = max_budget
        self.n_iter = n_iter
        self.config_space = cs
        self.search_algorithm = search_algorithm # can be random or grid 
        self.best_solution = self.config_space.sample()
        self.evaluation = eval
        
    def rs_search(self, sample=100):
        D = self.cs.sample() # initial decision variables
        best_f = 0.0 # initialize the "best found" - both the function value and the x values
        best_x = [None]*D

        for i in range(self.n_iter):
            # use an "operator" to generate a new candidate solution
            new_x = self.cs.sample()
            new_f = self.evaluation(new_x)
            if new_f > best_f: # see if it's an improvement -- in multiobjective, this is the Pareto sort
                best_f = new_f
                best_x = new_x

        return {'best_x': best_x, 'best_f': best_f}
    
    def ea_search(self, population_size, mutation_prob=0.9, crossover_prob=0.5):
        D = [self.cs.sample()]* # initial decision variables
        best_f = 0.0 # initialize the "best found" - both the function value and the x values
        best_x = [None]*D

        for i in range(self.n_iter):
            # use an "operator" to generate a new candidate solution
            # this is "uniform mutation" in MOEA lin
            new_x = self.cs.mutate(D) 
            new_f = self.evaluation(new_x)
            if new_f > best_f: # see if it's an improvement -- in multiobjective, this is the Pareto sort
                best_f = new_f
                best_x = new_x

        return {'best_x': best_x, 'best_f': best_f}
    
    # surrogate or approximation for the objective function
    def surrogate(self, model, X):
        # catch any warning generated when making a prediction
        return model.predict(X, return_std=True)

    # probability of improvement acquisition function
    def acquisition(self, X, Xsamples, model):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        mu = mu[:, 0]
        # calculate the probability of improvement
        probs = (mu - best) / (std+1E-9)
        return probs

    # optimize the acquisition function
    def bayesian_search(self, X, y, model):
        # random search, generate random samples
        Xsamples = self.rs_search(100)
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        scores = self.acquisition(X, Xsamples, model)
        # locate the index of the largest scores
        ix = np.argmax(scores)
        return Xsamples[ix, 0]