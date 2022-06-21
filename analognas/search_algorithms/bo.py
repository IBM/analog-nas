"""Bayesian Optimizer."""

class BOptimizer:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        
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