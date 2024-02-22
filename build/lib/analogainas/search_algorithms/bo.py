"""Bayesian Optimizer."""
import numpy as np

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

    def maximize(self):
        for _ in range(self.n_iter):
            x_next = self.propose_next_point()
            y_next = self.evaluate_func(x_next)

            self.X.append(x_next)
            self.y.append(y_next)

        best_idx = np.argmax(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]
        return best_x, best_y

    def propose_next_point(self):
        x_candidates = self.random_state.uniform(
            low=self.search_space[:, 0],
            high=self.search_space[:, 1],
            size=(100, self.search_space.shape[0])
        )

        best_x = None
        best_acquisition = float('-inf')

        for x in x_candidates:
            acquisition = self.acquisition(x)
            if acquisition > best_acquisition:
                best_x = x
                best_acquisition = acquisition

        return best_x

    def gaussian_process_regression(self):
        # Define your surrogate model (Gaussian Process) and fit it to the data
        # Example: Mean of 0, Standard Deviation of 1
        mean = 0.0
        std = 1.0
        return mean, std

    # optimize the acquisition function
    def run(self, X, y, model):
        # random search, generate random samples
        Xsamples = self.rs_search(100)
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        scores = self.acquisition(X, Xsamples, model)
        # locate the index of the largest scores
        ix = np.argmax(scores)
        return Xsamples[ix, 0]
