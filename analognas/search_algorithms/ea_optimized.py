import random 
from analognas.search_spaces.sample import random_sample 

class EAOptimizer:
    def __init__(self, max_iter, population_size, mutation_prob_width, mutation_prob_depth, mutation_prob_other):
        self.max_iter = max_iter
        self.population_size = population_size
        self.mutation_prob_width = mutation_prob_width
        self.mutation_prob_depth = mutation_prob_depth
        self.mutation_prob_other = mutation_prob_other

    def mutate_width(self, architecture):
        if random.random() > self.mutation_prob_width:
            architecture = random_sample()
        return architecture

    def mutate_depth(self, architecture):
        if random.random() > self.mutation_prob_depth:
            architecture = random_sample()
        return architecture

    def mutate_other(self, architecture):
        if random.random() > self.mutation_prob_other:
            architecture = random_sample()
        return architecture

    def run(self):
        D = [self.cs.sample()]* # initial decision variables
        best_f = 0.0 # initialize the "best found" - both the function value and the x values
        best_x = [None]*D

        for i in range(self.n_iter):
            # use an "operator" to generate a new candidate solution
            # this is "uniform mutation" in MOEA lin
            new_x = self.mutate(D) 
            new_f = self.evaluation(new_x)
            if new_f > best_f: # see if it's an improvement -- in multiobjective, this is the Pareto sort
                best_f = new_f
                best_x = new_x

        return {'best_x': best_x, 'best_f': best_f}
    