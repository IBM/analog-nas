import random
from analogainas.search_spaces.sample import random_sample


class EAOptimizer:
    def __init__(self,
                 max_iter,
                 population_size,
                 mutation_prob):
        self.max_iter = max_iter
        self.population_size = population_size
        self.mutation_prob = mutation_prob

    def mutate(self, architecture):
        if random.random() > self.mutation_prob:
            architecture = random_sample()
        return architecture

    def run(self):
        D = [self.cs.sample()]*self.population_size
        best_f = 0.0
        best_x = [None]*D

        for i in range(self.n_iter):
            # use an "operator" to generate a new candidate solution
            # this is "uniform mutation" in MOEA lin
            new_x = self.mutate(D)
            new_f = self.evaluation(new_x)
            if new_f > best_f:
                best_f = new_f
                best_x = new_x

        return {'best_x': best_x, 'best_f': best_f}
