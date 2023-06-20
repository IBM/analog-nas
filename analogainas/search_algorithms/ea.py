"""Classical Evolutionary Algorithm."""
import random
from analogainas.search_spaces.sample import random_sample

class EAOptimizer:
    def __init__(self, max_iter, population_size, mutation_prob):
        self.max_iter = max_iter
        self.population_size = population_size
        self.mutation_prob = mutation_prob

    def mutate(self, architecture):
        if random.random() > self.mutation_prob:
            architecture = random_sample()
        return architecture

    def run(self):
        D = [self.cs.sample() for _ in range(self.population_size)]
        best_f = 0.0
        best_x = [None] * self.population_size

        for _ in range(self.max_iter):
            new_x = [self.mutate(x) for x in D]
            new_f = [self.evaluation(x) for x in new_x]
            
            for j in range(self.population_size):
                if new_f[j] > best_f:
                    best_f = new_f[j]
                    best_x = new_x[j]
            
            D = new_x

        return {'best_x': best_x, 'best_f': best_f}
