import random 
from analognas.search_spaces.sample import random_sample 

class EAOptimizer:
    """
    Evolutionary Algorithm with optimized mutations and robustness constraint.
    
    The NAS problem is cast to: 
            Max Acc(arch)
            s.t nb_param(arch) < max_nb_param
                drop(arch) < 10

    Args:
        nb_iter: maximum number of iterations. 
        population_size: number of architectures in the population.

        mutation_prob_width: Mutation probability of modifying the width.
                            - Increase/Decrease widening factor of one block.
                            - Add/Remove branches.
                            -Increase/Decrease Initial output channel size.

        mutation_prob_depth: Mutation probability for modifying the depth.
                            - Increase/Decrease the number of residual blocks. 
                            - Modify the type of convolution from BasicBlock to BottleneckBlock.

        mutation_prob_other: Mutation probability for applying various other transformations:
                            - Add/Remove a residual connection.
                            - Modify initial kernel size.

        max_nb_param: constraint applied to the number of parameters. 
        max_drop: constraint applied on the predicted slope (robustness check).

    """
    def __init__(self, 
                surrogate, 
                nb_iter = 200, 
                population_size=100, 
                mutation_prob_width=0.8, 
                mutation_prob_depth=0.8, 
                mutation_prob_other=0.6, 
                max_nb_param=1, 
                max_drop =10):
        
        assert(population_size < 10, 
            "Population size needs to be at least 10.")
        
        self.surrogate = surrogate
        self.nb_iter = nb_iter
        self.population_size = population_size
        self.mutation_prob_width = mutation_prob_width
        self.mutation_prob_depth = mutation_prob_depth
        self.mutation_prob_other = mutation_prob_other
        self.max_nb_param = max_nb_param
        self.max_drop = max_drop
    
    def mutate_width(self, architecture):
        if random.random() < self.mutation_prob_width:
            architecture = random_sample()
        return architecture

    def mutate_depth(self, architecture):
        if random.random() < self.mutation_prob_depth:
            architecture = random_sample()
        return architecture

    def mutate_other(self, architecture):
        if random.random() < self.mutation_prob_other:
            architecture = random_sample()
        return architecture

    def generate_initial_population(self): 
        P = [self.cs.sample(self.max_nb_param)]* self.population_size
        _, slope = self.surrogate.query(P)

        while (not self.satisfied_constrained(P)):
            for i, s in enumerate(slope):
                if s > self.max_drop:
                    P[i] = self.cs.sample(self.max_nb_param)
        return P

    def satisfied_constrained(self, P):
        _, slope = self.surrogate.query(P)
        for i, s in enumerate(slope):
                if s > self.max_drop:
                    return False
        return True

    def run(self, cs):
        P = self.generate_initial_population(cs)
        best_f = 0.0 # initialize the "best found" - both the function value and the x values
        best_x = [None]*P

        for i in range(self.nb_iter):
            # use an "operator" to generate a new candidate solution
            # this is "uniform mutation" in MOEA lin
            new_x = self.mutate(P) 
            new_f = self.surrogate.query(new_x)
            if new_f > best_f: # see if it's an improvement -- in multiobjective, this is the Pareto sort
                best_f = new_f
                best_x = new_x

        return {'best_x': best_x, 'best_f': best_f}
    