import random
from analogainas.search_spaces.sample import random_sample

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
        T_AVM: constraint applied on the predicted AVM (robustness check).
    """
    def __init__(self,
                surrogate,
                nb_iter = 200,
                population_size=50,
                mutation_prob_width=0.8,
                mutation_prob_depth=0.8, 
                mutation_prob_other=0.6,
                max_nb_param=1,
                T_AVM =10):
        
        assert population_size > 10, f"Population size needs to be at least 10, got {population_size}"

        self.surrogate = surrogate
        self.nb_iter = nb_iter
        self.population_size = int(population_size/10)
        self.mutation_prob_width = mutation_prob_width
        self.mutation_prob_depth = mutation_prob_depth
        self.mutation_prob_other = mutation_prob_other
        self.max_nb_param = max_nb_param
        self.T_AVM = T_AVM
    
    def mutate(self, cs, architecture):
        r = random.random() 
        if r < 0.4:
            architecture= self.mutate_width(cs,architecture)
        elif r < 0.8:
            architecture= self.mutate_depth(cs,architecture)
        else: 
            architecture= self.mutate_other(cs,architecture)
            
        return architecture

    def mutate_width(self, cs, architecture):
        if random.random() < self.mutation_prob_width:
            architecture = cs.sample_arch_uniformly(1)
        return architecture

    def mutate_depth(self, cs, architecture):
        if random.random() < self.mutation_prob_depth:
            architecture = cs.sample_arch_uniformly(1)
        return architecture

    def mutate_other(self, cs, architecture):
        if random.random() < self.mutation_prob_other:
            architecture = cs.sample_arch_uniformly(1)
        return architecture

    def generate_initial_population(self, cs):
        P = [cs.sample_arch_uniformly(1)] * self.population_size
        _, slope = self.surrogate.query_pop(P)

        while (not self.satisfied_constrained(P)):
            for i, s in enumerate(slope):
                if s > self.T_AVM:
                    P[i] = cs.sample_arch_uniformly(1)
        return P

    def satisfied_constrained(self, P):
        _, slope = self.surrogate.query_pop(P)
        for i, s in enumerate(slope):
                if s > self.T_AVM:
                    return False
        return True

    def run(self, cs):
        P = self.generate_initial_population(cs)
        best_f = 0.0
        best_x = [None]*self.population_size

        for i in range(self.nb_iter):
            best_accs =[]
            new_P = []
            for a in P:
                new_a = self.mutate(cs, a)
                new_P.append(new_a)
                acc, _ = self.surrogate.query(new_a)
                best_accs.append(acc)
            new_f = max(best_accs)
            if new_f > best_f:
                best_f = new_f
                best_x = new_a[0]
            
            P = new_P

            print("ITERATION {} completed: best acc {}".format(i, best_f))

        return best_x, best_f
