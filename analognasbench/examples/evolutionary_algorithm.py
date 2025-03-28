import random
from analognasbench.data import AnalogNASBench, DriftMeasurement

def evaluate_fitness(benchmark, arch, metric='analog_drift_86400'):
    """
    Returns the drift value of the specified architecture for the given metric.
    If it's not found, returns -inf to effectively invalidate the architecture.
    """
    result = benchmark.query_metric(arch, metric)
    if isinstance(result, str) and result == "architecture not found":
        return float('-inf')  # or you could return 0
    if isinstance(result, DriftMeasurement):
        return result.value
    elif isinstance(result, (int, float)):
        return float(result)
    return float('-inf')  # fallback for unexpected cases

def tournament_selection(population, fitness_dict, k=3):
    """
    Perform tournament selection on the population.
    :param population: List of architectures.
    :param fitness_dict: Dictionary mapping arch -> fitness.
    :param k: Tournament size.
    :return: Selected 'winner' architecture.
    """
    # Randomly sample k individuals
    contenders = random.sample(population, k)
    # Return the individual with the highest fitness
    best = max(contenders, key=lambda arch: fitness_dict[arch])
    return best

def single_point_crossover(parent1, parent2):
    """
    Single-point crossover between two architecture tuples.
    :param parent1: tuple (e.g., (0,1,0,4,3,2))
    :param parent2: tuple
    :return: child1, child2
    """
    if len(parent1) != len(parent2):
        # For demonstration, assume they're always the same length
        return parent1, parent2
    
    length = len(parent1)
    if length < 2:
        # No meaningful crossover if there's only 1 dimension
        return parent1, parent2
    
    # pick a random crossover point
    point = random.randint(1, length - 1)
    # combine slices
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate_architecture(arch, valid_range=(0, 5), mutation_rate=0.2):
    """
    Randomly mutate one position of the tuple if a random draw < mutation_rate.
    :param arch: tuple
    :param valid_range: A guessed valid range for each position (0..5).
                       Adjust if your real architecture search space is different.
    :param mutation_rate: Probability that we apply the mutation.
    :return: mutated architecture
    """
    if random.random() < mutation_rate:
        arch_list = list(arch)
        # pick one position to mutate
        idx = random.randint(0, len(arch_list) - 1)
        # pick a new random value in the valid range
        arch_list[idx] = random.randint(valid_range[0], valid_range[1])
        return tuple(arch_list)
    else:
        return arch

def evolutionary_search_analog_nasbench(
    pop_size=20,
    generations=10,
    metric='analog_drift_86400',
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_k=3,
    random_seed=42
):
    """
    A simple evolutionary algorithm to maximize the specified metric 
    (defaults to 'analog_drift_86400').
    
    :param pop_size: Size of the population
    :param generations: Number of generations to run
    :param metric: Metric to optimize (maximize)
    :param crossover_rate: Probability of performing crossover on each mating
    :param mutation_rate: Mutation probability for each child
    :param tournament_k: Tournament size
    :param random_seed: Seed for reproducibility
    :return: (best_arch, best_fitness)
    """
    random.seed(random_seed)
    benchmark = AnalogNASBench()
    
    all_archs = benchmark.list_available_architectures()
    if len(all_archs) == 0:
        print("No architectures found in the benchmark.")
        return None, float('-inf')

    population = [random.choice(all_archs) for _ in range(pop_size)]
    
    fitness_dict = {arch: evaluate_fitness(benchmark, arch, metric) for arch in population}
    
    best_arch = max(population, key=lambda a: fitness_dict[a])
    best_fitness = fitness_dict[best_arch]

    for gen in range(generations):
        new_population = []
        
        while len(new_population) < pop_size:
            
            parent1 = tournament_selection(population, fitness_dict, k=tournament_k)
            parent2 = tournament_selection(population, fitness_dict, k=tournament_k)
            
            if random.random() < crossover_rate:
                child1, child2 = single_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = mutate_architecture(child1, mutation_rate=mutation_rate)
            child2 = mutate_architecture(child2, mutation_rate=mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        for arch in new_population:
            if arch not in fitness_dict:
                fitness_dict[arch] = evaluate_fitness(benchmark, arch, metric)
        
        population = new_population
        
        current_best_arch = max(population, key=lambda a: fitness_dict[a])
        current_best_fitness = fitness_dict[current_best_arch]
        if current_best_fitness > best_fitness:
            best_arch, best_fitness = current_best_arch, current_best_fitness
        
        print(f"Generation {gen+1}/{generations}, Best so far: {best_arch}, Fitness={best_fitness:.4f}")
    
    return best_arch, best_fitness


if __name__ == "__main__":
    best_architecture, best_val = evolutionary_search_analog_nasbench(
        pop_size=300,
        generations=200,
        metric='analog_drift_86400',
        crossover_rate=0.8,
        mutation_rate=0.5,
        tournament_k=8,
        random_seed=42
    )
    print("\nFinal Best Architecture Found:", best_architecture)
    print("Final Best Metric Value:", best_val)

    benchmark = AnalogNASBench()
    details = benchmark.get_architecture_details(best_architecture)
    print(details)
