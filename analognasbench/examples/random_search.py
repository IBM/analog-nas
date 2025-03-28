import random
from analognasbench.data import AnalogNASBench, DriftMeasurement

def random_search_analog_nasbench(
    num_samples=100,
    objective_metric='analog_drift_86400',
    seed=42
):
    """
    Perform a random search over the AnalogNASBench to find
    the architecture with the best (maximum) 'analog_drift_86400' value.
    
    :param num_samples: Number of random architectures to evaluate
    :param objective_metric: Metric to optimize (default: 'analog_drift_86400')
    :param seed: Random seed for reproducibility
    :return: (best_arch, best_metric_value)
    """
    random.seed(seed)
    
    benchmark = AnalogNASBench()

    all_archs = benchmark.list_available_architectures()

    best_arch = None
    best_metric_value = -float('inf') 

    for _ in range(num_samples):
        arch = random.choice(all_archs)
        
        result = benchmark.query_metric(arch, objective_metric)
        
        if isinstance(result, str) and result == "architecture not found":
            continue
        
        if isinstance(result, DriftMeasurement):
            metric_value = result.value
        elif isinstance(result, (int, float)):
            metric_value = float(result)
        else:
            # If it's something else or None
            continue
        
        if metric_value > best_metric_value:
            best_arch = arch
            best_metric_value = metric_value
    
    return best_arch, best_metric_value


if __name__ == "__main__":
    best_arch, best_val = random_search_analog_nasbench(
        num_samples=500, 
        objective_metric='analog_drift_86400',
        seed=42
    )
    print("Best architecture found:", best_arch)
    print("Best metric value:", best_val)

    benchmark = AnalogNASBench()
    arch_details = benchmark.get_architecture_details(best_arch)
    print("Architecture details:")
    print(arch_details)
