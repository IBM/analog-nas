import math
from analognasbench.data import AnalogNASBench, DriftMeasurement

def brute_force_best_architecture(metric='analog_drift_86400'):
    """
    Searches exhaustively through all architectures in the AnalogNASBench
    to find the one with the highest value of the given metric.
    
    :param metric: The metric name to optimize (e.g., 'analog_drift_86400')
    :return: (best_arch, best_value)
    """
    benchmark = AnalogNASBench()
    
    all_archs = benchmark.list_available_architectures()
    if not all_archs:
        print("No architectures found in the benchmark.")
        return None, float('-inf')
    
    best_arch = None
    best_value = -math.inf  # or float('-inf')
    
    for arch in all_archs:
        result = benchmark.query_metric(arch, metric)
        if isinstance(result, DriftMeasurement):
            val = result.value
        elif isinstance(result, (int, float)):
            val = float(result)
        else:
            # If the result is a string like "architecture not found" or None
            continue
        
        if val > best_value:
            best_value = val
            best_arch = arch
    
    return best_arch, best_value


if __name__ == "__main__":
    best_arch, best_val = brute_force_best_architecture(metric='analog_drift_86400')
    print("Best architecture found (brute force):", best_arch)
    print("Best metric value:", best_val)
    
    # Optionally retrieve more details
    benchmark = AnalogNASBench()
    details = benchmark.get_architecture_details(best_arch)
    print(details)
