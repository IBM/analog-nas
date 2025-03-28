from analognasbench.data import AnalogNASBench

# Initialize
benchmark = AnalogNASBench()

# Query a specific metric for an architecture
# Availible metrics : 
# baseline_accuracy
# ptq_accuracy
# qat_accuracy
# noisy_accuracy
# analog_accuracy
# baseline_drift_60
# analog_drift_60
# baseline_drift_3600
# analog_drift_3600
# baseline_drift_86400
# analog_drift_86400
# baseline_drift_2592000
# analog_drift_2592000

baseline_drift_3600 = benchmark.query_metric((0, 0, 0, 4, 3, 2), 'baseline_drift_3600')

print("baseline_drift_3600 :",baseline_drift_3600)
print("baseline_drift_3600 value :",baseline_drift_3600.value)
print("baseline_drift_3600 uncertainty :",baseline_drift_3600.uncertainty)

analog_accuracy = benchmark.query_metric((0, 0, 0, 4, 3, 2), 'analog_accuracy')

print("analog_accuracy :",analog_accuracy)

# # Get full architecture details
arch_details = benchmark.get_architecture_details((0, 1, 0, 4, 3, 2))
print(arch_details)
