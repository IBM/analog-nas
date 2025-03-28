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
# noisy_drift_60
# analog_drift_60
# noisy_drift_3600
# analog_drift_3600
# noisy_drift_86400
# analog_drift_86400
# noisy_drift_2592000
# analog_drift_2592000
# params

noisy_drift_3600 = benchmark.query_metric((0, 0, 2, 0, 0, 0), 'noisy_drift_3600')

print("noisy_drift_3600 :",noisy_drift_3600)
print("noisy_drift_3600 value :",noisy_drift_3600.value)
print("noisy_drift_3600 uncertainty :",noisy_drift_3600.uncertainty)

analog_accuracy = benchmark.query_metric((0, 0, 2, 0, 0, 0), 'analog_accuracy')

print("analog_accuracy :",analog_accuracy)

# # Get full architecture details
arch_details = benchmark.get_architecture_details((0, 0, 2, 0, 0, 0))
print(arch_details)


