import os
import pandas as pd
import numpy as np
import pickle
import pkg_resources
from ast import literal_eval


class Architecture:
    def __init__(self, architecture, baseline_accuracy,
                 ptq_accuracy, qat_accuracy, 
                 noisy_accuracy, analog_accuracy, baseline_drift_60,
                 baseline_drift_3600,baseline_drift_86400,baseline_drift_2592000,
                 analog_drift_60,analog_drift_3600,analog_drift_86400,analog_drift_2592000
                 ):
        
        self.architecture = architecture  # Tuple (e.g., (0,0,0,4,3,2))
        self.baseline_accuracy = baseline_accuracy

        self.ptq_accuracy = ptq_accuracy
        self.qat_accuracy = qat_accuracy

        self.noisy_accuracy = noisy_accuracy
        self.analog_accuracy = analog_accuracy

        self.baseline_drift_60 = self._parse_drift(baseline_drift_60)
        self.baseline_drift_3600 = self._parse_drift(baseline_drift_3600)
        self.baseline_drift_86400 = self._parse_drift(baseline_drift_86400)
        self.baseline_drift_2592000 = self._parse_drift(baseline_drift_2592000)
        self.analog_drift_60 = self._parse_drift(analog_drift_60)
        self.analog_drift_3600 = self._parse_drift(analog_drift_3600)
        self.analog_drift_86400 = self._parse_drift(analog_drift_86400)
        self.analog_drift_2592000 = self._parse_drift(analog_drift_2592000)

    def _parse_drift(self, drift_str):
        """
        Parse drift string into DriftMeasurement object.
        
        :param drift_str: String in format "value ± uncertainty"
        :return: DriftMeasurement object
        """
        if isinstance(drift_str, (int, float)):
            return DriftMeasurement(drift_str, 0)
        
        if isinstance(drift_str, str):
            drift_str = drift_str.strip()
            
            try:
                return DriftMeasurement(float(drift_str), 0)
            except ValueError:
                pass
            
            if '±' in drift_str:
                try:
                    parts = drift_str.split('±')
                    value = float(parts[0].strip())
                    uncertainty = float(parts[1].strip())
                    return DriftMeasurement(value, uncertainty)
                except:
                    print(f"Warning: Could not parse drift string: {drift_str}")
                    return DriftMeasurement(0, 0)
        
        return DriftMeasurement(0, 0)



class DriftMeasurement:
    def __init__(self, value, uncertainty):
        """
        Represents a drift measurement with value and uncertainty.
        
        :param value: Mean drift value
        :param uncertainty: Uncertainty of the measurement
        """
        self.value = float(value)
        self.uncertainty = float(uncertainty)
    
    def __repr__(self):
        """String representation of the drift measurement"""
        return f"{self.value} ± {self.uncertainty}"
    
    def __str__(self):
        """Human-readable string representation"""
        return self.__repr__()




class AnalogNASBench:
    def __init__(self):
        """
        Initialize the AnalogNAS-Bench dataset.
        """
        data_path = pkg_resources.resource_filename('analognasbench', 'data.anb')
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # List of all metrics for querying
        self.metrics = [
            'architecture', 'baseline_accuracy', 'ptq_accuracy', 
            'qat_accuracy', 'noisy_accuracy', 'analog_accuracy',
            'baseline_drift_60', 'baseline_drift_3600', 
            'baseline_drift_86400', 'baseline_drift_2592000', 
            'analog_drift_60', 'analog_drift_3600', 
            'analog_drift_86400', 'analog_drift_2592000'
        ]
        
    def query_metric(self, architecture, metric):
        """
        Query a specific metric for a given architecture ID.
        
        :param architecture: Architecture identifier
        :param metric: Metric to retrieve
        :return: Metric value or None if not found
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found. Available metrics: {self.metrics}")
        
        arch = self.data.get(architecture)
        if not arch:
            return "architecture not found"
        
        return getattr(arch, metric)
    
    def get_architecture_details(self, architecture):
        """
        Retrieve full details for a specific architecture.
        
        :param architecture: Architecture identifier
        :return: Formatted string of architecture details
        """
        arch = self.data.get(architecture)
        if not arch:
            return "Architecture not found."
        
        # Format the details into a readable string
        details = f"""
Architecture Details:
--------------------
Architecture: \t\t{arch.architecture}
Baseline Accuracy: \t{arch.baseline_accuracy}
PTQ Accuracy: \t\t{arch.ptq_accuracy}
QAT Accuracy: \t\t{arch.qat_accuracy}
Noisy Accuracy: \t{arch.noisy_accuracy}
Analog Accuracy: \t{arch.analog_accuracy}

Baseline Drift:
- 60s: \t\t{arch.baseline_drift_60}
- 3600s: \t{arch.baseline_drift_3600}
- 86400s: \t{arch.baseline_drift_86400}
- 2592000s: \t{arch.baseline_drift_2592000}

Analog Drift:
- 60s: \t\t{arch.analog_drift_60}
- 3600s: \t{arch.analog_drift_3600}
- 86400s: \t{arch.analog_drift_86400}
- 2592000s: \t{arch.analog_drift_2592000}
        """
        
        return details
    
    def list_available_architectures(self):
        """
        List all available architectures.
        
        :return: List of architectures (tuples)
        """
        return list(self.data.keys())