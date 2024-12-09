"""Hyperparameter Configuration Space."""
import numpy as np
import random
from abc import ABC, abstractmethod


class ConfigSpace(ABC):
    @abstractmethod
    def add_hyperparameter(self, name, type, min_value, max_value, step=1):
        pass

    @abstractmethod
    def add_hyperparameter_range(self, name, type, range):
        pass

    @abstractmethod
    def sample_arch(self):
        pass

    @abstractmethod
    def sample_arch_uniformly(self, n):
        pass

    @abstractmethod
    def set_hyperparameters(self):
        pass

    @abstractmethod
    def remove_hyperparameter(self, name):
        pass

    @abstractmethod
    def compute_cs_size(self):
        pass

    @abstractmethod
    def get_hyperparameters(self):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass