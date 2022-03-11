import numpy as np 

class Hyperparameter:
    def __init__(self, name, type, min_value, max_value): 
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.type = type # Discrete, continuous, categorical 
        self.sampling = "uniform"  # uniform, gaussian
        
    def sample_hyp(self):
        if self.type == "discrete": 
            return np.random.randint(self.min_value, high=self.max_value) 
        if self.type == "continuous": 
            return np.random.uniform(self.min_value, high=self.max_value) 
        
        
class ConfigSpace:
    def __init__(self, dataset):
        self.dataset = dataset # VWW, KWS
        self.search_space = "resnet-like" # for now only resnet-like 
        self.hyperparameters = [] # list of Hyperparameters to search for
        self.set_hyperparameters()
        
    def add_hyperparameter(self, name, type, min_value, max_value):
        hyp = Hyperparameter(name, type, min_value, max_value) 
        self.hyperparameters.append(hyp)
        
    def sample_arch(self): 
        arch = {}
        for hyp in self.hyperparameters: 
            arch[hyp.name] = hyp.sample_hyp()
            
        return arch
    
    def set_hyperparameters(self):
        if self.search_space == "resnet-like": 
            self.add_hyperparameter("out_channel0", "discrete", 8, 64)
            self.add_hyperparameter("M", "discrete", 1, 5)
            self.add_hyperparameter("R1", "discrete", 1, 16)
            self.add_hyperparameter("R2", "discrete", 0, 16)
            self.add_hyperparameter("R3", "discrete", 0, 16)
            self.add_hyperparameter("R4", "discrete", 0, 16)
            self.add_hyperparameter("R5", "discrete", 0, 16)
            
            for i in range(5):
                self.add_hyperparameter("convblock%d" % i, "discrete", 1, 2)
                self.add_hyperparameter("widenfact%d" % i, "continuous", 0.5, 0.8)
                self.add_hyperparameter("B%d" % i, "discrete", 1, 5)
            
def main():
    CS = ConfigSpace("VWW") 
    print(CS.sample_arch()) 
    
if __name__ == "__main__":
    main()
        