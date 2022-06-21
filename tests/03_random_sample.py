from analognas.search_spaces.sample import random_sample 

dataset = "VWW"
number_of_architectures = 30 
architectures = random_sample(dataset, number_of_architectures) 
print(architectures[0])