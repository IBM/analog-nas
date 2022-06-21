from analognas.search_spaces.sample import random_sample 
from analognas.search_spaces.resnet_macro_architecture import Network 

arch = random_sample('VWW', 1)
model = Network(arch)

print(model)
