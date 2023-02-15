# analogai-nas
**AnalogAINas** is a modular and flexible framework to facilitate implementation of Analog-aware Neural Architecture Search. It offers high-level classes to define: the search space, the accuracy evaluator, and the search strategy. It leverages [the aihwkit framework](https://github.com/IBM/aihwkit) to apply hardware-aware training with analog non-idealities and noise included. **AnalogAINAS** obtained architectures are more robust during inference on Analog Hardware. We also include two evaluators trained to rank the architectures according to their analog training accuracy. 

## Setup 
While installing the repository, creating a new conda environment is recomended.

```
git clone https://github.com/IBM/analog-nas/
pip install -r requirements.txt 
pip setup.py install 
```

## Usage
To get started, check out ```nas_search_demo.py``` to make sure that the installation went well. 

This python script describes how to use the package. 

