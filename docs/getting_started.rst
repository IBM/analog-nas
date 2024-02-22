Tutorial
========

*AnalogAINAS* is a framework that aims at building analog-aware efficient deep learning models. AnalogNAS is built on top of the [AIHWKIT](https://github.com/IBM/aihwkit). IBM Analog Hardware Acceleration Kit (AIHWKIT) is an open source Python toolkit for exploring and using the capabilities of in-memory computing devices in the context of artificial intelligence.

In a high-level AnalogAINAS consists of 4 main building blocks which (can) interact with each other:

* Configuration spaces: a search space of architectures targeting a specific dataset.
* Evaluator: a ML predictor model to predict: 
    * 1-day Accuracy: the evaluator models the drift effect that is encountered in Analog devices. The accuracy after 1 day of drift is then predicted and used as an objective to maximize. 
    * The Accuracy Variation for One Month (AVM): The difference between the accuracy after 1 month and the accuracy after 1 sec. 
    * The 1-day accuracy standard deviation: The stochasticity of the noise induces different variation of the model's accuracy depending on its architecture. 
* Optimizer: a optimization strategy such as evolutionary algorithm or bayesian optimization. 
* Worker: A global object that runs the architecture search loop and the final network training pipeline

Create a Configuration Space
----------------------------

AnalogNAS presents a general search space composed of ResNet-like architectures. 

The macro-architecture defined in the file ```search_spaces/resnet_macro_architecture.py``` is customizable to any image classification dataset, given an input shape and output classes. 

.. warning::
    The hyperparameters in the configuration space should have a unique name ID each. 

Evaluator 
---------

To speed up the search, we built a machine learning predictor to evaluate the accuracy and robustness of any given architecture from the configuration space. 

Search Optimizer and Worker
---------------------------

In this example, we will use evolutionary search to look for the best architecture in CS using our evaluator. 

::

    from analogainas.search_algorithms.ea_optimized import EAOptimizer
    from analogainas.search_algorithms.worker import Worker

    optimizer = EAOptimizer(evaluator, population_size=20, nb_iter=10)  

    NB_RUN = 2
    worker = Worker(CS, optimizer=optimizer, runs=NB_RUN)

    worker.search()

    worker.result_summary()


