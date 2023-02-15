import numpy as np 
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

class AccuracyDataLoader:
    def __init__(self, dataset_file="dataset_cifar10.csv", transforms=None):
        self.dataset_file = dataset_file
        self.data = genfromtxt(self.dataset_file, delimiter=',')

        # Applies encoding
        if transforms != None:
            self.data = transforms(self.data)

    def get_train(self): 
        X = self.data[1:24]
        y = self.data[27]
        slope = self.data[26]-self.data[-1]
        
        x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test,y_test), slope