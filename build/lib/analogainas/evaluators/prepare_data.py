import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AccuracyDataLoader:
    def __init__(self, dataset_file="dataset_cifar10.csv", transforms=None):
        self.dataset_file = dataset_file
        self.data = genfromtxt(self.dataset_file, delimiter=',')

        # Applies encoding
        if transforms is not None:
            self.data = transforms(self.data)

    def get_train(self):
        X = self.data[1:24]
        y = self.data[27]
        slope = self.data[26] - self.data[-1]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return (X_train, y_train), (X_test, y_test), slope
