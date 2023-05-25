import numpy as np


class KNN(object):
    def __init__(self, k = 3, distance = "l2"):
        self.k = k
        self.distance = distance

    def train(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def predict(self, test_data):
        num_test = test_data.shape[0]
        predict_results = np.zeros(num_test, dtype=self.train_label.dtype)
        for i in range(num_test):
            if self.distance == 'l2':
                dist = np.sqrt(np.sum(np.square(test_data[i] - self.train_data), axis = 1))
            elif self.distance == 'l1':
                dist = np.sum(np.abs(test_data[i] - self.train_data), axis = 1)
            close_k = self.train_label[np.argsort(dist)[:self.k]]
            predict_results[i] = np.argmax(np.bincount(close_k))

        return predict_results
    