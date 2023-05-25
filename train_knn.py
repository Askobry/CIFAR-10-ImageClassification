import time

import matplotlib.pyplot as plt
import numpy as np

from classifiers.KNN import KNN
from data_utils import *

cifar10_dir = 'dataset/cifar-10-batches-py/'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])

num_train = 5000
num_test = 1000

X_train = X_train[:num_train]
y_train = y_train[:num_train]
X_test = X_test[:num_test]
y_test = y_test[:num_test]


k_choice = [1, 3, 5, 10, 15, 20, 40, 100]
k1_accuracy = []
k2_accuracy = []
for k in k_choice:
    k_start_time = time.time()
    knn = KNN(k, 'l1')
    knn.train(X_train, y_train)
    predict = knn.predict(X_test)
    num_right = np.sum((predict==y_test).astype('float'))
    accuracy = (num_right / num_test) * 100
    k1_accuracy.append(accuracy)
    print("K_l1 = {}\t accuracy = {:.2f}%\t Time: {:.2f}".format(k, accuracy, time.time() - k_start_time))

print("-" * 50)
for k in k_choice:
    k_start_time = time.time()
    knn = KNN(k, 'l2')
    knn.train(X_train, y_train)
    predict = knn.predict(X_test)
    num_right = np.sum((predict==y_test).astype('float'))
    accuracy = (num_right / num_test) * 100
    k2_accuracy.append(accuracy)
    print("K_l2 = {}\t accuracy = {:.2f}%\t Time: {:.2f}".format(k, accuracy, time.time() - k_start_time))

plt.plot(k_choice,k1_accuracy,'r-', label='L1')
plt.plot(k_choice,k1_accuracy,'go')
plt.plot(k_choice,k2_accuracy,'b-', label='L2')
plt.plot(k_choice,k2_accuracy,'c^')
plt.legend()
plt.xlabel('K')
plt.ylabel('Accuracy(%)')
plt.title('K choice for accuracy')
plt.show()
