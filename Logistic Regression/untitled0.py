import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv').values
N, d = data.shape
X = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.scatter(X[:10,0], X[:10,1], color = 'red', edgecolors='none',s = 30, label = 'Cho vay')
plt.scatter(X[10:, 0], X[10:,1], color = 'blue', edgecolors='none',s = 30, label = 'Từ chối')
plt.legend(loc=1)

X = np.hstack((np.ones((N,1)), X))
w = np.array([0.,0.1,0.1]).reshape(-1,1)

record = 1000
cost = np.zeros((record,1))
learning_rate = 0.01

for i in range(1,record):
    y_pred = sigmoid(X @ w)
    cost[i] = -np.sum(np.multiply(y,np.log(y_pred)) + np.multiply(1 - y, np.log(1 - y_pred)))
    w = w - learning_rate * (X.T @ (y_pred - y))
    print(cost[i])
rate = 0.5
plt.plot((4, 10),(-(w[0]+4 * w[1]+ np.log(1/rate-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/rate-1))/w[2]), 'g')
plt.show()



























