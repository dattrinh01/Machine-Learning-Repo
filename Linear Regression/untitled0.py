"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
X = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)

plt.figure(figsize = (15,10))
plt.scatter(X,y)
plt.xlabel('Diện tích', color = 'darkred')
plt.ylabel('Giá', color = 'darkred')
plt.title('Mô hình dự đoán giá nhà dựa trên diện tích', color = 'darkred')

X = np.hstack((np.ones((N,1)), X))

w = np.array([0.,1.]).reshape(-1,1)

record = 10000
cost = np.zeros((record,1))
learning_rate = 0.0000001
for i in range(1, record):
    r = X @ w - y
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, X[:,1].reshape(-1,1)))
    print(cost[i])

predict = np.dot(X, w)
plt.plot((X[0][1], X[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

plt.figure(figsize = (15,10))
plt.scatter(X_train,y_train)
plt.plot(X_train, lin_reg.predict(X_train), color = 'r')
plt.xlabel('Diện tích', color = 'darkred')
plt.ylabel('Giá', color = 'darkred')
plt.title('Mô hình dự đoán giá nhà dựa trên diện tích', color = 'darkred')

