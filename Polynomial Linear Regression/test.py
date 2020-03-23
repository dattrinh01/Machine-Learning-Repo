import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print("Polynomial Regression's score:", lin_reg.score(X_poly, y))

from sklearn.linear_model import LinearRegression 
lin_reg1 = LinearRegression()
lin_reg1.fit(X, y)
print("Simple Linear Regression's score:", lin_reg1.score(X, y))

plt.scatter(X, y, color = 'blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'red')
plt.show()



#from sklearn.linear_model import LinearRegression 
#lr = LinearRegression()
#lr.fit(X , y)

#plt.scatter(X, y, color = 'blue')
#plt.plot(X, lr.predict(X), color = 'red')
#plt.show()