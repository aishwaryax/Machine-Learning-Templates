#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #to convert vector to matrix
y = dataset.iloc[:, 2].values

#Fitting Linear Regression Model to dataset
from sklearn.linear_model import LinearRegression
linearRegressor=LinearRegression()
linearRegressor.fit(X,y)

#Fitting Polynomial Regresion Model to dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
polynomialRegressor=PolynomialFeatures(degree=5)
X_polynomial=polynomialRegressor.fit_transform(X)
linearRegression_poly=LinearRegression()
linearRegression_poly.fit(X_polynomial,y)

#Visualizing Linear regression model
plt.scatter(X, y, color="red")
plt.plot(X, linearRegressor.predict(X), color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualizing Polynomial regression model
X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, linearRegression_poly.predict(polynomialRegressor.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()