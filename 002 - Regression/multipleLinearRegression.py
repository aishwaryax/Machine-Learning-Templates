# Multiple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3]=labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
X=oneHotEncoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_prediction=regressor.predict(X_test)

#Building the optimized model
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_optimum=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_optimum).fit()
regressor_OLS.summary()
X_optimum=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_optimum).fit()
regressor_OLS.summary()
X_optimum=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_optimum).fit()
regressor_OLS.summary()
X_optimum=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y, exog=X_optimum).fit()
regressor_OLS.summary()
X_optimum=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y, exog=X_optimum).fit()
regressor_OLS.summary()
