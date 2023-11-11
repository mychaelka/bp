# Simulation study
import numpy as np
import doubleml as dml
from doubleml.datasets import make_pliv_CHS2015
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from doubleml import DoubleMLData


np.random.seed(3141)

n = 2000  # sample size
p = 100  # number of covariates
s = 2  # number of covariates that are confounders
sigma = np.array([[1, 0.5], [0.5, 1]])
e = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=n).T
X = np.random.randn(n, p)  # covariate matrix
beta = np.hstack((np.repeat(0.25, s), np.repeat(0, p - s)))  # Coefficients determining the degree of confounding
d = np.where(np.dot(X, beta) + np.random.randn(n) > 0, 1, 0)  # Treatment equation
z = np.random.randn(n)
s = np.where(np.dot(X, beta) + 0.25 * d + z + e[0] > 0, 1, 0)  # Selection equation
y = np.dot(X, beta) + 0.5 * d + e[1]  # Outcome equation
y[s == 0] = 0  # Setting values to 0 based on the selection equation

#  The true ATE is equal to 0.5


## Creating the DoubleMLData object
simul_data = DoubleMLData.from_arrays(X, y, d)
print(simul_data)