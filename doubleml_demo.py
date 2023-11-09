import numpy as np
from doubleml.datasets import fetch_bonus
from doubleml import DoubleMLData

# Partially linear model
from doubleml import DoubleMLPLR

# machine learners to estimate the nuisance models
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


# Dataset from the Pennsylvania Reemployment Bonus Experiment 
df_bonus = fetch_bonus('DataFrame')
print(df_bonus.head(5))

# Simulated data
np.random.seed(3141)
n_obs = 500
n_vars = 100
theta = 3
X = np.random.normal(size=(n_obs, n_vars))
d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

# Specify the data and the variables for the causal model
dml_data_bonus = DoubleMLData(df_bonus,
                                y_col='inuidur1',
                                d_cols='tg',
                                x_cols=['female', 'black', 'othrace', 'dep1', 'dep2',
                                        'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54',
                                        'durable', 'lusd', 'husd'])

print(dml_data_bonus)


# array interface to DoubleMLData -- Initializes numpy.ndarray as a DoubleMLData object
dml_data_sim = DoubleMLData.from_arrays(X, y, d)

print(dml_data_sim)


# machine learners to estimate the nuisance models (m_0 and g_0 for PLR model)
# For the bonus data we use a random forest regression model 
# For the simulated data from a sparse partially linear model we use a Lasso regression model. 

# Pennsylvania bonus data
learner = RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', max_depth= 5)
ml_l_bonus = clone(learner)  # Estimator for E[Y|X]
ml_m_bonus = clone(learner)  # Estimator for E[D|X]

# Simulated data
learner = LassoCV()
ml_l_sim = clone(learner)
ml_m_sim = clone(learner)

# Estimating treatment effect on the Bonus data
np.random.seed(3141)
obj_dml_plr_bonus = DoubleMLPLR(dml_data_bonus, ml_l_bonus, ml_m_bonus)
obj_dml_plr_bonus.fit();
print(obj_dml_plr_bonus)

# Estimating theta on the simulated data
obj_dml_plr_sim = DoubleMLPLR(dml_data_sim, ml_l_sim, ml_m_sim)
obj_dml_plr_sim.fit();
print(obj_dml_plr_sim)