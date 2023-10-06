#%% Machine Learning for Process Engineers - Workshop (Part 3)
#  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
# 
# We move to a new example from system identification:  
# We have linear state space system with two inputs u1 and u2, 
# and one output y. 
# We assume a general linear form:
#  y(k+1) =   a_0*y(k)  + a_1*y(k-1)   + a_2*y(k-2)  ... + a_L*y(k-L)
#           + b_0*u1(k) + b_1*u1(k-1)  + b_2*u1(k-2) ... + b_L*u1(k-L)
#           + c_0*u2(k) + c_1*u2(k-1)  + c_2*u2(k-2) ... + c_L*u2(k-L)
#
# The output therefor depends on LAGGED data at sample times (k-l) 
# for l = 0, 1, 2, 3... L
#
# Time series data is contained in "ProcessData.csv", which is required to
# run this script.

import types
import numpy as np
import matplotlib.pyplot as plt
from WorkshopFunctions import *
from sklearn import linear_model, preprocessing, model_selection, feature_selection
from sklearn import cross_decomposition, decomposition
plt.ion()

L = 10
Data = ReadProcessData('ProcessData.csv')
X, y = CreateLaggedDesignMatrix(Data, L, 0.1)

#%% Example 10: Linear model fit to timeseries data
# Fit a linear model without regularization
mdl = linear_model.LinearRegression(fit_intercept = False)
mdl.fit(X, y)
lin_mdl = types.SimpleNamespace(Q = np.eye(X.shape[1]),
                                coef = mdl.coef_[:, np.newaxis])
#y_linear = PredictTimeSeries(lin_mdl, Data, L)

mdl = linear_model.Ridge(fit_intercept = False, alpha = 0.1)
mdl.fit(X, y)
rdg_mdl = types.SimpleNamespace(Q = np.eye(X.shape[1]),
                                coef = mdl.coef_[:, np.newaxis])
y_ridge = PredictTimeSeries(rdg_mdl, Data, L)


fig, ax = plt.subplots()
ax.set_ylim(bottom = -1, top = 1)

plt_train, = ax.plot(Data.t, Data.y, 'k.',markersize = 2)
#plt_lin, = ax.plot(Data.t, y_linear)
plt_ridge, = ax.plot(Data.t, y_ridge)

ax.legend([plt_train, plt_ridge], ['Training data','Ridge regression'])


#%% Example 11: Use PCA regression to predict time series
#  Fit a model using PCA regression
#  Obtain the PCA loadings and the fraction variance explained
pca = decomposition.PCA(n_components = 20)
pca.fit(X)
pca_mdl = types.SimpleNamespace(Q = pca.components_.transpose()[:, 0:4])
T = np.matmul(X, pca_mdl.Q)

mdl = linear_model.LinearRegression(fit_intercept = False)
mdl.fit(T, y)
pca_mdl.coef = mdl.coef_[:, np.newaxis]

y_pca = PredictTimeSeries(pca_mdl, Data, L)
plt_pca, = ax.plot(Data.t, y_pca)

ax.legend([plt_train, plt_ridge, plt_pca], ['Training data','Ridge regression','PCA regression'])


#%% Example 11: Use PCA regression to predict time series
#  Fit a model using PCA regression
#  Obtain the PCA loadings and the fraction variance explained
pls = cross_decomposition.PLSRegression(n_components = 20)
pls.fit(X, y)

pls_mdl = types.SimpleNamespace(Q = pls.x_rotations_[:,0:4])
T = np.matmul(X, pls_mdl.Q)
mdl = linear_model.LinearRegression(fit_intercept = False)
mdl.fit(T, y)
pls_mdl.coef = mdl.coef_[:, np.newaxis]

y_pls = PredictTimeSeries(pls_mdl, Data, L)
plt_pls, = ax.plot(Data.t, y_pls)

ax.legend([plt_train, plt_ridge, plt_pca, plt_pls],
          ['Training data','Ridge regression','PCA regression','PLS regression'])
