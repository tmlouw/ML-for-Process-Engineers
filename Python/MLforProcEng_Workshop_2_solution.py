#%% Machine Learning for Process Engineers - Workshop (Part 2)
#  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2022
#
#  The functions "GenerateData" and "CreateGaussDesignMatrix"
#  are required to run all the scripts

import types
import numpy as np
import matplotlib.pyplot as plt
from WorkshopFunctions import *
from sklearn import linear_model, preprocessing, model_selection, feature_selection

plt.ion()

# Initialize the mean function and create the "Fit" data table
# (see MLforProcEng_Workshop_1.py for details)
f = lambda t: 6*np.exp(-t**2) * np.sin(t)
sig_eps = 0.2
t = np.linspace(-4,4).reshape(-1, 1)
Fit = types.SimpleNamespace(t = t,
                            f = f(t))

#%% Example 6:  Use feature selection to identify a reduced-order polynomial model
# The code below has been copied from example 2 exactly, then trimmed down
# for brevity, and the maximum model order was set to p = 50
fig, axs = plt.subplots(2)
AllData = GenerateData(f, sig_eps, 100, axs[0], [-4, 4, -4, 4])

# Set the maximum polynomial model order
p_max = 2
X_full = preprocessing.PolynomialFeatures(p_max).fit_transform(AllData.t)
mdl = linear_model.LinearRegression()
k = 5
Error_CV = np.zeros([k, p_max])
mdl_fs = []
mdl_list = []
for p in range(p_max):
    mdl_fs.append(feature_selection.SequentialFeatureSelector(mdl, n_features_to_select = p+1).fit(X_full, AllData.y))
    X = mdl_fs[-1].transform(X_full)
    #mdl_list.append(mdl.fit(X, AllData.y))
    Error_CV[:,p] = -model_selection.cross_validate(mdl, X, AllData.y, cv = k, scoring = 'neg_mean_squared_error')['test_score']

best = np.argmin(np.mean(Error_CV, axis = 0))
X = mdl_fs[best].transform(X_full)
mdl_best = mdl.fit(X, AllData.y)

X_fit = preprocessing.PolynomialFeatures(p_max).fit_transform(Fit.t)
X_fit = mdl_fs[best].transform(X_fit)
Fit.y_pred = mdl_best.predict(X_fit)

axs[0].plot(Fit.t, Fit.y_pred,'r', Fit.t, Fit.f, 'b')

axs[1].boxplot(Error_CV)
axs[1].set_yscale('log')

#%% Example 7:  Use an alternative model using "Gaussian" radial basis functions (G-RBFs)
#  We will now use a different model, where we assume y = sum( theta_j * exp( -(t - c_j)^2 / s )
#  and theta_j are the parameters to be learnt.
#  The centroid positions "c_j" and shape parameters "s" are pre-specified
#  Since "y" is linear in the parameters theta, this is still a linear
#  regression problem.
#  We refer to each basis function as a "Radial Basis Function" (RBF)
#  as the value of the function depends only on the distance to the
#  centroid, |x - c_j|. Further, our RBFs have a Gaussian shape, so we
#  refer to them as G-RBFs.

fig, ax = plt.subplots(1)
Data = GenerateData(f, sig_eps, 100, ax,[-4, 4, -4, 4])

# We use 10 G-RBFs with centroids equally spaced between -3 and 3
# The design matrix is created using the custom function "CreateGaussDesignMatrix".
# We will use the default shape factor throughout
c = np.linspace(-3, 3, 10)
X_train = CreateGaussDesignMatrix(Data.t, c)

# Plot the G-RBFs, evaluated at the data points "t"
ax.plot(Data.t, X_train,'.', Data.t, 0*Data.t+1.2, 'k|')

# Fit the model using linear regression
mdl = linear_model.LinearRegression().fit(X_train, Data.y)

# Evaluate the function at the equally spaced "Fit.t" points
# and plot the function
X_fit = CreateGaussDesignMatrix(Fit.t, c)
Fit.RBF = mdl.predict(X_fit)
ax.plot(Fit.t, Fit.RBF, 'r', Fit.t, Fit.f, 'b', linewidth = 2)

#%% Example 8:  Regularize the G-RBF model using ridge regression
#  The RBFs tend to "fit-to-noise". We can reduce this overfitting by
#  introducing bias to the model using ridge regression.
#  Here, we are minimizing the loss function
#    J = sum( (y_data - y_predicted)^2 ) + alpha*sum( beta^2 )
#  Where alpha > 0 is the regularisation parameter.
#  If alpha = 0, then we have normal sum-of-squares minimization
#  As alpha increases, it penalizes large coefficients more and more

fig, ax = plt.subplots()
Data = GenerateData(f, sig_eps, 100, ax, [-4, 4, -4, 4])
c = np.linspace(-3, 3, 10)
X_train = CreateGaussDesignMatrix(Data.t, c)

# Performs ridge regression using "linear_model.Ridge"
# "alpha" is the value of the regularization parameter
# Run the cell for alpha = 0, alpha = 0.1, alpha = 1 and alpha = 10
mdl = linear_model.Ridge(alpha = 0)
mdl.fit(X_train, Data.y)

X_fit = CreateGaussDesignMatrix(Fit.t, c)
Fit.RBF_ridge = mdl.predict(X_fit)
ax.plot(Fit.t, Fit.RBF_ridge, 'r', Fit.t, Fit.f, 'b', linewidth = 2)

#%% Example 9:  Regularize the G-RBF model using the "lasso" with cross-validation
# We often use cross-validation to estimate the test error using different
# values of the regularization parameter, "alpha", and then choose the
# alpha that corresponds to the "smallest" model. This is typically the
# largest "alpha" value that corresponds to an MSE within one standard
# error of the minimum MSE. 
#
# The "lasso" performs "elastic net regularisation" across a range
# of lambda values provided as input. It will also perform K-fold
# cross-validation to estimate the test error
# If "Alpha" = 1, then "lasso" performs L1 regularisation (lasso), whereas
# if "Alpha" ~ 0, then "lasso" performs L2 regularisation (ridge).
# Type "doc lasso" and "doc Lasso and Elastic Net" for more information
np.random.seed(1)

fig, axs = plt.subplots(2)
Data = GenerateData(f, sig_eps, 100, axs[0],[-4, 4, -4, 4])

c = np.linspace(-3, 3, 10)
X_train = CreateGaussDesignMatrix(Data.t, c)
k = 5 # Number of folds in K-fold cross-validation
l1_ratio = 1e0
alpha_vec = np.logspace(-3,0) # Vector of lambda values to evaluate
Error_CV = np.zeros([k, len(alpha_vec)])
for i in range(len(alpha_vec)):
    mdl = linear_model.ElasticNet(l1_ratio=l1_ratio, alpha=alpha_vec[i])
    Error_CV[:,i] = -model_selection.cross_validate(mdl, X_train, Data.y, cv = k, scoring = 'neg_mean_squared_error')['test_score']


mdl = linear_model.ElasticNetCV(l1_ratio = l1_ratio, alphas = alpha_vec, cv = k)
mdl.fit(X_train, Data.y)

# Show boxplots of the CV error as a function of alpha
axs[1].boxplot(Error_CV, positions = alpha_vec, widths = 0.01)
axs[1].set_yscale('log')

# Evaluate the best fit model
X_fit = CreateGaussDesignMatrix(Fit.t, c)
Fit.RBF_lasso = mdl.predict(X_fit)
axs[0].plot(Fit.t, Fit.RBF_lasso, 'r', Fit.t, Fit.f, 'b', linewidth = 2)
