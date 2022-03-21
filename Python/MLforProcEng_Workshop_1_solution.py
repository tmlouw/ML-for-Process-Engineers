#%% Machine Learning for Process Engineers - Workshop (Part 1)
#  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2022
#
#  We will assume that we are measuring samples from
#  a process described by the equation:  y = f(t) + eps, 
#  where "f(t)" is the function described below and "eps" is the 
#  zero mean Gaussian noise with standard deviation sig_eps, 
#  such that eps ~ N(0, sig_eps)
#
#  Samples of our independent variable "t" also follow a standard normal
#  distribution
#
#  The function "GenerateData" is required to run all the scripts

import types
import numpy as np
import matplotlib.pyplot as plt
from WorkshopFunctions import *
from sklearn import linear_model, preprocessing, model_selection

plt.ion()

#%% Create an example of the data to be generated
N = 100
f = lambda t: 6*np.exp(-t**2) * np.sin(t)
sig_eps = 0.2
# Generate 100 observations and plot data
fig, ax = plt.subplots()
Data = GenerateData(f, sig_eps, N, ax, [-4, 4, -4, 4])

t = np.linspace(-4,4).reshape(-1, 1)
Fit = types.SimpleNamespace(t = t,
                            f = f(t))

ax.plot(Fit.t, Fit.f, 'b', linewidth = 2)
ax.plot(Fit.t, 0*Fit.t - 3.75, 'b|', Data.t, 0*Data.t - 3.25, 'k|')

#%% Example 1: fit a first order polynomial model
#  In this first example, we will simply fit a straight line through our
#  data. We notice that, for each independently generated data SET, the fit
#  is slightly different

# Generate 100 observations and plot data
fig, ax = plt.subplots()
Data = GenerateData(f, sig_eps, N, ax, [-4, 4, -4, 4])

# Fit a first order polynomial model using scikit-learn's LinearRegression method
# scikit-learn doesn't provide a complete regression summary as MATLAB does.
# See https://devdocs.io/statsmodels/ for an alternative linear regression library
mdl = linear_model.LinearRegression()
mdl.fit(Data.t, Data.y)

# Evaluated the fitted model at equally spaced points "Fit.t"
# using the "predict" function.
Fit.linear = mdl.predict(Fit.t)

# Plot the fitted linear model as well as the true mean function "Fit.f"
# The function "GenerateData" above already plotted the datapoints
# defined the axis, so any additional plots will be added to the current axes.
ax.plot(Fit.t, Fit.linear, 'r', Fit.t, Fit.f, 'b')
ax.legend(['Training data', 'Linear model', 'True function'])

# EXERCISE:
# Before moving on, you can run this cell multiple times. Notice that each
# time you run the cell, the data-points are slightly different, and so too
# the model fit.

#%% Example 2:  Fit a p-order polynomial model
#  We repeat the exercise above, but this time we fit a p-order polynomial to the data
# Generate a new set of 100 observations
fig, ax = plt.subplots()
Data = GenerateData(f, sig_eps, 100, ax, [-4, 4, -4, 4])

# Fit a p-th order polynomial model
p = 4

# Create a "design matrix" X, where each column of "X" corresponds to a "feature"
# and each row of "X" corresponds to an observation.
# For the polynomial regression, the j'th column
# of the design matrix corresponds to t^j.
# We can use "x2fx(Data.t, (1:p)')" for this.
# Type "doc x2fx" for details
poly = preprocessing.PolynomialFeatures(p)
X = poly.fit_transform(Data.t)
# Alternative: X = preprocessing.PolynomialFeatures(p).fit_transform(Data.t)

# Let's see what each column of "X" looks like
ax.plot(Data.t, X, '.', markersize = 2)
ax.plot(Data.t, 0*Data.t-3.5, 'k|')

# Fit the linear model using the design matrix "X"
mdl = linear_model.LinearRegression()
mdl.fit(X, Data.y)

# Create a design matrix "X" using the equally spaced vector "Fit.t",
# then evaluate the model at the equally spaced points using "predict"
# Plot your results
X = preprocessing.PolynomialFeatures(p).fit_transform(Fit.t)
Fit.poly = mdl.predict(X)
ax.plot(Fit.t, Fit.poly,'r', Fit.t, Fit.f, 'b')

# EXERCISE:
# Before moving on, you can run this cell multiple times. Notice that each
# time you run the cell, the data-points are slightly different, and so too
# the model fit. Do higher-order fits vary more or less than lower order fits?

#%% Example 3:  Estimate the "test error" for a p-order polynomial model
#  So far, we have only compared our model to the data used to train the
#  model. We now introduce new data and see how our model performs. We call
#  the original dataset the "training" dataset, and the new data the "test"
#  dataset.

# Generate a TRAINING and a TEST dataset with 100 observations each
Train = GenerateData(f, sig_eps, 100)
Test = GenerateData(f, sig_eps, 100)

# Fit a p-th order polynomial model and evaluate the TRAINING error
# Set "p" to a value of your choice, then follow the same procedure as
# before to fit a polynomial model, this time using "Train.t" and "Train.y"
# as your data set. Then, use "predict" to evaluate the model predictions
# at the training data points

p = 4
X_train = preprocessing.PolynomialFeatures(p).fit_transform(Data.t)
# Train the model using the TRAINING data set, then evaluate at the original training data points
mdl = linear_model.LinearRegression().fit(X_train, Train.y)
Train.y_pred = mdl.predict(X_train)

# Calculate the mean squared error (MSE) using
#  "MSE_train = np.mean( (Train.y - Train.y_pred).^2 )"
MSE_Train = np.mean( (Train.y - Train.y_pred)**2 )

# Now calculate the MSE using the test data.
# First evaluate your model at the Test data points "Test.t" without
# retraining your model (notice that you must define your design matrix "X"
# using "Test.t" in order to make the predictions at the new points)
# Then compare your model predictions to the test data points using
#   "MSE_test = mean( (Test.y - Test.y_pred).^2 )"
X_test = preprocessing.PolynomialFeatures(p).fit_transform(Test.t)
Test.y_pred = mdl.predict(X_test)
MSE_Test = np.mean( (Test.y - Test.y_pred)**2 )

print(f"The training MSE is {MSE_Train}, \n"
      f"The testing MSE is  {MSE_Test}")

# %% Example 4:  Estimate the "test error" for a p-order polynomial model using multiple datasets
#  Now, we will fit all of the polynomials from order 1 to order "p_max",
#  and compare their test errors. However, since the test error depends on
#  the dataset used during training and testing, we will repeat the
#  comparison 50 times - that is, we will train and test each polynomial 50
#  times.

np.random.seed(1) # Specify random number generator seed for reproducible results
p_max = 8 # Evaluate all polynomials up to order 8
All_Poly_Fits = np.zeros([len(Fit.t), 50, p_max])
MSE_Train = np.zeros([50, p_max])
MSE_Test = np.zeros([50, p_max])
for i in range(50):
    # Generate a new training dataset and a new test dataset
    Train = GenerateData(f, sig_eps, 100)
    Test = GenerateData(f, sig_eps, 100)
    for p in range(p_max):
        # Estimate the model and the training error
        X_train = preprocessing.PolynomialFeatures(p+1).fit_transform(Train.t)
        mdl = linear_model.LinearRegression().fit(X_train, Train.y)
        MSE_Train[i, p] = np.mean( (Train.y - mdl.predict(X_train))**2 )

        # Estimate the test error without retraining the model
        X_test = preprocessing.PolynomialFeatures(p + 1).fit_transform(Test.t)
        MSE_Test[i, p]= np.mean( (Test.y - mdl.predict(X_test))**2 )

        # Store the current model fit in the 3D array "All_Poly_Fits"
        X_fit = preprocessing.PolynomialFeatures(p + 1).fit_transform(Fit.t)
        All_Poly_Fits[:, i, p] = mdl.predict(X_fit).flatten()

fig, axs = plt.subplots(2,2)

# Plot all of the p = 1 fits, as well as the mean p = 1 fit
p = 1
axs[0,0].plot(Fit.t, All_Poly_Fits[:,:,p-1] , color = '#BBBBBB')
axs[0,0].plot(Fit.t, Fit.f, 'b',
              Fit.t, np.mean(All_Poly_Fits[:,:,p-1], axis = 1), 'r')
axs[0,0].set_xlim(left = -4, right = 4)
axs[0,0].set_ylim(bottom = -4, top = 4)
axs[0,0].set_title('First order polynomial fit')

# Plot all of the p = p_max fits, as well as the mean p = p_max fit
p = p_max
axs[0,1].plot(Fit.t, All_Poly_Fits[:,:,p-1] , color = '#BBBBBB')
axs[0,1].plot(Fit.t, Fit.f, 'b', Fit.t, np.mean(All_Poly_Fits[:,:,p-1], axis = 1), 'r')
axs[0,1].set_xlim(left = -4, right = 4)
axs[0,1].set_ylim(bottom = -4, top = 4)
axs[0,1].set_title(f'{p_max}-order polynomial fit')

# Create a box plot of the training and test errors
axs[1,0].boxplot(MSE_Train)
axs[1,0].plot(np.mean(MSE_Train, axis = 0),'k--',linewidth = 2)
axs[1,0].set_yscale('log')
axs[1,0].set_xlabel('Polynomial order')
axs[1,0].set_ylabel('Mean Squared Error')
axs[1,0].legend(['Training Data MSE'])

axs[1,1].boxplot(MSE_Test)
axs[1,1].plot(np.mean(MSE_Test, axis = 0),'k--',linewidth = 2)
axs[1,1].set_yscale('log')
axs[1,1].set_xlabel('Polynomial order')
axs[1,1].legend(['Testing Data MSE'])

#%% Example 5:  Estimate "test error" for p-order polynomial model using cross-validation
# Unfortunately, we typically don't have additional datasets to use as test
# data, or we want to use all of our training data as effectively as
# possible. To address this, we can use cross-validation

# Generate data
fig, axs = plt.subplots(2)
AllData = GenerateData(f, sig_eps, 100, ax_lims = [-4, 4, -4, 4], ax = axs[1])
# Set polynomial order and create design matrix using training data,
# then estimate the test error using cross-validation and fit the model using the training data

p = 5
X = preprocessing.PolynomialFeatures(p).fit_transform(AllData.t)
mdl = linear_model.LinearRegression().fit(X, AllData.y)
Error_CV = model_selection.cross_validate(mdl, X, AllData.y, cv = 5, scoring = 'neg_mean_squared_error')

# Evaluate the fitted model at the data points "Fit.t"
X_fit = preprocessing.PolynomialFeatures(p).fit_transform(Fit.t)
Fit.poly = mdl.predict(X_fit)

# Plot the results
axs[0].boxplot(-Error_CV['test_score'])
axs[0].set_ylabel('Mean Squared Error')
axs[0].set_title(f'Cross-validation MSE for {p}th order polynomial')

#axs[1].plot(AllData.t, AllData.y, 'ko')
axs[1].plot(Fit.t, Fit.poly, 'r', Fit.t, Fit.f, 'b', linewidth = 2)
axs[1].legend(['Data','Polynomial fit','True function'])
#axs[1].set_xlim(left = -4, right = 4)
#axs[1].set_ylim(bottom = -4, top = 4)

# EXERCISE:
# With a little effort, you can modify this script to loop over all polynomial orders
# p = 1:p_max as in Example 4, to generate the CV-error estimate for all polynomial orders