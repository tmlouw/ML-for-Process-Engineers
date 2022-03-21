%% Machine Learning for Process Engineers - Workshop (Part 3)
%  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
% 
% We move to a new example from system identification:  
% We have linear state space system with two inputs u1 and u2, 
% and one output y. 
% We assume a general linear form:
%  y(k+1) =   a_0*y(k)  + a_1*y(k-1)   + a_2*y(k-2)  ... + a_L*y(k-L)
%           + b_0*u1(k) + b_1*u1(k-1)  + b_2*u1(k-2) ... + b_L*u1(k-L)
%           + c_0*u2(k) + c_1*u2(k-1)  + c_2*u2(k-2) ... + c_L*u2(k-L)
%
% The output therefor depends on LAGGED data at sample times (k-l) 
% for l = 0, 1, 2, 3... L
%
% Time series data is contained in "ProcessData.mat", which is required to
% run this script.
% This script requires the custom function "CreateLaggedDesignMatrix.m" to run

%% Initialize
%  Clear all variables and load the process dataset
clc
clear
clf
load ProcessData

% Display the first five rows of the dataset
disp(Data(1:5,:))

% Prepare the design matrix "X"
% Set the maximum number of lagged variables "L" and create the
% corresponding design matrix, then plot the correlation matrix of X
L = 10;
X = CreateLaggedDesignMatrix(Data, L);
heatmap(corr(X));

% Load the data into an array of predictors and a response vector to 
% be used for training. 
% Only use the first 10% of data for training the subsequent models
f = 0.1;
[X, y] = CreateLaggedDesignMatrix(Data, L, f);

%% Example 10: Linear model fit to timeseries data
% Fit a linear model without regularization
mdl = fitlm(X, y, 'Intercept',false);
linear_mdl.Q = 1;
linear_mdl.beta = mdl.Coefficients.Estimate;
y_linear = PredictTimeSeries(linear_mdl, Data, L);

% Fit a linear model with ridge regression and call the prediction
% "y_ridge"

% Plot and compare the results
clf
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
...<Delete the "..." and this comment to add "y_ridge" to the plotted data>          Data.t, y_ridge, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
ylim([-0.4 0.4])
legend('Training data', 'No regularisation', 'Ridge regression');

% EXERCISE: try fitting a linear model without regularization by using the
% "lasso" function and setting Lambda = 0. Do you get the same results?
% Why or why not?

%% Example 11: Rotating a dataset with 3 dimensions
% Create a 3D dataset with highly correlated variables
% to visualize and rotate
N = 5000;
Example_X = randn(N, 3);
X(:,2) = X(:,1) + 0.1*X(:,2);

clf
plot3(Example_X(:,1), Example_X(:,2), Example_X(:,3),'.');
xlabel('X_1'); ylabel('X_2'); zlabel('X_3'); 
%% Example 12: Use PCA regression to predict time series
%  Fit a model using PCA regression
%  Obtain the PCA loadings and the fraction variance explained
[loadings, ~, ~, ~, explained]  = pca(X, 'NumComponents',20);

% Plot the variance explained as a function of number of components
clf
subplot(2,1,1)
bar(explained);
xlabel('Number of principal components');
ylabel('% variance explained');

% Fit the linear model to the reduced set of predictors X*Q
PCA_mdl.Q = loadings(:, 1:4);
T = X*PCA_mdl.Q;
mdl = fitlm(T, y, 'Intercept', false);
PCA_mdl.beta = mdl.Coefficients.Estimate;

% Simulate the model response
y_PCA = PredictTimeSeries(PCA_mdl, Data, L);

% Plot and compare the results
subplot(2,1,2)
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
...<Delete the "..." and this comment to add "y_ridge" to the plotted data>          Data.t, y_ridge, ...
     Data.t, y_PCA, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
 
ylim([-0.4 0.4])
legend('Training data','No regularization', 'Ridge', 'PCA');

% EXERCISE: You can obtain the same information (loadings, explained) using
% the "eigs" function, since the loadings are the eigenvectors of the
% covariance matrix X'*X. How would you obtain "explained"? Recall that the
% eigenvalues give the variance in each principal component direction.

%% Example 13: Use PLS regression to predict time series
%  Fit a model using PLS regression
%  Obtain the PLS loadings and the fraction variance explained
%  The "plsregress" function provides many more outputs. 
%  Type "doc plsregress" for more info
[loadings, ~, ~, ~, ~, explained]  = plsregress(X, y,4);

% Plot the variance explained as a function of number of components

% Fit the linear model to the reduced set of predictors X*Q and call the
% prediction "y_PLS"

% Simulate the model response

% Plot and compare the results
subplot(2,1,2)
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
...<Delete the "..." and this comment to add "y_ridge" to the plotted data>          Data.t, y_ridge, ...
     Data.t, y_PCA, ...
...<Delete the "..." and this comment to add "y_PLS" to the plotted data>     Data.t, y_PLS, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
 
ylim([-0.4 0.4])
legend('Training data','No regularization', 'Ridge', 'PCA', 'PLS');