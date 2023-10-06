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
% This script requires the custom function "CreateLaggedDesignMatrix.m" 
% and "PredictTimeSeries.m" to run

%% Initialize
%  Clear all variables and load the process dataset
clc
clear
clf
load ProcessData

% Display the first five rows of the dataset
disp(Data(1:5,:))

% Load the data into an array of predictors and a response vector to 
% be used for training. Set the number of lagged variables "L"
% Only use the first 10% of data for training the subsequent models
L = 10;
f = 0.1;
[X, y] = CreateLaggedDesignMatrix(Data, L, f);


%% Example 10: Linear model fit to timeseries data
% Fit a linear model without regularization
mdl = fitlm(X, y, 'Intercept',false);
linear_mdl.Q = 1;
linear_mdl.beta = mdl.Coefficients.Estimate;
y_linear = PredictTimeSeries(linear_mdl, Data, L);

% Plot and compare the results
clf
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
ylim([-0.4 0.4])
legend('Training data', 'No regularisation');


%%
% Fit a linear model with ridge regression and call the prediction
% "y_ridge"
ridge_mdl.Q = 1;
ridge_mdl.beta = lasso(X, y, 'Alpha', 1e-6, 'Lambda', 0.1);
y_ridge = PredictTimeSeries(ridge_mdl, Data, L);

% Plot and compare the results
clf
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
     Data.t, y_ridge, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
ylim([-0.4 0.4])
legend('Training data', 'No regularisation', 'Ridge regression');

% EXERCISE: try fitting a linear model without regularization by using the
% "lasso" function and setting Lambda = 0. Do you get the same results?
% Why or why not?

%% Example 11: Use PCA regression to predict time series
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
     Data.t, y_ridge, ...
     Data.t, y_PCA, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
 
ylim([-0.4 0.4])
legend('Training data','No regularization', 'Ridge', 'PCA');

% EXERCISE: You can obtain the same information (loadings, explained) using
% the "eigs" function, since the loadings are the eigenvectors of the
% covariance matrix X'*X. How would you obtain "explained"? Recall that the
% eigenvalues give the variance in each principal component direction.

%% Example 12: Use PLS regression to predict time series
%  Fit a model using PLS regression
%  Obtain the PLS loadings and the fraction variance explained
%  The "plsregress" function provides many more outputs. 
%  Type "doc plsregress" for more info
[loadings, ~, ~, ~, ~, explained]  = plsregress(X, y,4);

% Plot the variance explained as a function of number of components
clf
subplot(2,1,1)
bar(explained);
xlabel('Number of principal components');
ylabel('% variance explained');

% Fit the linear model to the reduced set of predictors X*Q and call the
% prediction "y_PLS"
PLS_mdl.Q = loadings(:, 1:4);
T = X*PLS_mdl.Q;
mdl = fitlm(T, y, 'Intercept', false);
PLS_mdl.beta = mdl.Coefficients.Estimate;

% Simulate the model response
y_PLS = PredictTimeSeries(PLS_mdl, Data, L);

% Plot and compare the results
subplot(2,1,2)
fill([0 Data.t(round(f*length(Data.t))) Data.t(round(f*length(Data.t))) 0], ...
     [-0.4 -0.4 0.4 0.4], [0.9 0.9 1],'LineStyle','none');
hold on
plot(Data.t, y_linear, ...
     Data.t, y_ridge, ...
     Data.t, y_PCA, ...
     Data.t, y_PLS, ...
     Data.t, Data.y, 'k.', ...
     'LineWidth',2);
 
ylim([-0.4 0.4])
legend('Training data','No regularization', 'Ridge', 'PCA', 'PLS');