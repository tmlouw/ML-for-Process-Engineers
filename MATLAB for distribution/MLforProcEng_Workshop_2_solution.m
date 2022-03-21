%% Control Conference Africa 2021 - Machine Learning Workshop (Part 2)
%  This script is used as basis for the second part of the ML workshop at CCA2021.
%  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
%  The functions "GenerateData" and "CreateGaussDesignMatrix" 
%  are required to run all the scripts

%% Initialize
clc
clear

% Initialize the mean function and create the "Fit" data table
% (see CCA2021_Workshop_1.m for details)
f = @(t) 6*exp(-t.^2) .* sin(t);
sig_eps = 0.2;

t = linspace(-4,4)';
Fit = table(t);
Fit.f = f(t);

%% Example 6:  Use feature selection to identify a reduced-order polynomial model
% The code below has been copied from example 2 exactly, then trimmed down
% for brevity, and the maximum model order was set to p = 50
clf
AllData = GenerateData(f, sig_eps, 100, true, [-4 4 -4 4], "on");

% Set the maximum polynomial model order
p = 50;
X = x2fx(AllData.t, (1:p)');

% Fit the linear model but replace "fitlm" with "stepwiselm"
% Display the model properties using "disp"
mdl = stepwiselm(X, AllData.y);
disp(mdl)

X = x2fx(Fit.t, (1:p)'); 
Fit.poly = predict(mdl, X);
plot(Fit.t, Fit.poly,'r', Fit.t, Fit.f, 'b', 'LineWidth',2);

% EXERCISE:
% What do you notice about the order of the retained polynomial terms?

%% Example 7:  Use an alternative model using "Gaussian" radial basis functions (G-RBFs)
%  We will now use a different model, where we assume 
%    y = sum( theta_j * exp( -(t - c_j)^2 / s )
%  and theta_j are the parameters to be learnt. The centroid positions
%  "c_j" and shape parameters "s" are pre-specified
%  Since "y" is linear in the parameters theta, this is still a linear
%  regression problem.
%  We refer to each basis function as a "Radial Basis Function" (RBF)
%  as the value of the function depends only on the distance to the
%  centroid, |x - c_j|. Further, our RBFs have a Gaussian shape, so we
%  refer to them as G-RBFs.
clf
Data = GenerateData(f, sig_eps, 100,true,[-4 4 -4 4],'on');

% We use 10 G-RBFs with centroids equally spaced between -3 and 3
% The design matrix is created using the custom function
% "CreateGaussDesignMatrix". 
% We will use the defaul shape factor throughout
% Type "help CreateGaussDesignMatrix" for more info
c = linspace(-3, 3, 10);      
X_train = CreateGaussDesignMatrix(Data.t, c);

% Plot the G-RBFs, evaluated at the data points "t"
plot(Data.t, X_train,'.', Data.t, 0*Data.t+1.2, 'k|')           

% Fit the model using linear regression
mdl = fitlm(X_train, Data.y);
disp(mdl)

% Evaluate the function at the equally spaced "Fit.t" points
% and plot the function
X_fit = CreateGaussDesignMatrix(Fit.t, c);
Fit.RBF = predict(mdl, X_fit);
plot(Fit.t, Fit.RBF, 'r', Fit.t, Fit.f, 'b', 'LineWidth', 2);

%% Example 8:  Regularize the G-RBF model using ridge regression
%  The RBFs tend to "fit-to-noise". We can reduce this overfitting by
%  introducting bias to the model using ridge regression. Here, we are
%  minimizing the loss function 
%    J = sum( (y_data - y_predicted)^2 ) + lambda*sum( beta^2 )
%  Where lambda > 0 is the regularisation parameter.
%  If lambda = 0, then we have normal sum-of-squares minimization
%  As lambda increases, it penalizes large coefficients more and more
clf
Data = GenerateData(f, sig_eps, 100,true,[-4 4 -4 4],'on');

c = linspace(-3, 3, 10);      
X_train = CreateGaussDesignMatrix(Data.t, c);
% Performs ridge regression using the function "lasso". 
% This function will be explored in more depth in the next example
% Alpha = 1e-6 ensures an L2 penalty is applied
% Lambda is the value of the regularization parameter
% Run the cell for Lambda = 0, Lambda = 0.1, Lambda = 1 and Lambda = 10
[beta, FitInfo] = lasso(X_train, Data.y, 'Lambda', 0.1, 'Alpha', 1e-6);
beta0 = FitInfo.Intercept;

% Plot the data at the points "Fit.t"
X_train = CreateGaussDesignMatrix(Fit.t, c);
Fit.RBF_ridge = beta0 + X_train*beta;
plot(Fit.t, Fit.RBF_ridge, 'r', Fit.t, Fit.f, 'b', 'LineWidth', 2);

%% Example 9:  Regularize the G-RBF model using the "lasso" function with cross-validation
% We often use cross-validation to estimate the test error using different
% values of the regularization parameter, "lambda", and then choose the
% lambda that corresponds to the "smallest" model. This is typically the
% largest "lambda" value that corresponds to an MSE within one standard
% error of the minimum MSE. 
%
% The "lasso" function performs "elastic net regularisation" across a range
% of lambda values provided as input. It will also perform K-fold
% cross-validation to estimate the test error
% If "Alpha" = 1, then "lasso" performs L1 regularisation (lasso), whereas
% if "Alpha" ~ 0, then "lasso" performs L2 regularisation (ridge).
% Type "doc lasso" and "doc Lasso and Elastic Net" for more information

rng(1)
subplot(2,1,2)
Data = GenerateData(f, sig_eps, 100,true,[-4 4 -4 4],'on');

c = linspace(-3, 3, 10);      
X_train = CreateGaussDesignMatrix(Data.t, c);

alpha = 1e-0; % Try 'Alpha' = 1e-3 (gives same results as "ridge") and 'Alpha' = 1
K = 10; % Number of folds in K-fold cross-validation
lambda_vec = logspace(-3,0); % Vector of lambda values to evaluate
[beta, FitInfo] = lasso(X_train, Data.y, 'Alpha', alpha, 'CV', K, 'Lambda', lambda_vec);

% Find the beta values that correspond to the "smallest" model as
% discussed above, and the corresponding MSE
beta_best = [FitInfo.Intercept(FitInfo.Index1SE); beta(:,FitInfo.Index1SE)]
MSE_best = FitInfo.MSE(FitInfo.Index1SE);


% Show boxplots of the CV error as a function of lambda
subplot(2,1,1)
errorbar(FitInfo.Lambda, FitInfo.MSE, FitInfo.SE);

% Identify the lambda value that corresponds to the largest lambda value
% (smallest model) within one standard error of the minimum MSE
hold on
plot(FitInfo.Lambda1SE, MSE_best, 'ro','LineWidth',2);
a = gca(); a.XScale = 'log'; a.YScale = 'log';
hold off
xlabel('\lambda'); 
ylabel('Mean Squared Error'); 
title(['MSE estimated by K-fold CV, \alpha = ', num2str(alpha)]);
legend('MSE', 'Optimal MSE','Location', 'NorthWest');

% Evaluate the best fit model
subplot(2,1,2)
X_train = CreateGaussDesignMatrix(Fit.t, c);
Fit.RBF_lasso = beta_best(1) + X_train*beta_best(2:end);
plot(Fit.t, Fit.RBF_lasso, 'r', Fit.t, Fit.f, 'b', 'LineWidth', 2);
