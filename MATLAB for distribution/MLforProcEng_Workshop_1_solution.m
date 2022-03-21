%% Machine Learning for Process Engineers - Workshop (Part 1)
%  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
%  We will assume that we are measuring samples from
%  a process described by the equation:  y = f(t) + eps, 
%  where "f(t)" is the function described below and "eps" is the 
%  zero mean Gaussian noise with standard deviation sig_eps, 
%  such that eps ~ N(0, sig_eps)
%
%  Samples of our independent variable "t" also follow a standard normal
%  distribution
%
%  The function "GenerateData" is required to run all the scripts

%% Initialize and create an example of the data to be generated
clc
clear

f = @(t) 6*exp(-t.^2) .* sin(t);
sig_eps = 0.2;

% Generate 100 observations and plot data.
% The function "GenerateData" is not a built-in MATLAB function, it was
% created specifically for this workshop
clf
Data = GenerateData(f, sig_eps, 100, true, [-4 4 -4 4], "on");

% Throughout the program, we will plot our approximating function at the
% predictor values "Fit.t". Whereas predictor data points "Data.t" are sampled
% form a standard normal distribution, "Fit.t" are equally spaced points
% from x = -4 to x = 4. "Fit" is saved as a table.
t = linspace(-4,4)';
Fit = table(t);
Fit.f = f(t);

% Plot the noise-free function "f"
plot(Fit.t, Fit.f, 'b','LineWidth', 2);
plot(Fit.t, 0*Fit.t -3.75, 'b|', Data.t, 0*Data.t - 3.25, 'k|');

% Run this cell a few times: the generated data points will be slightly
% different every time.

%% Example 1:  Fit a first order polynomial model
%  In this first example, we will simply fit a straight line through our
%  data. We notice that, for each independently generated data SET, the fit
%  is slightly different

% Generate 100 observations and plot data
clf
Data = GenerateData(f, sig_eps, 100, true, [-4 4 -4 4], "on");

% Fit a first order polynomial model using "fitlm"
% Type "doc fitlm" or "doc LinearModel" for more info
mdl = fitlm(Data.t, Data.y); % Returns a linear model object "mdl". 

% Display your fitted model using "disp"
disp(mdl)

% Evaluated the fitted model at equally spaced points "Fit.t" 
% using the "predict" function.
Fit.linear = predict(mdl, Fit.t);

% Plot the fitted linear model as well as the true mean function "Fit.f"
% The function "GenerateData" above already plotted the datapoints
% and set the axes "hold" property to "on", so any additional plots will
% simply be added to the current axes.
plot(Fit.t, Fit.linear, 'r', Fit.t, Fit.f, 'b', 'LineWidth',2);
legend('Training data','Linear model', 'True function','Location','SouthEast');

% EXERCISE:
% Before moving on, you can run this cell multiple times. Notice that each
% time you run the cell, the data-points are slightly different, and so too
% the model fit.

%% Example 2:  Fit a p-order polynomial model
%  We repeat the exercise above, but this time we fit a p-order polynomial to the data
% Generate a new set of 100 observations
clf
Data = GenerateData(f, sig_eps, 100, true, [-4 4 -4 4], "on");

% Fit a p-th order polynomial model
p = 4;

% Create a "design matrix" X, where each column of "X" corresponds to a "feature"
% and each row of "X" corresponds to an observation. 
% For the polynomial regression, the j'th column
% of the design matrix corresponds to t^j. 
% We can use "x2fx(Data.t, (1:p)')" for this. 
% Type "doc x2fx" for details
X = x2fx(Data.t, (1:p)');

% Let's see what each column of "X" looks like
plot(Data.t, X, '.', ...   
     Data.t, 0*Data.t-3.5, 'k|');
 
% Fit the linear model using "fitlm" and the design matrix "X",
% and display the model properties using "disp"
mdl = fitlm(X, Data.y);
disp(mdl)

% Create a design matrix "X" using "x2fx" and
% the equally spaced vector "Fit.t", then evaluate
% the model at the equally spaced points using "predict"
% Plot your results
X = x2fx(Fit.t, (1:p)'); 
Fit.poly = predict(mdl, X);
plot(Fit.t, Fit.poly,'r', Fit.t, Fit.f, 'b', 'LineWidth',2);

% EXERCISE:
% Before moving on, you can run this cell multiple times. Notice that each
% time you run the cell, the data-points are slightly different, and so too
% the model fit. Do higher-order fits vary more or less than lower order fits?

%% Example 3:  Estimate the "test error" for a p-order polynomial model
%  So far, we have only compared our model to the data used to train the
%  model. We now introduce new data and see how our model performs. We call
%  the original dataset the "training" dataset, and the new data the "test"
%  dataset.

Train = GenerateData(f, sig_eps, 100); % Generate a TRAINING dataset with 100 observations
Test = GenerateData(f, sig_eps, 100);  % Generate a TEST dataset with 100 observations

% Fit a p-th order polynomial model and evaluate the TRAINING error
% Set "p" to a value of your choice, then follow the same procedure as
% before to fit a polynomial model, this time using "Train.t" and "Train.y"
% as your data set. Then, use "predict" to evaluate the model predictions
% at the training data points, e.g. "Train.y_pred = predict(mdl, X)"
% Calculate the mean squared error (MSE) using 
%  "MSE_train = mean( (Train.y - Train.y_pred).^2 )"
p = 4;
X_train = x2fx(Train.t, (1:p)');    
mdl = fitlm(X_train, Train.y);         % Train the model using the TRAINING data set
Train.y_pred = predict(mdl, X_train);  % Evaluate the model predictions at the original TRAINING data points
MSE_Train = mean( (Train.y - Train.y_pred).^2 )

% Now calculate the MSE using the test data.
% First evaluate your model at the Test data points "Test.t" without
% retraining your model (notice that you must define your design matrix "X"
% using "Test.t" in order to make the predictions at the new points)
% Then compare your model predictions to the test data points using
%   "MSE_test = sum( (Test.y - Test.y_pred).^2 )"
X_test = x2fx(Test.t, (1:p)');      % Here, we create the predictor matrix again, but we don't refit the model
Test.y_pred = predict(mdl, X_test); % Evaluate the model predictions at the TEST data-points
MSE_Test = mean( (Test.y - Test.y_pred).^2 )

%% Example 4:  Estimate the "test error" for a p-order polynomial model using multiple datasets
%  Now, we will fit all of the polynomials from order 1 to order "p_max",
%  and compare their test errors. However, since the test error depends on
%  the dataset used during training and testing, we will repeat the
%  comparison 50 times - that is, we will train and test each polynomial 50
%  times.

% Clear the variables from previous runs
clear All_Poly_Fits MSE_Train MSE_Test

rng(1)  % We specify the seed of the random number generator for reproducible results
p_max = 8; % We will evaluate all polynomials up to order 8
All_Poly_Fits = zeros(length(Fit.t), 50, p_max);
for i = 1:50
    % Generate a new training dataset and a new test dataset
    Train = GenerateData(f, sig_eps, 100);
    Test = GenerateData(f, sig_eps, 100);

    for p = 1:p_max
        X_train = x2fx(Train.t, (1:p)');
        mdl = fitlm(X_train, Train.y);
        MSE_Train(i,p) = mean( (Train.y - predict(mdl, X_train)).^2 );  % Estimate the training error

        X_test = x2fx(Test.t, (1:p)');
        MSE_Test(i,p) = mean( (Test.y - predict(mdl, X_test)).^2 );    % Estimate the test error
        
        % Store the current model fit in the 3D array "All_Poly_Fits"
        X_fit = x2fx(Fit.t, (1:p)');
        All_Poly_Fits(:,i,p) = predict(mdl, X_fit);
    end
end

% Plot all of the p = 1 fits, as well as the mean p = 1 fit
clf
subplot(2,2,1);
plot(Fit.t, All_Poly_Fits(:,:,1) ,'Color',0.8*[1 1 1]);
hold on
plot(Fit.t, Fit.f, 'b', Fit.t, mean(All_Poly_Fits(:,:,1),2), 'r');
axis([-4 4 -4 4]);
%axis square
title('First order polynomial fit');

% Plot all of the p = p_max fits, as well as the mean p = p_max fit
subplot(2,2,2);
plot(Fit.t, All_Poly_Fits(:,:,p_max) ,'Color',0.8*[1 1 1]);
hold on
plot(Fit.t, Fit.f, 'b', Fit.t, mean(All_Poly_Fits(:,:,p_max),2), 'r');
axis([-4 4 -4 4]);
%axis square
title([num2str(p_max), 'th order polynomial fit']);

% Create a box plot of the training and test errors
subplot(2,2,3)
boxplot(MSE_Train);
hold on
plot(mean(MSE_Train),'k--','LineWidth', 2);
hold off
a = gca(); 
a.YScale = 'log';
xlabel('Polynomial order');
ylabel('Mean Squared Error');
title('Training Data MSE');

subplot(2,2,4)
boxplot(MSE_Test);
hold on
plot(mean(MSE_Test),'k--','LineWidth', 2);
hold off
a = gca(); 
a.YScale = 'log';
xlabel('Polynomial order');
ylabel('Mean Squared Error');
title('Test Data MSE');

%% Example 5:  Estimate the "test error" for a p-order polynomial model using cross-validation 
% Unfortunately, we typically don't have additional datasets to use as test
% data, or we want to use all of our training data as effectively as
% possible. To address this, we can use cross-validation
% Generate and plot data
clf; subplot(2,1,2)
AllData = GenerateData(f, sig_eps, 100,true,[-4 4 -4 4],'on');

% Set polynomial order and create design matrix using training data, 
% then estimate the test error using cross-validation and fit the model using the training data
p = 5;
AllData.X = x2fx(AllData.t, (1:2:p)'); % This time, we are adding our design matrix to the table "Train"
Error_CV = crossval(@EstimateError, AllData); % Type "doc crossval" for more information
mdl = fitlm(AllData.X, AllData.y);

% Evaluate the fitted model at the data points "Fit.t"
X_fit = x2fx(Fit.t, (1:2:p)');
Fit.poly = predict(mdl, X_fit);

% Plot the results
subplot(2,1,1)
boxplot(Error_CV)
ylabel('Mean Squared Error');
title(['Cross-validation MSE for ', num2str(p),'th order polynomial']);

subplot(2,1,2)
plot(Fit.t, Fit.poly, 'r', Fit.t, Fit.f, 'b', 'LineWidth',2)
hold off
legend('Data','Polynomial fit','True function','Location','SouthEast');

% EXERCISE: 
% With a little effort, you can modify this script to loop over all polynomial orders 
% p = 1:p_max as in Example 4, to generate the CV-error estimate for all polynomial orders



