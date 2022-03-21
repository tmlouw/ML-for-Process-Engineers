function MSE = EstimateError(Train, Test)
% Estimate_Error(Train, Test) will train a linear model using the
%  "Train" table, with columns "X" (predictors) and "y" (responses), 
%  such that y = X*theta.
% 
% Estimate_Error(Train, Test) will then calculate the Mean Squared Error (MSE) 
%  by evaluating the model on the test data predictions "Test.X", 
%  and compare it to the test responses "Test.y"

mdl = fitlm(Train.X, Train.y);
y_pred = predict(mdl, Test.X);
MSE = mean( (Test.y - y_pred).^2 );      
end