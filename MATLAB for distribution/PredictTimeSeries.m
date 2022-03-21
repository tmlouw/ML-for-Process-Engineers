function y = PredictTimeSeries(mdl, TimeData, L)
% Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
%
%  y = PredictTimeSeries(beta, Data, L) accepts a parameter vector "beta", 
%  a time table "TimeData" and a lag order "L" as inputs. 
%
%  The "mdl" object must represents a linear model of the form:
%    y = X*Q*beta
%  and contains the following information
%    mdl.Q: the projection matrix "Q"
%    mdl.beta: the coefficients "beta", with "beta(1)" corresponding to the
%    intercept
%
%  The input "TimeData" must be a timetable containing a column "y" to be
%  predicted as well as an input "u", where "u" may be multi-column
% 
%  y = PredictTimeSeries(Data, L) will forecast the timeseries "y" using the
%  initial values of y(1) to y(L+1) in the "Data" table, as well as the values
%  of the input variables. All subsequent values y(L+2), y(L+3) ... will be
%  calculated using the predicted values of y(k), not the values provided in
%  the "Data" table. This corresponds to forecasting (or simulation) as
%  opposed to "prediction"

LaggedData = CreateLaggedTable(TimeData, L);

y = TimeData.y(1:L+1);
for i = L+2:length(TimeData.t)
    X_current = [LaggedData.u1_lagged(i-L-1,:) ...
                 LaggedData.u2_lagged(i-L-1,:) ...
                 y(i-L-1 : i-1)'   ];
    
	y(i) = X_current*mdl.Q * mdl.beta;
end

end

function LaggedData = CreateLaggedTable(TimeData, L)
% Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
%
% CreateLaggedTable will create a table containing all lagged variables.
%
% The created table will contain a column "y_current", 
% as well as "u_lagged" and "y_lagged". 
%
% The input "TimeData" must be a timetable containing a column "y" to be
% predicted as well as an input "u", where "u" may be multi-column
% 
% The number of lags is given by "L". If L = 0, then only the previous data
% points are used to predict the current time.
%
% A given row in the created table consists of:
%  y(i)   u(i-1) u(i-2) ... u(i-L)   y(i-1) y(i-2) ... y(i-L)

% Create the table with variable "y_current" being the response variable
LaggedData = TimeData(:, 'y');
LaggedData = renamevars(LaggedData, 'y', 'y_current');

% Add the previous datapoint and rename these to "u_lagged" and "y_lagged"
LaggedData = synchronize(LaggedData, lag(TimeData, 1));
LaggedData = renamevars(LaggedData, {'y', 'u1', 'u2'}, {'y_lagged', 'u1_lagged', 'u2_lagged'});

% Add lagged variables one by one, merging newly added lags into the
% columns "u_lagged" and "y_lagged"
for l = 1:L
    LaggedData = synchronize(LaggedData, lag(TimeData, l+1));
    LaggedData = mergevars(LaggedData, {'y_lagged', 'y'}, 'NewVariableName', 'y_lagged');
    LaggedData = mergevars(LaggedData, {'u1_lagged', 'u1'}, 'NewVariableName', 'u1_lagged');
    LaggedData = mergevars(LaggedData, {'u2_lagged', 'u2'}, 'NewVariableName', 'u2_lagged');
end
% Remove any missing data (e.g. the first L+1 data points)
LaggedData = rmmissing(LaggedData);
end

