function [X, y] = CreateLaggedDesignMatrix(TimeData, L, f)
% Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
% CreateLaggedDesignMatrix(TimeData, L) will create a design matrix X
% and a vector of response observations y, given data table "TimeData", 
% with a number of lags "L"
%
% The input "TimeData" must be a timetable containing a column "y" to be
% predicted as well as an input "u", where "u" may be multi-column
% 
% The number of lags is given by "L". If L = 0, then only the previous data
% points are used to predict the current time.
%
% A given row in the concatenation [y X] will give:
%  y(i) u1(i-1) u1(i-2) ... u1(i-L) u2(i-1) u2(i-2) ... u2(i-L)  y(i-1) y(i-2) ... y(i-L)
%
% CreateLaggedDesignMatrix(TimeData, L, f) will create the same arrays, but
% will only use the first fraction "f" of data points. This represents only
% a fraction of data being available for training. 0 < f < 1, default 1

if nargin < 3
    f = 1;
end

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

X = table2array(LaggedData(1:round(f*length(LaggedData.t)), {'u1_lagged', 'u2_lagged', 'y_lagged'})); 
y = table2array(LaggedData(1:round(f*length(LaggedData.t)), 'y_current'));

end
