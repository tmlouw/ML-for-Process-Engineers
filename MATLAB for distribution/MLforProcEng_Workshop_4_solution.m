%% Machine Learning for Process Engineers - Workshop (Part 4)
%  Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
% 
%  The context of the exercise is an industrial penicillin fermenter. A simulation 
%  of the fermenter has been developed and is described in detail at 
%  http://www.industrialpenicillinsimulation.com/, as well as the paper 
%     Goldrick, Duran-Villalobos, Jankauskas, Lovett, Farid, Lennox, (2019), 
%       Modern day monitoring and control challenges outlined on an industrial-scale 
%       benchmark fermentation process, Comp. & Chem. Eng., 130, 
%       https://doi.org/10.1016/j.compchemeng.2019.05.037.
%
%  The flowrate of phenylacetic acid (PAA) is a manipulated variable of the
%  system. The goal of this exercise is to develop a machine learning model to
%  predict the concentration of PAA using Raman spectroscopy (spec) data, 
%  for use as a soft-sensor in control.
%
%  The spec data records a Raman spectrum over a range of wavenumbers from
%  200 to 2400 cm^(-1), with a resolution of 1 cm^(-1). Each measured
%  spectrum is accompanied by an offline measurement of PAA concentration,
%  which is only available once every 12 hrs. If an ML model can be
%  developed to correlate PAA tot the spec data, then online measurement of
%  PAA will be possible, which will greatly improve control
%
%  The data is provided in "PAA_Raman_data.csv", which has the columns:
%  Time | PAA | x2400 | x2399 | x2398 | ... | x204 | x203
%  The first two columns correspond to the offline measurement time and PAA
%  concentration, and the remaining columns represent the corresponding PAA
%  spectrum
%
%  Use "readtable('PAA_Raman_data', 'ReadVariableNames', true);" to ensure
%  the function "readtable" recognises the first row as containing column
%  headings. 
%  This is a very large table: DO NOT TRY TO DISPLAY IT IN THE COMMAND WINDOW

%%
clc
clear

Data = readtable('PAA_Raman_data', 'ReadVariableNames', true);

N = 80;
for n = 1 : N
    Error_CV(:,n) = crossval(@(Train, Test) EstimateError(Train, Test, n), Data);
    fprintf('Regressed for %0.f components (max %0.f) \n', n, N);
end

%%
clc

subplot(2,1,1)
hold off
boxplot(Error_CV)
hold on
plot(1:N, mean(Error_CV),'r','LineWidth',2)
hold off
a = gca(); a.YScale = 'log';
idx = [1 5:5:length(a.XTick)];
a.XTick = a.XTick(idx);
a.XTickLabel = {a.XTickLabel{idx}};

n = 35;

NT = round(height(Data)/5);
Data = Data(randperm(height(Data)), :);
X = table2array(Data(1:NT, 3:end));
y = Data.PAA(1:NT);
Q = plsregress(X, y, n);
mdl = fitlm(X*Q, y, 'Intercept', false);
beta = mdl.Coefficients.Estimate;

X = table2array(Data(:, 3:end));
predict_y = X*Q*beta;

subplot(2,1,2)
mx = max(max([Data.PAA predict_y]));
mn = min(min([Data.PAA predict_y]));
hold off
fill([mn mx mx mn], [0.75*mn 0.75*mx 1.25*mx 1.25*mn], 0.9*[1 1 1]);
hold on
fill([mn mx mx mn], [0.9*mn 0.9*mx 1.1*mx 1.1*mn], 0.6*[1 1 1]);
plot(Data.PAA(1:NT), predict_y(1:NT), 'b.', ...
     Data.PAA(NT+1:end), predict_y(NT+1:end), 'r.', ...
     [mn mx], [mn mx], 'k--');
a = gca(); a.XScale = 'log'; a.YScale = 'log';
axis([mn mx mn mx])
axis square

%%
function MSE = EstimateError(Train, Test, N)

X = table2array(Train(:, 3:end));
y = Train.PAA;

Q = plsregress(X, y, N);
mdl = fitlm(X*Q, y, 'Intercept', false);
beta = mdl.Coefficients.Estimate;

X = table2array(Test(:, 3:end));
y = Test.PAA;

Predict_y = X*Q*beta;

MSE = mean( (Predict_y - y).^2 );

end