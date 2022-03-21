%% Control Conference Africa 2021 - Machine Learning Workshop (Part 4)
%  This script is used as basis for the fourth part of the ML workshop at CCA2021.
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


