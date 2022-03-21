function X = CreateGaussDesignMatrix(t, centroids, s)
% Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
% CreateGaussDesignMatrix(t, c, s) will create a design matrix "X"
%  using predictor data vector "t", such that the i'th column in "X" will
%  provide the values exp( -( t - c(i) )^2 / s)
% The input arguments are:
%  t: predictor data values
%  c: list of Gaussian radial basis function (RBF) centroids
%  s: a shape factor. By default, the shape factor "s" is the squareroot of the mean 
%  of the space between the centroids

if nargin < 3
    s = sqrt(mean(diff(centroids)));
end

X = [];

for c = centroids
    X = [X exp( -(t-c).^2/s)];
end
   
end