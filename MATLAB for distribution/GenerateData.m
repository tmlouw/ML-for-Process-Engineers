function Output = GenerateData(f, sig_eps, N, plot_data, ax_lims, hold_value)
% Tobi Louw, Dept Process Engineering, Stellenbosch University, 2021
%
% GenerateData(f, sig_eps, N) accepts a function "f" with a single vector valued input argument, 
% noise parameter "sig_eps", and number of observations "N" as input.
%
% GenerateData(f, sig_eps, N) generates a table as outputs with columns:
%   t: an Nx1 vector of predictor values  t ~ N(0, 1)
%   y: an Nx1 vector of response values   y ~ N(f(x), sig_eps)
%
% GenerateData(f, sig_eps, N, plot_data, a_lims, hold_value) accepts three optional arguments:
%   plot_data: a boolean (default: false), if "plot_data" is true, then the data is plotted
%   a_lims: an input to the "axis" command (default: auto) specifying the axis limits
%   hold_value: a string (default: "off") used as input to the "hold" function

if nargin < 4
    plot_data = false;
elseif nargin < 5
    ax_lims = 'auto';
    hold_value = "off";
elseif nargin < 6
    hold_value = "off";
end

t = randn(N, 1);    
y = f(t) + sig_eps*randn(N,1); 

if nargin > 3
    if plot_data
        plot(t, y, 'ko','LineWidth',1.5,'MarkerFaceColor',[0.8 0.8 1],'MarkerSize',4)
        axis(ax_lims);
        hold(hold_value);
    end
end


Output = table(t, y);
        