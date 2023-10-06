%%
clc
clear
clf

% y = theta*x
% y1 = y(x1) = y(x = 1)
% y2 = y(x2) = y(x = 2)

sig = 0.1;
sig_theta = 1;
x = [1; 2];
f.pdf_y_l_theta = @(y, theta, x) ( 1 / (sig*sqrt(2*pi)) )*exp( -(y - theta*x).^2 / sig);
f.pdf_theta     = @(theta)  ( 1 / (sig_theta*sqrt(2*pi)) )*exp( -(theta).^2 / sig);
[Theta, Y1, Y2] = meshgrid(linspace(-4,4), linspace(-4,4), linspace(-4,4));

pdf_xyz = f.pdf_theta(Theta).*f.pdf_y_l_theta(Y1, Theta, x(1)).*f.pdf_y_l_theta(Y2, Theta, x(2));
pdf_max = max(pdf_xyz,[],'all');
surfs = logspace(-6, log10(pdf_max - 1e-4), 10);
alphas = linspace(0.2, 0.8, length(surfs))
hold off
for i = 1:length(surfs)
    current = surfs(i);
    [f,v,c] = isosurface(Theta, Y1, Y2, pdf_xyz, current, log10(pdf_xyz));
    patch('Vertices', v, 'Faces', f, 'FaceVertexCData', c,...
          'EdgeColor','none','FaceColor','interp','FaceAlpha',alphas(i))
    hold on
end
% caxis([1e-6 pdf_max])
colorbar
