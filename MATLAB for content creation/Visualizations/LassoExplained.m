%%
clc
clear
clf
rng(1)

N = 50;
x = randn(N, 1);
y = 1.2 + 0.5*randn(N,1);

[mG, cG] = meshgrid(linspace(-2,2,100), linspace(-2,2,100));

SSE = 0*mG;
for i = 1:N
    SSE = SSE + (y(i) - (mG*x(i) + cG)).^2;
end

lambda = 100;
fSSE = @(theta) sum( (y - theta(1)*x - theta(2)).^2 ) + lambda*( sum(abs(theta)) );

theta_0 = [x ones(N,1)]\y;
theta_l = fminunc(fSSE, theta_0);

colormap parula

subplot(2,3,[1 3])
p = plot(x, [x ones(N,1)]*theta_0, ...
         x, [x ones(N,1)]*theta_l, ...
         x, y,'k.', ...
         [-2.5 2.5 0 0 0], [0 0 0 2.5 -2.5],'k');
p(1).LineWidth = 2;
p(2).LineWidth = 2;
p(3).MarkerSize = 8;
axis([-2.5 2.5 -2.5 2.5]); axis square

subplot(2,3,4)
PlotContour(mG, cG, SSE, theta_0, logspace(-1, 3, 50));
xlabel('\beta_0','FontSize',14); ylabel('\beta_1','FontSize',14);
title('Sum of squared error (SSE)')

subplot(2,3,5)
PlotContour(mG, cG, (abs(mG) + abs(cG)), [0; 0], linspace(0,8,20));
xlabel('\beta_0','FontSize',14); ylabel('\beta_1','FontSize',14);
title('L_1 penalty')

subplot(2,3,6)
PlotContour(mG, cG, SSE + lambda*(abs(mG) + abs(cG)), [theta_0 theta_l], logspace(-1, 3, 50));
xlabel('\beta_0','FontSize',14); ylabel('\beta_1','FontSize',14);
title('SSE + L_1 penalty')


%%
clf
s = surf(mG, cG, SSE + lambda*(abs(mG) + abs(cG)));

%%
function PlotContour(mG, cG, Z, theta, v)
hold off
%contour(mG, cG, Z, linspace(min(Z,[],'all'), max(Z,[],'all'),100));
contour(mG, cG, Z, v);
hold on
plot([-2 2 0 0 0], [0 0 0 2 -2],'k', ...
     theta(1,end), theta(2,end), 'r-o','LineWidth',1.2);
hold off
axis square
end