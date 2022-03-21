%%
clc
clear
clf

X = [-0.2 0.6; -0.5 -0.5; 0.5 0];

subplot(1,3,1)
plot([zeros(3,1) X(:,1)]', [zeros(3,1) X(:,2)]','-o',[0 0], [0 0], 'k.', ...
     'LineWidth',2, 'MarkerFaceColor','auto');

axis([-1 1 -1 1])
axis square

theta = 30 * pi/180;
Q = [cos(theta) sin(theta); -sin(theta) cos(theta)];

subplot(1,3,2)
X2 = X*Q;
plot([zeros(3,1) X2(:,1)]', [zeros(3,1) X2(:,2)]','-o',[0 0], [0 0], 'k.', ...
     'LineWidth',2, 'MarkerFaceColor','auto');
hold on

set(gca,'ColorOrderIndex',1)
plot([zeros(3,1) X(:,1)]', [zeros(3,1) X(:,2)]','--o',...
     'LineWidth',0.8);

axis([-1 1 -1 1])
axis square

subplot(1,3,3)
X2 = X*Q;
plot([zeros(3,1) X2(:,1)]', [zeros(3,1) X2(:,2)]','-o',[0 0], [0 0], 'k.', ...
     'LineWidth',2, 'MarkerFaceColor','auto');
hold on

set(gca,'ColorOrderIndex',1)
plot([zeros(3,1) X(:,1)]', [zeros(3,1) X(:,2)]','--o',...
     'LineWidth',0.8);

plot([-1 1], [0 0], 'k')
set(gca,'ColorOrderIndex',1)
plot([zeros(3,1) X2(:,1) X2(:,1)]', [zeros(3,1) zeros(3,1) X2(:,2)]','--|',...
     'LineWidth',0.8, 'MarkerFaceColor','auto');
set(gca,'ColorOrderIndex',1)
plot([zeros(3,1) X2(:,1)]', 0*[zeros(3,1) X2(:,2)]','-|',...
     'LineWidth',2, 'MarkerFaceColor','auto');

axis([-1 1 -1 1])
axis square

%%

%% Example 12: Rotating a dataset with 3 dimensions
clear theta
N = 5000;
X = randn(N, 3);
X(:,2) = X(:,1) + 0.1*X(:,2);

XL = pca(X);

subplot(3,3,[1 2 4 5 7 8])
plot3(X(:,1), X(:,2), X(:,3),'.');
xlabel('X_1'); ylabel('X_2'); zlabel('X_3'); 
axis([-4 4 -4 4 -4 4])

subplot(3,3,3)
Q = XL(:,1);
X2 = X*Q;
plot(X2(:,1), 0*X2(:,1),'.');
axis([-4 4 -4 4])

subplot(3,3,6)
Q = XL(:,[1 2]);
X2 = X*Q;
plot(X2(:,1), X2(:,2),'.');
axis([-4 4 -4 4])

subplot(3,3,9)
X2 = X*XL;
plot3(X2(:,1), X2(:,2), X2(:,3),'.');
axis([-4 4 -4 4 -4 4])

%%
clc
clear
clf

xv = linspace(-4,4);
f = @(x) 1/(sqrt(2*pi)) * exp(-x.^2);

N = 20;
x = 2 + 0.4*randn(N,1);
z = (x-2)/0.4;

plot(xv, f(xv), 'LineWidth',2);
hold on
plot(x', 0*z' - 0.15, '|','MarkerSize',5,'LineWidth',1); 
plot(x'-2, 0*z' - 0.1, '|','MarkerSize',5,'LineWidth',1); 
plot(z', 0*z' - 0.05, 'k|','MarkerSize',5,'LineWidth',1); 
plot([z z]', [zeros(N,1) f(z)]', 'k-o','MarkerSize',3,'MarkerFaceColor','k'); 
hold off
xlabel('x');
a = gca(); a.YTick = [];
axis([-4 4 -0.2 0.5])

%%
clc
clear
clf

[xv, yv] = meshgrid(linspace(-4,4));
f = @(x, y) 1/(2*pi) * exp( -(x.^2 + y.^2) );

N = 50;
mu = [0 2];
sig = [0.3 0.3; 0.3 0.4];
x = mvnrnd(mu, sig, N);
[Q,L] = eigs(sig,2);

subplot(4,1,1)
contour(xv, yv, f(xv, yv), logspace(-7, -1, 10),'k','LineWidth',0.9);
hold on
set(gca,'ColorOrderIndex',1)
plot(x(:,1), x(:,2),'o','LineWidth',2,'MarkerSize',2)
axis([-4 4 -4 4])
axis square

subplot(4,1,2)
z = (x - mu);
contour(xv, yv, f(xv, yv), logspace(-7, -1, 10),'k','LineWidth',0.9);
hold on
set(gca,'ColorOrderIndex',2)
plot(z(:,1), z(:,2),'o','LineWidth',2,'MarkerSize',2)
axis([-4 4 -4 4])
axis square

subplot(4,1,3)
z = (x - mu)*Q;
contour(xv, yv, f(xv, yv), logspace(-7, -1, 10),'k','LineWidth',0.9);
hold on
set(gca,'ColorOrderIndex',3)
plot(z(:,1), z(:,2),'o','LineWidth',2,'MarkerSize',2)
axis([-4 4 -4 4])
axis square

subplot(4,1,4)
z = (x - mu)*Q*diag(diag(L).^(-1/2));
contour(xv, yv, f(xv, yv), logspace(-7, -1, 10),'k','LineWidth',0.9);
hold on
plot(z(:,1), z(:,2),'ko','MarkerFaceColor','k','MarkerSize',3.5)
axis([-4 4 -4 4])
axis square
%%
hold on
plot(x', 0*z' - 0.15, '|','MarkerSize',5,'LineWidth',1); 
plot(x'-2, 0*z' - 0.1, '|','MarkerSize',5,'LineWidth',1); 
plot(z', 0*z' - 0.05, 'k|','MarkerSize',5,'LineWidth',1); 
plot([z z]', [zeros(N,1) f(z)]', 'k-o','MarkerSize',3,'MarkerFaceColor','k'); 
hold off
xlabel('x');
a = gca(); a.YTick = [];
axis([-4 4 -0.2 0.5])