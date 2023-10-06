%%
clc
clear
clf

t = linspace(-4,4);
f = @(t) 6*exp(-t.^2) .* sin(t);

s = 1;
c = [-3; -1; 1; 3];
h_1 = @(t) exp( -(t - c(1)).^2/s );
h_2 = @(t) exp( -(t - c(2)).^2/s );
h_3 = @(t) exp( -(t - c(3)).^2/s );
h_4 = @(t) exp( -(t - c(4)).^2/s );

plot(t, h_1(t), t, h_2(t), t, h_3(t), t, h_4(t), ...
     t, f(t),'b--', 'LineWidth',2)




%%
beta = [0.1; -2.3; 2.3; -0.1];
y_pred =   beta(1)*h_1(t) + beta(2)*h_2(t) ...
         + beta(3)*h_3(t) + beta(4)*h_4(t);

plot(t, beta(1)*h_1(t), t, beta(2)*h_2(t), ...
     t, beta(3)*h_3(t), t, beta(4)*h_4(t),...
     t, y_pred, 'r--', t, f(t), 'b--','LineWidth', 2)













%%
beta = [0; -2.5; 2.5; 0];
X(:,1) = h_1(t);
X(:,2) = h_2(t);
X(:,3) = h_3(t);
X(:,4) = h_4(t);

y_pred = X*beta;
plot(t, beta(1)*X(:,1), t, beta(2)*X(:,2), ...
     t, beta(3)*X(:,3), t, beta(4)*X(:,4),...
     t, y_pred, 'r--', t, f(t), 'b--','LineWidth', 2)









%%
clc
clear
clf

f = @(t) 6*exp(-t.^2) .* sin(t);

N = 100;
t = linspace(-2.5, 2.5).';
y = f(t) + 0.1*randn(N,1);

c = [-3; -1; 1; 3];
s = [1; 1; 1; 1];
b = [-0.1; -2.5; 2.5; -0.1];
p = [c; s; b];

p = lsqnonlin(@(p) funcLSQ(p, t, y), p);
[y_pred, h] = funcNN(p, t);

plot(t, h{1}, t, h{2}, t, h{3}, t, h{4}, ...
     t, y_pred, 'r', t, y, 'k.', 'MarkerSize', 15, 'LineWidth',2)





function [y, h] = funcNN(p, x)
c = p(1:4);
s = p(5:8);
b = p(9:12);
y = 0;
for i = 1:4
    h{i} = b(i)*exp(-(x - c(i)).^2 / s(i));
    y = y + h{i};
end
end

function Err = funcLSQ(p, x, y)
y_pred = funcNN(p, x);
Err = y_pred - y;
end
















