% example using 1-D data
clc; clear

f = @(x) 10 * sin(pi * x(:,1) .* x(:,2)) + 20. * (x(:,3) - .5).^2 + ...
    10 * x(:,4) + 5. * x(:,5);
n = 500;
p = 10;
X = rand(n,p);
xx = rand(1000,p);
y = f(X) + randn(n,1);

mod = bppr(X, y);
pred = mod.predict(xx);
mod.plot()

disp(var(mean(mod.predict(xx),1)'-f(xx)))
