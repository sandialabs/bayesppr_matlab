% example using multivariate data
clc; clear

f = @(x) 10. .* sin(pi .* linspace(0,1,50) .* x(:,1)) + 20. .* (x(:,2) - .5).^2 + 10 .* x(:,3) + 5. .* x(:,4);

n = 500;
p = 9;
x = rand(n,p) - 0.5;
xx = rand(900,p) - 0.5;
y = f(x);
 
mod = bpprPCA(x, y, NaN, 99.99, 1);
mod.plot()
pred = mod.predict(xx);

tmp = squeeze(mean(mod.predict(x),3))-f(x);
disp(var(tmp(:)))

tmp = squeeze(mean(mod.predict(xx),3))-f(xx);
disp(var(tmp(:)))
