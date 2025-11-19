function theta = rps(mu, kappa)
% Generate random draw from the power-spherical distribution
d = length(mu);
uhat = -mu;
uhat(1) = uhat(1) + 1;
u = uhat / sqrt(sum(uhat.^2));

b = (d - 1)/2;
a = b + kappa;
z = betarnd(a, b);

y = zeros(d,1);
y(1) = 2*z - 1;
temp = randn(1,d-1);
v = temp ./ sqrt(sum(temp.^2));
y(2:end) = sqrt(1-y(1)^2) .* v;

u = u(:);
uy = u' * y;

theta = y - 2 .* u .* uy;

