function basis = get_mns_basis(u, knots)
% Make basis funciton using continuous variables

n_knots = length(knots);
df = n_knots - 2;
n = length(u);

basis = zeros(n,df);
basis(:, 1) = relu(u - knots(1));
if df > 1
    n_internal_knots = n_knots - 3;
    r = zeros(n, n_knots-1);
    d = zeros(n, df);
    for k = 2:n_knots
        r(:,k-1) = relu(u-knots(k)).^3;
    end

    for k = 1:df
        d(:,k) = (r(:,k) - r(:, df+1)) ./ (knots(df+2) - knots(k+1));
    end

    for k = 1:n_internal_knots
        basis(:, k+1) = d(:,k) - d(:, n_internal_knots+1);
    end
end
