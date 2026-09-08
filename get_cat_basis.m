function basis = get_cat_basis(Xj)
% Basis function for a ridge function whose active features are all categorical

p = size(Xj, 2);
if p > 1
    basis = 1 - prod(1 - Xj, 2);
else
    basis = Xj;
end
