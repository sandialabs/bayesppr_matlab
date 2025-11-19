function bm = bpprPCA(X, y, npc, percVar, ncores, center, scale, opts)
% Wrapper to get principal components and call bpprBasis, which then calls bppr function to fit the BPPR model for
% functional (or multivariate) response data.

% xx: matrix of predictors of dimension nxp, where n is the number of training examples and p is
%     the number of inputs (features).
% y: response matrix of dimension nxq, where q is the number of multivariate/functional
%    responses.
% npc: number of principal components to use (integer, optional if percVar is specified).
% percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
%          (if npc=None).
% ncores: number of threads to use when fitting independent BPPR models (integer less than or equal to npc).
% center: whether to center the responses before principal component decomposition (boolean).
% scale: whether to scale the responses before principal component decomposition (boolean).
% opts: optional arguments to bppr function.
% returns object of class bpprBasis, with predict and plot functions.

arguments
    X {mustBeNumeric}
    y {mustBeNumeric}
    npc = NaN;
    percVar = 99.9;
    ncores = 1;
    center = true;
    scale = false;
    opts.n_ridge_mean = 10.0
    opts.n_ridge_max = nan
    opts.n_act_max = nan
    opts.df_spline = 4
    opts.prob_relu = 2/3
    opts.prior_coefs = "zs"
    opts.shape_var_coefs = nan
    opts.scale_var_coefs = nan
    opts.n_dat_min = nan
    opts.scale_proj_dir_prop = nan
    opts.adapt_act_feat = true
    opts.w_n_act = nan
    opts.w_feat = nan
    opts.n_post = 1000
    opts.n_burn = 9000
    opts.n_adapt = 0
    opts.n_thin = 1
    opts.silent = false
end

setup = bpprPCAsetup(y, center, scale);

if isnan(npc)
    cs = cumsum(setup.evals) / sum(setup.evals) * 100;
    npc = find(cs >= percVar, 1);
end

if ncores > npc
    ncores = npc;
end

basis = setup.basis(:, 1:npc);
newy = setup.newy(1:npc, :);
trunc_error = basis * newy - setup.y_scale';

fprintf('Starting bpprPCA with %d components, using %d cores.\n',npc, ncores)

bm = bpprBasis(X, y, basis, newy, setup.y_mean, setup.y_sd, trunc_error, ncores, opts);
