function model = bppr(X, y, options)
% **Bayesian Projection Pursuit Regression - model fitting**

% This function takes training data, priors, and algorithmic constants and fits a BASS model.  The result is a set of
%posterior samples of the model.  The resulting object has a predict function to generate posterior predictive
%samples.  Default settings of priors and algorithmic parameters should only be changed by users who understand
%the model.

% X: matrix (numpy array) of predictors of dimension nxp, where n is the number of training examples and p is
% the number of inputs (features).
% y: response vector (numpy array) of length n.
% options structure with the following fields
% n_ridge_mean: mean number of ridge functions
% n_ridge_max: max number of ridge functions
% n_act_max: maximum number of activation functions
% df_spline: degree of splines
% prob_relu: relu probability
% prior_coefs: coeficient prior ("zs" or "flat")
% share_var_coefs: shape of prior for coefficients
% scale_var_coefs: scale of prior for coeffficients
% n_data_min: min number of data point
% scale_proj_dir_prop:
% adapt_act_feat: perform adapation in mcmc chain
% w_n_act:
% w_feat:
% n_post: number of posterior samples
% n_burn: number of burn in samples
% n_adapt: number of adaption samples
% n_thin: number of samples to thin
% silent: print out timing information of fitting
% returns an object of class bpprModel, which includes predict and plot functions.

arguments
    X {mustBeNumeric}
    y {mustBeNumeric}
    options.n_ridge_mean = 10.0
    options.n_ridge_max = nan
    options.n_act_max = nan
    options.df_spline = 4
    options.prob_relu = 2/3
    options.prior_coefs = "zs"
    options.shape_var_coefs = nan
    options.scale_var_coefs = nan
    options.n_dat_min = nan
    options.scale_proj_dir_prop = nan
    options.adapt_act_feat = true
    options.w_n_act = nan
    options.w_feat = nan
    options.n_post = 1000
    options.n_burn = 9000
    options.n_adapt = 0
    options.n_thin = 1
    options.silent = false
end

n_ridge_mean = options.n_ridge_mean;
n_ridge_max = options.n_ridge_max;
n_act_max = options.n_act_max;
df_spline = options.df_spline;
prob_relu = options.prob_relu;
prior_coefs = options.prior_coefs;
shape_var_coefs = options.shape_var_coefs;
scale_var_coefs = options.scale_var_coefs;
n_dat_min = options.n_dat_min;
scale_proj_dir_prop = options.scale_proj_dir_prop;
adapt_act_feat = options.adapt_act_feat;
w_n_act = options.w_n_act;
w_feat = options.w_feat;
n_post = options.n_post;
n_burn = options.n_burn;
n_adapt = options.n_adapt;
n_thin = options.n_thin;
silent = options.silent;

data = bpprData(X, y);
prior = bpprPrior(n_ridge_mean, n_ridge_max, n_act_max, df_spline, prob_relu, prior_coefs, shape_var_coefs, scale_var_coefs, n_dat_min);
specs = bpprSpecs(n_post, n_burn, n_adapt, n_thin, w_n_act, w_feat, adapt_act_feat, scale_proj_dir_prop);

% pre-processing
data = data.summarize(prior);
prior = prior.calibrate(data);
specs = specs.calibrate(data, prior);

% initialize the state of the markov chain
state = bpprState(data, prior, specs);

% initialize the posterior samples
samples = bpprSamples(prior, specs, state);

% run MCMC
if specs.n_draws > 1

    if ~silent
        obj = ProgressBar(specs.n_draws, 'Title', 'Running BPPR MCMC');
    end

    for it = 1:specs.n_draws
        if it == specs.n_adapt
            if specs.n_burn > 0 
                state.phase = 'burn';
            else 
                state.phase = 'post-burn';
            end
        end

        if it == specs.n_pre
            state.phase = 'post-burn';
        end

        % update the state
        state = state.update(data, prior, specs);

        if strcmpi(state.phase, 'post-burn') && (mod(it-specs.n_burn, specs.n_thin) == 0)
            % write to samles
            samples = samples.writeState(state);
            state.idx = state.idx + 1;
        end
        if ~silent
            obj.step([], [], []);
        end
    end
    if ~silent
        obj.release()
    end

    model = bpprModel(data, prior, specs, samples);
end
