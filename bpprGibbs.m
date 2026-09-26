classdef bpprGibbs < handle
    % BayesPPR as one block of a larger Gibbs sampler.
    %
    % bppr fits a fixed response from start to finish.  When the response is
    % itself a latent variable that another part of a sampler updates between
    % visits (data augmentation), the BayesPPR chain has to carry its state
    % from one visit to the next and be advanced a few steps at a time
    % against whatever the response currently is.  This class does that:
    %
    %   g = bpprGibbs(X, y0, n_keep, 'n_ridge_mean', 2, ...);
    %   for it = 1:n_iter
    %       ...update the latent response y...
    %       g.setResponse(y);        % keep the ridge functions, swap y
    %       g.step(3);               % 3 RJMCMC updates (each also draws s2, coefs)
    %       mu = g.fitted();         % regression function at the training X
    %       s2 = g.s2();             % its error variance
    %       if keep, g.save(); end   % record the state as a posterior draw
    %   end
    %   mod = g.model();             % bpprModel: predict(), plot()
    %
    % Validity.  Each step is the ordinary BayesPPR update: an RJ move on the
    % ridge functions with the coefficients and s2 integrated out, then exact
    % draws of s2 and the coefficients given the ridge functions.  That kernel
    % leaves p(ridge functions, s2, coefs | y) invariant, so interleaving it
    % with updates of y is a valid Gibbs sampler for the joint model.
    %
    % Options are those of bppr that describe the model or the proposals
    % (n_ridge_mean, n_ridge_max, n_act_max, df_spline, prob_relu,
    % prior_coefs, shape_var_coefs, scale_var_coefs, n_dat_min,
    % scale_proj_dir_prop, adapt_act_feat, w_n_act, w_feat) plus the options
    % added for this use:
    %   intercept     include a constant basis function (default true)
    %   center_basis  center ridge basis columns over the training inputs
    %                 (default false).  intercept = false with
    %                 center_basis = true gives a regression function that
    %                 averages to exactly zero over the training inputs.
    %   shape_s2, scale_s2  inverse-gamma prior on s2 (default 0, 0: the
    %                 original improper p(s2) ~ 1/s2)
    %   s2_fixed      hold s2 fixed at this value (default NaN: sample it)
    % n_keep is the number of states save() can record.

    properties
        data
        prior
        specs
        state
        samples
    end

    methods
        function obj = bpprGibbs(X, y, n_keep, options)
            arguments
                X {mustBeNumeric}
                y {mustBeNumeric}
                n_keep (1,1) {mustBePositive, mustBeInteger} = 1
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
                options.intercept = true
                options.center_basis = false
                options.shape_s2 = 0
                options.scale_s2 = 0
                options.s2_fixed = nan
            end

            obj.data = bpprData(X, y(:));
            obj.prior = bpprPrior(options.n_ridge_mean, options.n_ridge_max, ...
                options.n_act_max, options.df_spline, options.prob_relu, ...
                options.prior_coefs, options.shape_var_coefs, ...
                options.scale_var_coefs, options.n_dat_min);
            obj.prior.intercept = options.intercept;
            obj.prior.center_basis = options.center_basis;
            obj.prior.shape_s2 = options.shape_s2;
            obj.prior.scale_s2 = options.scale_s2;
            obj.prior.s2_fixed = options.s2_fixed;
            % n_post = n_keep, no burn-in or adaptation: the calling sampler
            % owns the schedule.
            obj.specs = bpprSpecs(n_keep, 0, 0, 1, options.w_n_act, ...
                options.w_feat, options.adapt_act_feat, options.scale_proj_dir_prop);

            obj.data = obj.data.summarize(obj.prior);
            obj.prior = obj.prior.calibrate(obj.data);
            obj.specs = obj.specs.calibrate(obj.data, obj.prior);
            if strcmpi(obj.prior.prior_coefs, 'flat')
                % calibrate() turns the whole run into adaptation for the flat
                % prior, which never draws s2 or the coefficients.
                error('bpprGibbs:flat', 'bpprGibbs needs prior_coefs = ''zs''.');
            end

            obj.state = bpprState(obj.data, obj.prior, obj.specs);
            obj.state.phase = 'post-burn';
            obj.samples = bpprSamples(obj.prior, obj.specs);
        end

        function setResponse(obj, y)
            % Replace the response, keeping the current ridge functions.
            y = y(:);
            if numel(y) ~= obj.data.n
                error('bpprGibbs:size', 'y must have %d elements.', obj.data.n);
            end
            obj.data.y = y;
            obj.data.ssy = y' * y;
            obj.state.setResponse(obj.data);
        end

        function step(obj, n_steps)
            % Advance the chain by n_steps RJMCMC updates against the
            % current response.
            if nargin < 2, n_steps = 1; end
            for k = 1:n_steps
                obj.state.update(obj.data, obj.prior, obj.specs);
            end
        end

        function mu = fitted(obj)
            % Current regression function at the training inputs (n x 1).
            mu = obj.state.fitted();
        end

        function s2 = s2(obj)
            % Current error variance.
            s2 = obj.state.s2;
        end

        function n = n_ridge(obj)
            n = obj.state.n_ridge;
        end

        function save(obj)
            % Record the current state as the next posterior draw.
            if obj.state.idx > obj.specs.n_keep
                error('bpprGibbs:full', 'All %d draws have been saved.', obj.specs.n_keep);
            end
            obj.samples = obj.samples.writeState(obj.state);
            obj.state.idx = obj.state.idx + 1;
        end

        function mod = model(obj)
            % The saved draws as a bpprModel, for predict() and plot().
            n_saved = obj.state.idx - 1;
            if n_saved < obj.specs.n_keep
                warning('bpprGibbs:partial', ...
                    'Only %d of %d draws saved; predict with mcmc_use = 1:%d.', ...
                    n_saved, obj.specs.n_keep, n_saved);
            end
            mod = bpprModel(obj.data, obj.prior, obj.specs, obj.samples);
        end
    end
end
