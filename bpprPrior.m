classdef bpprPrior
    % Structure to store prior

    properties
        n_ridge_mean
        n_ridge_max
        n_act_max
        df_spline
        prob_relu
        knot_quants
        shape_var_coefs
        scale_var_coefs
        prior_coefs
        n_dat_min
        p_dat_max
        proj_dir_mn
        % Options for fitting as one block of a larger Gibbs sampler (see
        % bpprGibbs).  The defaults reproduce the original model.
        intercept = true        % include a constant basis function
        center_basis = false    % center each ridge basis column over the training inputs
        shape_s2 = 0            % inverse-gamma prior on s2; (0, 0) is the
        scale_s2 = 0            %   original improper p(s2) ~ 1/s2
        s2_fixed = nan          % hold s2 at this value instead of sampling it
    end

    methods
        function obj = bpprPrior(n_ridge_mean, n_ridge_max, n_act_max, df_spline, prob_relu, prior_coefs, shape_var_coefs, scale_var_coefs, n_dat_min)
            obj.n_ridge_mean = n_ridge_mean;
            obj.n_ridge_max = n_ridge_max;
            obj.n_act_max = n_act_max;
            obj.df_spline = df_spline;
            obj.prob_relu = prob_relu;
            obj.knot_quants = linspace(0, 1, obj.df_spline+1);
            obj.shape_var_coefs = shape_var_coefs;
            obj.scale_var_coefs = scale_var_coefs;
            obj.prior_coefs = prior_coefs;
            obj.n_dat_min = n_dat_min;
        end

        function obj = calibrate(obj, data)
            if strcmpi(obj.prior_coefs,'zs')
                if isnan(obj.shape_var_coefs)
                    obj.shape_var_coefs = 0.5;
                end
                if isnan(obj.scale_var_coefs)
                    obj.scale_var_coefs = data.n/2;
                end
            end

            if isnan(obj.n_dat_min)
                obj.n_dat_min = min(20, 0.1*data.n);
            end
            if obj.n_dat_min <= obj.df_spline
                warning("n_dat_min too small. If n_dat_min was set by default, df_spline is large compared to the sample size. Setting n_dat_min = df_spline + 1")
                obj.n_dat_min = obj.df_spline+1;
            end
            obj.p_dat_max = 1.0 - obj.n_dat_min / data.n;

            if isnan(obj.n_act_max)
                n_cat = sum(data.feat_type == "cat");
                obj.n_act_max = min(3, data.p - n_cat) + min(3, ceil(n_cat/2));
            end

            if ~obj.intercept && ~obj.center_basis
                warning(['intercept = false without center_basis = true: the ridge ' ...
                    'functions must then carry the mean of y themselves.'])
            end
            if obj.shape_s2 < 0 || obj.scale_s2 < 0
                error('shape_s2 and scale_s2 must be nonnegative')
            end
            if ~isnan(obj.s2_fixed) && ~(obj.s2_fixed > 0)
                error('s2_fixed must be positive (or NaN to sample s2)')
            end

            obj.proj_dir_mn = cell(obj.n_act_max, 1);
            for a = 1:(obj.n_act_max)
                obj.proj_dir_mn{a} = repelem(1/sqrt(a), a);
            end

            if isnan(obj.n_ridge_max)
                obj.n_ridge_max = min(150, floor(data.n/obj.df_spline)-2);
            end

            if (obj.n_ridge_max <= 0)
                error('n_ridge_max <= 0. If n_ridge_max was set by default, df_spline is too large compared to the sample size.')
            end
        end

        function ll = log_lik_sse(obj, sse, n, s2)
            % Log marginal likelihood of the response given the basis, up to a
            % constant that is the same for every basis, as a function of
            % sse = y'y - c*qf.  The coefficients are integrated out under the
            % prior in prior_coefs; the -n_basis/2*log(1+var_coefs) term (zs)
            % or the flat-prior penalty is added by the caller.
            %   s2 integrated out, p(s2) ~ 1/s2:    -n/2 log(sse)
            %   s2 integrated out, IG(shape, scale): -(shape+n/2) log(scale+sse/2)
            %   s2 held fixed:                       -sse/(2 s2)
            if nargin < 4, s2 = obj.s2_fixed; end
            if ~isnan(obj.s2_fixed)
                ll = -sse ./ (2 .* s2);
            elseif obj.shape_s2 == 0 && obj.scale_s2 == 0
                ll = -n/2 .* log(sse);
            else
                ll = -(obj.shape_s2 + n/2) .* log(obj.scale_s2 + sse/2);
            end
        end
    end
end
