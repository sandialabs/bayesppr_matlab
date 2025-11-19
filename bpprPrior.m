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
                warning("n_dat_min too small. If n_dat_min was set by default , df_spline is large compared to the sample size. Setting nDatMin = df_spline + 1")
                obj.n_dat_min = obj.df_spline+1;
            end
            obj.p_dat_max = 1.0 - obj.n_dat_min / data.n;

            if isnan(obj.n_act_max)
                n_cat = 0;
                obj.n_act_max = min(3, data.p - n_cat) + min(3, ceil(n_cat/2));
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
    end
end
