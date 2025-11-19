classdef bpprSpecs
    % Structure to store data

    properties
        n_post
        n_burn
        n_adapt
        n_thin
        n_keep
        n_pre
        n_draws
        n_data
        w_n_act
        w_feat
        adapt_act_feat
        proj_dir_prop_prec
    end

    methods
        function obj = bpprSpecs(n_post, n_burn, n_adapt, n_thin, w_n_act, w_feat, adapt_act_feat, scale_proj_dir_prop)
            if (n_thin > n_post)
                error('n_thin > n_post. No posterior samples will be obtained.')
            end
            obj.n_post = n_post;
            obj.n_burn = n_burn;
            obj.n_adapt = n_adapt;
            obj.n_thin = n_thin;
            obj.n_post = obj.n_post - mod(obj.n_post, obj.n_thin);
            obj.n_keep = floor(obj.n_post/obj.n_thin);
            obj.n_pre = obj.n_adapt + obj.n_burn;
            obj.n_draws = obj.n_pre + obj.n_post;

            if isnan(w_n_act)
                obj.w_n_act = nan;
            else
                obj.w_n_act = w_n_act;
            end
            if isnan(w_feat)
                obj.w_feat = nan;
            else
                obj.w_feat = w_feat;
            end
            obj.adapt_act_feat = adapt_act_feat;

            if isnan(scale_proj_dir_prop)
                obj.proj_dir_prop_prec = 1000.0;
            else
                if (scale_proj_dir_prop > 0 && scale_proj_dir_prop <= 1)
                    error('scale_proj_dir_prop must be in (0, 1]')
                end
                inv_scale_proj_dir_prop = 1/scale_proj_dir_prop;
                obj.proj_dir_prop_prec = (inv_scale_proj_dir_prop - 1) + sqrt(inv_scale_proj_dir_prop * (inv_scale_proj_dir_prop - 1));
            end
        end

        function obj = calibrate(obj, data, prior)
            for j=1:length(data.feat_type)
                if strcmpi(data.feat_type(j),'')
                    obj.w_feat(j) = 0.0;
                end
            end

            if strcmpi(prior.prior_coefs,'flat')
                obj.n_data = obj.n_pre;
                obj.n_burn = 0;
            end

            if isnan(obj.w_n_act)
                obj.w_n_act = ones(prior.n_act_max,1);
            end

            if isnan(obj.w_feat)
                obj.w_feat = ones(data.p,1);
            end
        end

    end
end
