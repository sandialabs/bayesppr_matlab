classdef PoolBPPR
    % class for parallel BASS

    properties
        x
        y
        opts
    end

    methods
        function obj = PoolBPPR(x, y, opts)
            obj.x = x;
            obj.y = y;
            obj.opts = opts;
            obj.opts.verbose = false;
        end

        function bm = rowbppr(obj, i)
            bm = bppr(obj.x, obj.y(i,:)', opts.n_ridge_mean, opts.n_ridge_max, opts.n_act_max, ...
                        opts.df_spline, opts.prob_relu, opts.prior_coefs, opts.shape_var_coefs, opts.scale_var_coefs, ...
                        opts.n_dat_min, opts.scale_proj_dir_prop, opts.adapt_act_feat, opts.w_n_act, opts.w_feat, ...
                        opts.n_post, opts.n_burn, opts.n_adapt, opts.n_thin, opts.silent);
        end

        function out = fit(obj, ncores, nrow_y)
            if isempty(gcp('nocreate'))
                parpool(ncores);
            end
            out = cell(1,nrow_y);
            bar = ProgressBar(nrow_y, ...
                'IsParallel', true, ...
                'WorkerDirectory', pwd, ...
                'Title', 'Running MCMC Chains' ...
                );
            bar.setup([], [], []);
            parfor i = 1:nrow_y
                out{i} = obj.rowbppr(i);
                updateParallel([], pwd);
            end
            bar.release();
        end
    end
end
